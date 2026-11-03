"""Optional white-box distillation: CE on teacher text + KL to an OPEN teacher's token distribution.
Teacher and student must share a tokenizer (e.g. Qwen2.5-7B -> Qwen2.5-0.5B/1.5B). Full fine-tune."""
from __future__ import annotations
import json
import time
import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments)
from distill.config import CURATED_DIR, MODELS_DIR, ROOT, read_jsonl
from distill.curation.format_for_sft import has_rationale

ZOO = ROOT / "distill/student/model_zoo.yaml"


class KDTrainer(Trainer):
    def __init__(self, *a, teacher=None, alpha=0.5, temp=2.0, **kw):
        super().__init__(*a, **kw)
        self.teacher = teacher.to(self.args.device).eval()
        self.alpha, self.temp = alpha, temp

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
        with torch.no_grad():
            t_logits = self.teacher(input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"]).logits
        s_logits = out.logits
        v = min(s_logits.size(-1), t_logits.size(-1))                    # vocab sizes can differ slightly
        s, t = s_logits[:, :-1, :v], t_logits[:, :-1, :v]
        mask = (labels[:, 1:] != -100).float()
        T = self.temp
        kl = F.kl_div(F.log_softmax(s / T, -1), F.softmax(t / T, -1), reduction="none").sum(-1) * T * T
        kl = (kl * mask).sum() / mask.sum().clamp(min=1)
        loss = self.alpha * kl + (1 - self.alpha) * out.loss
        return (loss, out) if return_outputs else loss


def _encode(ex, tok, max_len=512):
    p = tok.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
    c = ex["completion"][0]["content"] + tok.eos_token
    p_ids = tok(p, add_special_tokens=False).input_ids
    c_ids = tok(c, add_special_tokens=False).input_ids
    return {"input_ids": (p_ids + c_ids)[:max_len],
            "attention_mask": [1] * len((p_ids + c_ids)[:max_len]),
            "labels": ([-100] * len(p_ids) + c_ids)[:max_len]}


def train(student: str, teacher_id: str, tag: str, n_train: int | None = None,
          epochs: int = 3, lr: float = 1e-5, bs: int = 4, accum: int = 4) -> None:
    base_id = yaml.safe_load(open(ZOO))[student]["hf_id"]
    tok = AutoTokenizer.from_pretrained(base_id)
    rows = read_jsonl(CURATED_DIR / (f"train_{n_train}.jsonl" if n_train else "train.jsonl"))
    vals = read_jsonl(CURATED_DIR / "val.jsonl")
    enc = lambda rs: Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rs]) \
        .map(lambda ex: _encode(ex, tok), remove_columns=["prompt", "completion"])
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
    teacher = AutoModelForCausalLM.from_pretrained(teacher_id, torch_dtype=torch.bfloat16)
    out = MODELS_DIR / tag
    args = TrainingArguments(output_dir=str(out / "checkpoints"), num_train_epochs=epochs,
                             per_device_train_batch_size=bs, gradient_accumulation_steps=accum,
                             learning_rate=lr, bf16=True, logging_steps=10, save_strategy="no",
                             report_to="none", remove_unused_columns=False)
    trainer = KDTrainer(model=model, args=args, train_dataset=enc(rows), eval_dataset=enc(vals),
                        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
                        teacher=teacher)
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    trainer.save_model(str(out)); tok.save_pretrained(str(out))
    (out / "meta.json").write_text(json.dumps({"kind": "kd", "student": student, "base_id": base_id,
                                               "method": "full", "n_train": len(rows), "train_seconds": secs,
                                               "rationale": has_rationale(rows)}))