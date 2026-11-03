"""Black-box distillation: SFT on (input, teacher output) with TRL. LoRA or full fine-tuning."""
from __future__ import annotations
import json
import time
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from distill.config import CURATED_DIR, MODELS_DIR, ROOT
from distill.curation.format_for_sft import has_rationale

ZOO = ROOT / "distill/student/model_zoo.yaml"


def train(student: str, config_path: str, tag: str, n_train: int | None = None) -> None:
    cfg = yaml.safe_load(open(ROOT / config_path if not str(config_path).startswith("/") else config_path))
    base_id = yaml.safe_load(open(ZOO))[student]["hf_id"]
    train_file = CURATED_DIR / (f"train_{n_train}.jsonl" if n_train else "train.jsonl")

    ds = load_dataset("json", data_files={"train": str(train_file),
                                          "validation": str(CURATED_DIR / "val.jsonl")})
    ds = ds.select_columns(["prompt", "completion"])   # loss is computed on the completion only

    tok = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
    peft_cfg = None
    if cfg["method"] == "lora":
        peft_cfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                              target_modules=cfg["target_modules"], task_type="CAUSAL_LM")

    out = MODELS_DIR / tag
    args = SFTConfig(
        output_dir=str(out / "checkpoints"), num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"], learning_rate=cfg["lr"],
        lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", bf16=True,
        max_length=cfg["max_length"], report_to="none", seed=cfg["seed"],
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"],
                         processing_class=tok, peft_config=peft_cfg)
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    (out / "meta.json").write_text(json.dumps({
        "kind": "sft", "student": student, "base_id": base_id, "method": cfg["method"],
        "n_train": len(ds["train"]), "train_seconds": secs, "rationale": has_rationale(ds["train"])}))
    print(f"saved to {out} ({secs / 60:.1f} min)")