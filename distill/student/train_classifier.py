"""Optional: encoder classifier (e.g. DistilBERT). Often 10-100x cheaper than an LLM student."""
from __future__ import annotations
import json
import time
import numpy as np
import torch
import yaml
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)
from distill.config import CURATED_DIR, LABELS, MODELS_DIR, ROOT, read_jsonl
from distill.task.schema import Prediction

ZOO = ROOT / "distill/student/model_zoo.yaml"


def train(student: str = "distilbert", tag: str = "distilbert", n_train: int | None = None,
          epochs: int = 4, lr: float = 5e-5, bs: int = 32) -> None:
    base_id = yaml.safe_load(open(ZOO))[student]["hf_id"]
    l2i = {l: i for i, l in enumerate(LABELS)}
    tok = AutoTokenizer.from_pretrained(base_id)
    tr = read_jsonl(CURATED_DIR / (f"train_{n_train}.jsonl" if n_train else "train.jsonl"))
    va = read_jsonl(CURATED_DIR / "val.jsonl")

    def enc(rows):
        ds = Dataset.from_dict({"text": [r["text"] for r in rows], "label": [l2i[r["label"]] for r in rows]})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_id, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=l2i)
    out = MODELS_DIR / tag
    args = TrainingArguments(output_dir=str(out / "checkpoints"), num_train_epochs=epochs,
                             per_device_train_batch_size=bs, learning_rate=lr, eval_strategy="epoch",
                             save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
                             metric_for_best_model="accuracy", report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=enc(tr), eval_dataset=enc(va),
                      processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=lambda p: {"accuracy": float((p.predictions.argmax(-1) == p.label_ids).mean())})
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    trainer.save_model(str(out)); tok.save_pretrained(str(out))
    (out / "meta.json").write_text(json.dumps({"kind": "classifier", "student": student, "base_id": base_id,
                                               "n_train": len(tr), "train_seconds": secs}))


class EncoderSystem:
    def __init__(self, tag: str):
        d = MODELS_DIR / tag
        self.name = tag
        self.tok = AutoTokenizer.from_pretrained(d)
        self.model = AutoModelForSequenceClassification.from_pretrained(d).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, texts: list[str], bs: int = 64) -> list[Prediction]:
        import time
        preds = []
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            enc = self.tok(chunk, truncation=True, max_length=128, padding=True, return_tensors="pt").to(self.device)
            t0 = time.perf_counter()
            probs = torch.softmax(self.model(**enc).logits, -1).cpu().numpy()
            dt = (time.perf_counter() - t0) / len(chunk)
            for p in probs:
                k = int(np.argmax(p))
                preds.append(Prediction(LABELS[k], float(p[k]), True, dt, LABELS[k]))
        return preds