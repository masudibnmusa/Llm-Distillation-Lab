"""Local HF model wrapper + the prompted (no training) small-model baseline + the trained StudentSystem."""
from __future__ import annotations
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from distill.config import MODELS_DIR, SMALL_BASELINE_MODEL
from distill.task.prompt_templates import full_messages, student_messages
from distill.task.schema import Prediction, to_prediction


class LocalLM:
    def __init__(self, model_id: str, adapter: str | None = None, max_new_tokens: int = 32):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.eval()
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def generate(self, convs: list[list[dict]], batch_size: int = 16) -> list[tuple[str, float, float]]:
        """Returns (text, sequence_probability, seconds_per_item). Sequence prob is used as confidence."""
        results = []
        for i in range(0, len(convs), batch_size):
            chunk = convs[i:i + batch_size]
            prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in chunk]
            enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            t0 = time.perf_counter()
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      return_dict_in_generate=True, output_scores=True,
                                      pad_token_id=self.tok.pad_token_id)
            dt = (time.perf_counter() - t0) / len(chunk)
            gen = out.sequences[:, enc["input_ids"].shape[1]:]
            lp = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            for j in range(len(chunk)):
                keep = gen[j] != self.tok.pad_token_id
                text = self.tok.decode(gen[j][keep], skip_special_tokens=True)
                results.append((text, float(torch.exp(lp[j][keep].sum())), dt))
        return results


class PromptedSmall:
    """Same base model as the student, few-shot prompt, NO training. The baseline to beat."""
    def __init__(self, model_id: str | None = None):
        self.name = "prompted_small"
        self.lm = LocalLM(model_id or SMALL_BASELINE_MODEL)

    def predict(self, texts: list[str]) -> list[Prediction]:
        outs = self.lm.generate([full_messages(t) for t in texts])
        return [to_prediction(raw, conf, dt) for raw, conf, dt in outs]


class StudentSystem:
    def __init__(self, tag: str):
        d = MODELS_DIR / tag
        meta = json.loads((d / "meta.json").read_text())
        self.name = tag
        max_new = 160 if meta.get("rationale") else 32
        if meta.get("method") == "lora":
            self.lm = LocalLM(meta["base_id"], adapter=str(d), max_new_tokens=max_new)
        else:
            self.lm = LocalLM(str(d), max_new_tokens=max_new)

    def predict(self, texts: list[str]) -> list[Prediction]:
        outs = self.lm.generate([student_messages(t) for t in texts])
        return [to_prediction(raw, conf, dt) for raw, conf, dt in outs]