"""Merge LoRA into the base model, then optionally quantize."""
from __future__ import annotations
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from distill.config import MODELS_DIR


def merge_lora(tag: str) -> str:
    d = MODELS_DIR / tag
    meta = json.loads((d / "meta.json").read_text())
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(meta["base_id"], torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, str(d)).merge_and_unload()
    out = MODELS_DIR / f"{tag}-merged"
    merged.save_pretrained(out)
    AutoTokenizer.from_pretrained(meta["base_id"]).save_pretrained(out)
    return str(out)


def awq_quantize(model_dir: str, out_dir: str | None = None) -> str:
    """Needs `pip install autoawq`. For GGUF (Ollama), use llama.cpp's convert_hf_to_gguf.py on the merged dir."""
    from awq import AutoAWQForCausalLM
    out_dir = out_dir or model_dir + "-awq"
    tok = AutoTokenizer.from_pretrained(model_dir)
    m = AutoAWQForCausalLM.from_pretrained(model_dir)
    m.quantize(tok, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"})
    m.save_quantized(out_dir); tok.save_pretrained(out_dir)
    return out_dir


if __name__ == "__main__":
    import sys
    print(merge_lora(sys.argv[1]))