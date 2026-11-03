"""Anthropic API wrapper with disk cache, retries, and cost tracking."""
from __future__ import annotations
import hashlib
import json
import time
import anthropic
from distill.config import LABELS_DIR, TEACHER_MODEL
from distill.teacher import cost_ledger

CACHE_DIR = LABELS_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class TeacherClient:
    def __init__(self, model: str | None = None):
        self.model = model or TEACHER_MODEL
        self.client = anthropic.Anthropic()

    def _key(self, messages, system, temperature, max_tokens, sample_idx) -> str:
        blob = json.dumps([self.model, messages, system, temperature, max_tokens, sample_idx], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete(self, messages: list[dict], system: str = "", temperature: float = 0.0,
                 max_tokens: int = 200, sample_idx: int = 0, use_cache: bool = True,
                 tag: str = "label") -> tuple[str, float]:
        """Returns (text, latency_seconds). Cache hits cost nothing and are not ledgered."""
        path = CACHE_DIR / f"{self._key(messages, system, temperature, max_tokens, sample_idx)}.json"
        if use_cache and path.exists():
            d = json.loads(path.read_text())
            return d["text"], d["latency"]

        kwargs = dict(model=self.model, max_tokens=max_tokens, temperature=temperature, messages=messages)
        if system:
            kwargs["system"] = system
        for attempt in range(6):
            try:
                t0 = time.perf_counter()
                resp = self.client.messages.create(**kwargs)
                latency = time.perf_counter() - t0
                break
            except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError):
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError("teacher call failed after retries")

        text = "".join(b.text for b in resp.content if b.type == "text")
        cost_ledger.record(self.model, resp.usage.input_tokens, resp.usage.output_tokens, tag)
        if use_cache:
            path.write_text(json.dumps({"text": text, "latency": latency}))
        return text, latency