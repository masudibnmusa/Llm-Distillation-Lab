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

# Current Claude models reject `temperature`, and thinking is on by default.
# Thinking tokens count toward max_tokens, so keep effort low for this simple task
# and leave enough room for the thinking plus the short JSON answer.
DEFAULT_EFFORT = "low"


class TeacherClient:
    def __init__(self, model: str | None = None, effort: str = DEFAULT_EFFORT):
        self.model = model or TEACHER_MODEL
        self.effort = effort
        self.client = anthropic.Anthropic()

    def _key(self, messages, system, max_tokens, sample_idx) -> str:
        blob = json.dumps([self.model, self.effort, messages, system, max_tokens, sample_idx], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete(self, messages: list[dict], system: str = "", max_tokens: int = 1024,
                 sample_idx: int = 0, use_cache: bool = True, tag: str = "label") -> tuple[str, float]:
        """Returns (text, latency_seconds). Cache hits cost nothing and are not ledgered.
        `sample_idx` gives each self-consistency sample its own cache entry; the model's
        default sampling supplies the variation between samples."""
        path = CACHE_DIR / f"{self._key(messages, system, max_tokens, sample_idx)}.json"
        if use_cache and path.exists():
            d = json.loads(path.read_text())
            return d["text"], d["latency"]

        kwargs = dict(model=self.model, max_tokens=max_tokens, messages=messages,
                      output_config={"effort": self.effort})
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
        # Only cache complete answers. A truncated (max_tokens) or refused reply would
        # otherwise be stuck in the cache and never retried.
        if use_cache and resp.stop_reason == "end_turn":
            path.write_text(json.dumps({"text": text, "latency": latency}))
        elif resp.stop_reason != "end_turn":
            print(f"warning: teacher stop_reason={resp.stop_reason} (tag={tag}); not cached")
        return text, latency
