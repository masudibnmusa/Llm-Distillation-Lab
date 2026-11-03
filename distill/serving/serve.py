"""Serve a merged student with vLLM (OpenAI-compatible) or Ollama."""
from __future__ import annotations
import subprocess
import sys


def serve_vllm(model_dir: str, name: str = "student", port: int = 8000, max_len: int = 1024,
               quantization: str | None = None) -> None:
    cmd = ["vllm", "serve", model_dir, "--served-model-name", name, "--port", str(port),
           "--max-model-len", str(max_len)]
    if quantization:
        cmd += ["--quantization", quantization]
    subprocess.run(cmd, check=True)


def serve_ollama(gguf_path: str, name: str = "student") -> None:
    with open("Modelfile", "w") as f:
        f.write(f"FROM {gguf_path}\nPARAMETER temperature 0\n")
    subprocess.run(["ollama", "create", name, "-f", "Modelfile"], check=True)
    subprocess.run(["ollama", "run", name])


if __name__ == "__main__":
    serve_vllm(sys.argv[1], *(sys.argv[2:3] or ["student"]))