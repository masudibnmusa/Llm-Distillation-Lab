"""API-backed systems: the teacher itself, or a cheaper tier of the same family."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from distill.task.prompt_templates import SYSTEM, teacher_messages
from distill.task.schema import Prediction, to_prediction
from distill.teacher.teacher_client import TeacherClient


class APISystem:
    def __init__(self, model: str, fewshot: bool = True, workers: int = 8, use_cache: bool = True):
        self.name, self.fewshot, self.workers, self.use_cache = model, fewshot, workers, use_cache
        self.client = TeacherClient(model)

    def predict_one(self, text: str) -> Prediction:
        raw, lat = self.client.complete(teacher_messages(text, self.fewshot), system=SYSTEM,
                                        use_cache=self.use_cache, tag="eval")
        return to_prediction(raw, None, lat)

    def predict(self, texts: list[str]) -> list[Prediction]:
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self.predict_one, texts))