from distill.task.schema import Prediction
from distill.task.scoring import accuracy, exact_match, macro_f1, schema_validity


def test_exact_match():
    assert exact_match("a", "a") == 1 and exact_match("a", "b") == 0 and exact_match(None, "a") == 0


def test_accuracy_and_f1():
    assert accuracy(["a", "b", None], ["a", "b", "a"]) == 2 / 3
    assert macro_f1(["a", "b"], ["a", "b"]) == 1.0


def test_schema_validity():
    ps = [Prediction("a", valid=True), Prediction(None, valid=False)]
    assert schema_validity(ps) == 0.5