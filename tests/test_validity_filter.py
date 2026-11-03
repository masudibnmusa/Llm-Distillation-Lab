from distill.config import LABELS
from distill.curation.validity_filter import filter_valid
from distill.task.schema import parse_output


def test_filter_drops_missing_labels():
    rows = [{"id": "a", "label": LABELS[0]}, {"id": "b", "label": None}]
    assert [r["id"] for r in filter_valid(rows)] == ["a"]


def test_parse_output():
    assert parse_output('{"category": "%s"}' % LABELS[0]).category == LABELS[0]
    assert parse_output('```json\n{"category": "%s"}\n```' % LABELS[0]).category == LABELS[0]
    assert parse_output('{"category": "not_a_label"}') is None
    assert parse_output("hello") is None