from distill.data.deduplicator import near_dup_mask
from distill.data.splitter import make_split


def test_near_dup_detected():
    mask = near_dup_mask(["i lost my credit card yesterday!", "how do i reset my password"],
                         ["I lost my credit card yesterday"], thr=0.8)
    assert mask.tolist() == [True, False]


def test_no_overlap_between_gold_and_pool():
    texts = [f"unique ticket number {i} about topic {chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(60)]
    inputs = [{"id": str(i), "text": t} for i, t in enumerate(texts)]
    # near-dup of input 11: same words, different case/punctuation, so a different id
    gold = inputs[:10] + [{"id": "x", "text": texts[11].upper() + "!!"}]
    pool, split = make_split(inputs, gold)
    gold_ids = {g["id"] for g in gold}
    assert not gold_ids & {x["id"] for x in pool}
    assert not set(split["train"]) & set(split["val"])
    assert "11" not in {x["id"] for x in pool}   # near-duplicate of gold was removed