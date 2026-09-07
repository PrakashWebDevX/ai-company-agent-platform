"""Reciprocal Rank Fusion for combining dense (vector) and sparse (BM25) hits."""


def reciprocal_rank_fusion(
    dense_hits: list[dict],
    sparse_hits: list[dict],
    limit: int = 5,
    k: int = 60,
) -> list[dict]:
    """Merge two ranked hit lists into one, ranked by Reciprocal Rank Fusion.

    RRF scores each item by 1 / (k + rank) in each list it appears in and
    sums across lists, so an item ranked highly by *both* retrievers rises
    to the top without needing the two retrievers' raw scores (cosine
    similarity vs. BM25) to be on comparable scales -- which they aren't.
    `k` is RRF's standard smoothing constant (60 is the commonly used
    default from the original TREC paper) that keeps low ranks from being
    over-penalized.
    """
    fused_scores: dict[tuple[str | None, str | None], float] = {}
    payloads: dict[tuple[str | None, str | None], dict] = {}

    for rank, hit in enumerate(dense_hits):
        key = (hit.get("filename"), hit.get("text"))
        fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        payloads[key] = hit

    for rank, hit in enumerate(sparse_hits):
        key = (hit.get("filename"), hit.get("text"))
        fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        payloads.setdefault(key, hit)

    ranked_keys = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    results = []
    for key, score in ranked_keys:
        item = dict(payloads[key])
        item["score"] = float(score)
        results.append(item)
    return results