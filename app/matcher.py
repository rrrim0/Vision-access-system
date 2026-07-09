from __future__ import annotations

from typing import Any

import numpy as np

from app.db import get_templates_database


def to_numpy_embedding(embedding: Any) -> np.ndarray:
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr


def cosine_similarity(a: Any, b: Any) -> float:
    va = to_numpy_embedding(a)
    vb = to_numpy_embedding(b)

    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0.0:
        return 0.0

    return float(np.dot(va, vb) / denom)


def match_face(query_embedding: Any, threshold: float = 0.75) -> dict[str, Any] | None:
    templates = get_templates_database()

    best_match: dict[str, Any] | None = None
    best_score = -1.0

    query = to_numpy_embedding(query_embedding)

    for item in templates:
        score = cosine_similarity(query, item["embedding"])

        if score > best_score:
            best_score = score
            best_match = {
                "user_id": item["user_id"],
                "username": item["username"],
                "full_name": item["full_name"],
                "template_id": item["template_id"],
                "score": score,
            }

    if best_match is None:
        return None

    if best_match["score"] < threshold:
        return None

    return best_match