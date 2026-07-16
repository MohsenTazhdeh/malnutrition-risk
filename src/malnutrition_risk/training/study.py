import hashlib
import json
from typing import Any, Mapping


def study_fingerprint(*, search_space: Mapping[str, Any], cv_folds: int, seed: int,
                      objective: str, dataset_digest: str) -> str:
    """Short hash of everything that defines trial comparability. Any change -> a NEW study."""
    payload = {"search_space": search_space, "cv_folds": cv_folds, "seed": seed,
               "objective": objective, "dataset_digest": dataset_digest}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]

def build_study_name(experiment_name: str, study_fingerprint: str, version: str):
    return '__'.join([experiment_name, study_fingerprint, f"v{version}"])


