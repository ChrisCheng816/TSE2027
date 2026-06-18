from __future__ import annotations


BIGCODEBENCH_FAMILY_DATASETS = frozenset(
    {
        "bigcodebench-v0.1.4",
        "bigcodebench-hard-v0.1.4",
    }
)


def is_bigcodebench_family_dataset(dataset_name: str) -> bool:
    return dataset_name in BIGCODEBENCH_FAMILY_DATASETS
