import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def image_stem(name):
    text = Path(str(name)).name
    while re.search(r"\.(jpe?g|png|bmp|tiff?)$", text, flags=re.I):
        text = re.sub(r"\.(jpe?g|png|bmp|tiff?)$", "", text, flags=re.I)
    return re.sub(r"^_\d+_", "", text)


def normalized_stem(name):
    text = unicodedata.normalize("NFKC", image_stem(name)).lower()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)


def patient_id_from_name(name):
    text = normalized_stem(name)
    return re.sub(r"(?:h[1-4]|[mf][1-4]|[1-4])$", "", text)


def _strata(values):
    values = pd.Series(values).reset_index(drop=True)
    for bins in range(min(4, len(values)), 1, -1):
        labels = pd.qcut(values.rank(method="first"), q=bins, labels=False, duplicates="drop")
        counts = labels.value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            return labels
    return None


def _split_ids(ids, values, test_size, seed):
    strata = _strata(values)
    try:
        left, right = train_test_split(
            ids,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=strata,
        )
    except ValueError:
        left, right = train_test_split(
            ids,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
    return list(left), list(right)


def assign_patient_split(frame, seed=42, test_ratio=0.2, validation_ratio=0.08):
    patient_values = frame.groupby("patient_id", sort=True)["emean"].mean()
    patient_ids = patient_values.index.to_list()
    development_ids, test_ids = _split_ids(
        patient_ids,
        patient_values.to_numpy(),
        test_ratio,
        seed,
    )
    validation_fraction = validation_ratio / (1.0 - test_ratio)
    development_values = patient_values.loc[development_ids]
    train_ids, validation_ids = _split_ids(
        development_ids,
        development_values.to_numpy(),
        validation_fraction,
        seed + 1000,
    )
    result = frame.copy()
    result["split"] = "train"
    result.loc[result["patient_id"].isin(validation_ids), "split"] = "val"
    result.loc[result["patient_id"].isin(test_ids), "split"] = "test"
    assert_patient_exclusivity(result)
    return result


def assert_patient_exclusivity(frame):
    groups = {
        split: set(frame.loc[frame["split"] == split, "patient_id"])
        for split in ("train", "val", "test")
    }
    if groups["train"] & groups["val"]:
        raise ValueError("Patient overlap between train and val")
    if groups["train"] & groups["test"]:
        raise ValueError("Patient overlap between train and test")
    if groups["val"] & groups["test"]:
        raise ValueError("Patient overlap between val and test")
