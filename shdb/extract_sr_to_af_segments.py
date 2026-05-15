import argparse
import csv
from pathlib import Path
from typing import Iterable

import wfdb


def normalize_label(aux_note: str) -> str:
    """
    Normalize rhythm labels from WFDB aux_note fields.
    """
    if aux_note is None:
        return ""
    label = aux_note.strip().split()[0] if aux_note.strip() else ""
    return label


def find_record_ids(dataset_dir: Path) -> list[str]:
    """
    Get sorted record IDs that have both .hea and .dat.
    """
    record_ids: set[str] = set()
    for hea_path in dataset_dir.glob("*.hea"):
        rid = hea_path.stem
        if (dataset_dir / f"{rid}.dat").exists():
            record_ids.add(rid)
    return sorted(record_ids)


def iter_sr_to_af_transitions(dataset_dir: Path, record_ids: Iterable[str]):
    """
    Yield dictionaries describing transitions from (N to (AFIB.
    """
    for rid in record_ids:
        atr_path = dataset_dir / f"{rid}.atr"
        if not atr_path.exists():
            continue

        rec = wfdb.rdrecord(str(dataset_dir / rid))
        ann = wfdb.rdann(str(dataset_dir / rid), "atr")

        fs = float(rec.fs)
        total_samples = int(rec.sig_len)

        rhythm_points: list[tuple[int, str]] = []
        for sample, aux in zip(ann.sample, ann.aux_note):
            label = normalize_label(aux)
            if label.startswith("("):
                rhythm_points.append((int(sample), label))

        if len(rhythm_points) < 2:
            continue

        for idx in range(len(rhythm_points) - 1):
            curr_start, curr_label = rhythm_points[idx]
            next_start, next_label = rhythm_points[idx + 1]

            if curr_label == "(N" and next_label == "(AFIB":
                sr_start = curr_start
                sr_end = next_start
                af_start = next_start
                af_end = total_samples
                if idx + 2 < len(rhythm_points):
                    af_end = rhythm_points[idx + 2][0]

                yield {
                    "data_id": rid,
                    "fs_hz": fs,
                    "sr_label": curr_label,
                    "af_label": next_label,
                    "sr_start_sample": sr_start,
                    "sr_end_sample": sr_end,
                    "af_start_sample": af_start,
                    "af_end_sample": af_end,
                    "sr_start_sec": sr_start / fs,
                    "sr_end_sec": sr_end / fs,
                    "af_start_sec": af_start / fs,
                    "af_end_sec": af_end / fs,
                    "sr_duration_sec": (sr_end - sr_start) / fs,
                    "af_duration_sec": (af_end - af_start) / fs,
                    "record_path": str(dataset_dir / rid),
                    "atr_file": str(dataset_dir / f"{rid}.atr"),
                    "dat_file": str(dataset_dir / f"{rid}.dat"),
                    "hea_file": str(dataset_dir / f"{rid}.hea"),
                }


def main():
    parser = argparse.ArgumentParser(
        description="Extract SHDB SR->AF transitions to CSV (no duration filtering)."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/home/intellisense01/EML-Labs/datasets/shdb"),
        help="Path to SHDB dataset directory containing WFDB files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("shdb/sr_to_af_segments.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    dataset_dir: Path = args.dataset_dir
    output_csv: Path = args.output_csv

    record_ids = find_record_ids(dataset_dir)
    rows = list(iter_sr_to_af_transitions(dataset_dir, record_ids))

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "data_id",
        "fs_hz",
        "sr_label",
        "af_label",
        "sr_start_sample",
        "sr_end_sample",
        "af_start_sample",
        "af_end_sample",
        "sr_start_sec",
        "sr_end_sec",
        "af_start_sec",
        "af_end_sec",
        "sr_duration_sec",
        "af_duration_sec",
        "record_path",
        "atr_file",
        "dat_file",
        "hea_file",
    ]

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Records scanned: {len(record_ids)}")
    print(f"SR->AF transitions found: {len(rows)}")
    print(f"CSV written to: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
