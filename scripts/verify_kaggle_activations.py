#!/usr/bin/env python
"""Verify downloaded activation blobs against the published Kaggle manifest.

``kaggle_activations._validate_blob`` checks model_name / layer / row count, which is
enough for high-stakes and hu_ha but NOT for the instructions concept: the manifest
says so in as many words — eval row counts are 200 four times and 194 twice, dev 68
four times and 66 twice — so a rows-only check cannot tell a swapped blob from the
right one. ``seq_len`` separates all seven, and the manifest records it for exactly
that reason. This pass checks both, for every concept.

It also re-checks the eval/dev split-stem collision the manifest warns about
(high-stakes and instructions use the same stems at different row counts), which the
``activations/{eval,dev}/<concept>/`` layout keeps apart by construction.

    .venv_claude/bin/python scripts/verify_kaggle_activations.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_redteam.kaggle_activations import _blob_header, _jsonl_rows  # noqa: E402

MANIFEST = REPO_ROOT / "activations" / "eval_activations_manifest.json"
OUT_ROOT = REPO_ROOT / "activations"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    rows_ok = rows_bad = missing = 0
    problems: list[str] = []

    for key, concept in sorted(manifest["concepts"].items()):
        kind = concept["kind"]                       # eval | dev
        name = concept["concept"]                    # highstakes | hu_harm | instructions
        # The manifest's `concept` field is the published name; the local directory is
        # the tail of eval_dataset_dir, which is what this repo's tree actually uses
        # (hu_harm -> hu_ha).
        local_dir_name = Path(concept["eval_dataset_dir"]).name
        stem = Path(concept["cache_stem"]).stem
        suffix = Path(concept["cache_stem"]).suffix or ".pt"
        blob_dir = OUT_ROOT / kind / local_dir_name
        split_dir = REPO_ROOT / concept["eval_dataset_dir"]

        print(f"\n=== {key}  ({name}/{kind})  {blob_dir.relative_to(REPO_ROOT)}")
        for split, spec in sorted(concept["splits"].items()):
            blob = blob_dir / f"{split}-{stem}{suffix}"
            want_rows, want_seq = int(spec["rows"]), int(spec["seq_len"])

            # The manifest's row count must agree with the split JSONL on disk, or the
            # blob is right about a split we do not have.
            jsonl = split_dir / f"{split}.jsonl"
            local_rows = _jsonl_rows(jsonl) if jsonl.is_file() else None
            if local_rows is not None and local_rows != want_rows:
                problems.append(
                    f"{key}/{split}: manifest says {want_rows} rows but "
                    f"{jsonl.relative_to(REPO_ROOT)} has {local_rows}"
                )

            if not blob.is_file():
                missing += 1
                print(f"  {split:<28} -- not downloaded")
                continue

            try:
                data = _blob_header(blob)
            except Exception as e:  # noqa: BLE001
                rows_bad += 1
                problems.append(f"{key}/{split}: unreadable blob {blob}: {e}")
                print(f"  {split:<28} UNREADABLE")
                continue

            shape = tuple(data["activations"].shape)
            got_rows, got_seq = int(shape[0]), int(shape[1])
            got_model, got_layer = data.get("model_name"), data.get("layer")
            bad = []
            if got_rows != want_rows:
                bad.append(f"rows {got_rows} != {want_rows}")
            if got_seq != want_seq:
                bad.append(f"seq_len {got_seq} != {want_seq}")
            if got_model is not None and got_model != concept["model_name"]:
                bad.append(f"model {got_model!r} != {concept['model_name']!r}")
            if got_layer is not None and int(got_layer) != int(concept["layer"]):
                bad.append(f"layer {got_layer} != {concept['layer']}")
            if bad:
                rows_bad += 1
                problems.append(f"{key}/{split}: " + "; ".join(bad))
                print(f"  {split:<28} MISMATCH  {'; '.join(bad)}")
            else:
                rows_ok += 1
                gb = blob.stat().st_size / 1e9
                print(f"  {split:<28} ok  {got_rows:>5} x {got_seq:<5} {gb:6.2f} GB")

    print(f"\n==== {rows_ok} verified, {rows_bad} mismatched, {missing} not downloaded ====")
    for p in problems:
        print(f"  PROBLEM {p}")
    return 1 if (rows_bad or problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
