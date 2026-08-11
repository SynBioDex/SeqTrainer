"""Download only the five E. coli checkpoints from the official iPro-MP ZIP."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from remotezip import RemoteZip


ARCHIVE_URL = "https://zenodo.org/api/records/15180139/files/model.zip/content"
ARCHIVE_MD5 = "00f29062715132f16977881969e25c43"
EXPECTED_BYTES = 358_961_146
SPECIES_ID = 10


def download_ecoli_weights(output_dir: str | Path) -> dict[str, object]:
    """Range-download species 10 checkpoints instead of the full 38 GB ZIP."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    members = [f"07-final/{SPECIES_ID}_fold_{fold}.pth" for fold in range(1, 6)]
    written: list[str] = []

    with RemoteZip(ARCHIVE_URL) as archive:
        available = {info.filename: info for info in archive.infolist()}
        missing = [member for member in members if member not in available]
        if missing:
            raise FileNotFoundError(f"Official archive is missing expected members: {missing}")

        for member in members:
            target = destination / Path(member).name
            expected_size = available[member].file_size
            if expected_size != EXPECTED_BYTES:
                raise RuntimeError(
                    f"Unexpected size for {member}: {expected_size}; expected {EXPECTED_BYTES}. "
                    "The Zenodo archive may have changed."
                )
            if target.is_file() and target.stat().st_size == expected_size:
                written.append(str(target))
                continue
            partial = target.with_suffix(target.suffix + ".part")
            with archive.open(member) as source, partial.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
            if partial.stat().st_size != expected_size:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"Incomplete download for {member}")
            partial.replace(target)
            written.append(str(target))

    metadata = {
        "zenodo_record": 15180139,
        "archive_url": ARCHIVE_URL,
        "archive_md5": ARCHIVE_MD5,
        "full_archive_bytes": 38_321_156_841,
        "species_id": SPECIES_ID,
        "species_name": "Escherichia coli str K-12 substr. MG1655",
        "checkpoint_bytes_each": EXPECTED_BYTES,
        "checkpoint_paths": written,
    }
    (destination / "download_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(download_ecoli_weights(args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
