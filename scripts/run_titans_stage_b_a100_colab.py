"""Capture a complete, shareable Stage B A100 pilot bundle from Google Colab.

Run this script with the editable environment created by the companion Colab
notebook.  It creates one immutable timestamped directory in Google Drive,
captures command logs and NVIDIA provenance, runs the strict pilot, then
re-verifies the bundle before writing a concise handoff summary.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Sequence


class CaptureCommandError(RuntimeError):
    """A failed child process whose complete output is saved in Drive."""

    def __init__(self, command: Sequence[str], log_path: Path, returncode: int):
        super().__init__(
            f"command failed with exit code {returncode}: {' '.join(command)}; see {log_path}"
        )
        self.command = tuple(command)
        self.log_path = log_path
        self.returncode = returncode
        self.output_directory: Path | None = None


def _tail(path: Path, *, lines: int = 120) -> str:
    """Return a bounded tail so Colab displays the actionable part of a log."""

    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _run(command: Sequence[str], log_path: Path, *, cwd: Path) -> None:
    """Run one command, preserving its complete output in the Drive folder."""

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        output = completed.stdout
        returncode = completed.returncode
    except OSError as error:
        output = f"could not start {' '.join(command)}: {error}\n"
        returncode = 127
    log_path.write_text(output, encoding="utf-8")
    if returncode:
        raise CaptureCommandError(command, log_path, returncode)


def _output_path(drive_root: Path, run_id: str | None) -> Path:
    chosen_id = run_id or datetime.now(timezone.utc).strftime("a100-%Y%m%dT%H%M%SZ")
    output = drive_root / chosen_id
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing evidence folder: {output}; choose --run-id"
        )
    return output


def _git_value(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def run_capture(
    *,
    repository_root: Path,
    drive_root: Path,
    warmup_runs: int,
    repetitions: int,
    run_id: str | None,
    run_tests: bool,
) -> Path:
    """Run preflight, evidence capture, and verification into a new Drive folder."""

    if repetitions < 3:
        raise ValueError("the strict Stage B A100 pilot requires at least 3 repetitions")
    python = Path(sys.executable)
    output = _output_path(drive_root, run_id)
    logs = output / "logs"
    evidence = output / "a100"
    logs.mkdir(parents=True)

    metadata = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repository_root.resolve()),
        "git_commit": _git_value(repository_root, "rev-parse", "HEAD"),
        "git_branch": _git_value(repository_root, "branch", "--show-current"),
        "python": str(python),
        "python_version": platform.python_version(),
        "warmup_runs": warmup_runs,
        "repetitions": repetitions,
        "evidence_directory": str(evidence),
    }
    (output / "capture_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        _run(["nvidia-smi"], logs / "nvidia-smi.txt", cwd=repository_root)
        if run_tests:
            _run(
                [
                    str(python),
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_titans_paper_mac_stage_b_a100_pilot.py",
                    "tests/test_titans_paper_mac_stage_b_training_step.py",
                ],
                logs / "contract_tests.txt",
                cwd=repository_root,
            )
        _run(
            [
                str(python),
                "-m",
                "seqtrainer.torch.titans_paper_mac_stage_b.a100_pilot",
                "--preflight-only",
            ],
            logs / "a100_preflight.txt",
            cwd=repository_root,
        )
        _run(
            [
                str(python),
                "-m",
                "seqtrainer.torch.titans_paper_mac_stage_b.a100_pilot",
                "--output-dir",
                str(evidence),
                "--warmup-runs",
                str(warmup_runs),
                "--repetitions",
                str(repetitions),
            ],
            logs / "a100_capture.txt",
            cwd=repository_root,
        )
        _run(
            [
                str(python),
                "-m",
                "seqtrainer.torch.titans_paper_mac_stage_b.a100_pilot",
                "--output-dir",
                str(evidence),
                "--verify-only",
            ],
            logs / "a100_verify.txt",
            cwd=repository_root,
        )
    except CaptureCommandError as error:
        error.output_directory = output
        (output / "FAILED.txt").write_text(
            f"Stage B A100 capture failed.\n\n{error}\n\n"
            f"Failed log: {error.log_path}\n"
            "Open the log above or rerun the final notebook cell to print its tail.\n",
            encoding="utf-8",
        )
        raise
    (output / "README.txt").write_text(
        "This directory is the complete Stage B named-A100 handoff. Share the "
        "directory containing a100/ and logs/; do not share only the manifest.\n"
        f"Evidence: {evidence}\n"
        f"Commit: {metadata['git_commit']}\n",
        encoding="utf-8",
    )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive/SeqTrainerA100"),
        help="Drive parent directory; a unique run folder is created inside it.",
    )
    parser.add_argument("--run-id", help="Optional unique subdirectory name.")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--skip-contract-tests", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = run_capture(
            repository_root=args.repository_root,
            drive_root=args.drive_root,
            warmup_runs=args.warmup_runs,
            repetitions=args.repetitions,
            run_id=args.run_id,
            run_tests=not args.skip_contract_tests,
        )
    except CaptureCommandError as error:
        print(f"Stage B A100 capture failed: {error}")
        assert error.output_directory is not None
        print(f"Drive debug folder: {error.output_directory}")
        print(f"\n--- tail: {error.log_path.name} ---")
        print(_tail(error.log_path))
        return 1
    except (FileExistsError, RuntimeError, ValueError) as error:
        print(f"Stage B A100 capture failed: {error}")
        return 1
    print(f"Stage B A100 capture verified: {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
