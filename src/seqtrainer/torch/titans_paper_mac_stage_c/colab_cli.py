"""Drive-backed, failure-visible command wrapper for Stage C Colab handoffs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
from typing import Sequence

import torch


def _git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _environment(repo: Path | None) -> dict[str, object]:
    packages = {}
    for name in ("seqtrainer", "numpy", "pandas", "torch", "transformers", "tokenizers", "vtx", "pyarrow"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    payload: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable",
        "packages": packages,
    }
    if repo is not None:
        payload.update(
            {
                "git_commit": _git_value(repo, "rev-parse", "HEAD"),
                "git_ref": _git_value(repo, "describe", "--always", "--dirty"),
                "git_status": _git_value(repo, "status", "--short"),
            }
        )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one Colab step with continuously persisted logs and a visible failure marker"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a command is required after --")
    if "/" in args.label or "\\" in args.label or args.label in {".", ".."}:
        parser.error("--label must be a single safe path component")

    run_dir = args.run_dir.resolve()
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.label}.log"
    failure_path = run_dir / "FAILED.txt"
    manifest_path = run_dir / "colab_run_manifest.json"
    started = datetime.now(timezone.utc).isoformat()
    environment = _environment(args.repo.resolve() if args.repo else None)
    prior: dict[str, object] = {"format_version": 1, "steps": []}
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("steps"), list):
            prior = loaded
    step: dict[str, object] = {
        "label": args.label,
        "command": command,
        "started_at": started,
        "status": "running",
        "log": str(log_path.relative_to(run_dir)),
    }
    steps = list(prior["steps"])
    steps.append(step)
    manifest = {"format_version": 1, "environment": environment, "steps": steps}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"started_at={started}\n")
        log.write(f"command={json.dumps(command)}\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return_code = process.wait()

    finished = datetime.now(timezone.utc).isoformat()
    step.update({"finished_at": finished, "return_code": return_code})
    if return_code == 0:
        step["status"] = "passed"
        if failure_path.exists():
            failure_path.unlink()
    else:
        step["status"] = "failed"
        failure_path.write_text(
            f"Stage C Colab step failed: {args.label}\n"
            f"return_code={return_code}\n"
            f"log={log_path}\n",
            encoding="utf-8",
        )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if return_code:
        raise SystemExit(return_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
