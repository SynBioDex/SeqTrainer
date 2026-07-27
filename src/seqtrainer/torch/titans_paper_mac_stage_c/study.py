"""Frozen Stage C study protocol, provenance ledger, and report compiler.

The ledger is deliberately plain JSON Lines.  A Drive-mounted directory is a
perfectly adequate append-only store, while canonical serialization and a hash
chain make accidental edits detectable without an additional dependency.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


STUDY_ID = "stage_c_ecoli_escherichia_medium_25m_v1"
SUPPORTED_STUDY_IDS = frozenset(
    {
        STUDY_ID,
        "stage_c_ecoli_escherichia_paper_deep_memory_v2",
    }
)
PROTOCOL_FILENAME = "protocol.json"
LEDGER_FILENAME = "ledger.jsonl"
GENESIS_HASH = "0" * 64


def canonical_json(value: Any) -> str:
    """Serialize JSON in the stable form used for every study hash."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_hash(protocol: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(protocol).encode("utf-8"))


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class StudyProtocol:
    """Validated protocol wrapper; the JSON remains the source of truth."""

    payload: Mapping[str, Any]

    REQUIRED = (
        "format_version", "study_id", "title", "scientific_motivation",
        "hypotheses", "scope_limitations", "claim_language", "dataset",
        "tokenizer", "configurations", "run_matrix", "decision_gates",
        "outcomes", "prohibited_inferences", "evidence_requirements",
    )

    def __post_init__(self) -> None:
        missing = [key for key in self.REQUIRED if key not in self.payload]
        if missing:
            raise ValueError(f"protocol is missing required fields: {missing}")
        if self.payload["study_id"] not in SUPPORTED_STUDY_IDS:
            raise ValueError(f"unexpected study_id: {self.payload['study_id']!r}")
        if not isinstance(self.payload["format_version"], int) or self.payload["format_version"] < 1:
            raise ValueError("protocol format_version must be a positive integer")
        for key in ("hypotheses", "configurations", "run_matrix", "decision_gates", "outcomes", "prohibited_inferences", "evidence_requirements"):
            value = self.payload[key]
            if not isinstance(value, (Mapping, list)) or not value:
                raise ValueError(f"protocol field {key!r} must be non-empty")
        dataset = _require_mapping(self.payload["dataset"], "dataset")
        for key in ("source_data_fingerprint", "subset_manifest_fingerprint", "eligibility"):
            if key not in dataset:
                raise ValueError(f"dataset is missing {key}")
        tokenizer = _require_mapping(self.payload["tokenizer"], "tokenizer")
        for key in ("name", "artifact_fingerprint"):
            if key not in tokenizer:
                raise ValueError(f"tokenizer is missing {key}")

    @classmethod
    def from_path(cls, path: str | Path) -> "StudyProtocol":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(_require_mapping(value, "protocol"))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json(self.payload))

    @property
    def hash(self) -> str:
        return protocol_hash(self.payload)

    def run_spec(
        self,
        run_id: str,
        *,
        amendment_paths: Sequence[str | Path] = (),
    ) -> Mapping[str, Any]:
        matrix = _require_mapping(self.payload["run_matrix"], "run_matrix")
        if run_id in matrix:
            return _require_mapping(matrix[run_id], f"run_matrix.{run_id}")
        additions: dict[str, Mapping[str, Any]] = {}
        for amendment_path in amendment_paths:
            amendment = _require_mapping(
                json.loads(Path(amendment_path).read_text(encoding="utf-8")),
                f"amendment {amendment_path}",
            )
            if amendment.get("format_version") != 1:
                raise ValueError(f"unsupported amendment format: {amendment_path}")
            if amendment.get("preceding_protocol_hash") != self.hash:
                raise ValueError(f"amendment is not linked to this frozen protocol: {amendment_path}")
            changes = _require_mapping(amendment.get("changes"), f"amendment changes {amendment_path}")
            raw_additions = changes.get("run_matrix_additions", {})
            added = _require_mapping(raw_additions, f"amendment run_matrix_additions {amendment_path}")
            for added_id, raw_spec in added.items():
                if not isinstance(added_id, str) or not added_id:
                    raise ValueError(f"amendment contains an invalid run ID: {amendment_path}")
                if added_id in matrix or added_id in additions:
                    raise ValueError(f"amendment attempts to replace run ID {added_id!r}")
                additions[added_id] = _require_mapping(
                    raw_spec, f"amendment run_matrix_additions.{added_id}"
                )
        if run_id not in additions:
            raise ValueError(f"run ID {run_id!r} is not in the frozen protocol or supplied amendments")
        return additions[run_id]

    def validate_run_config(
        self,
        run_id: str,
        config: Mapping[str, Any],
        *,
        amendment_paths: Sequence[str | Path] = (),
    ) -> None:
        """Reject protocol conflicts; extra runtime metadata is permitted."""

        expected = self.run_spec(run_id, amendment_paths=amendment_paths)
        conflicts = []
        for key, value in expected.items():
            if key in config and config[key] != value:
                conflicts.append(f"{key}: expected {value!r}, got {config[key]!r}")
        if conflicts:
            raise ValueError(f"run {run_id!r} conflicts with frozen protocol: {'; '.join(conflicts)}")


def validate_protocol(path: str | Path) -> StudyProtocol:
    return StudyProtocol.from_path(path)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _artifact(root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        display = str(resolved.relative_to(root.resolve()))
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid ledger JSON at line {line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"ledger line {line_number} is not an object")
        events.append(value)
    return events


def _event_hash(event: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in event.items() if key != "event_hash"}
    return sha256_bytes(canonical_json(unsigned).encode("utf-8"))


def _append(root: Path, event: dict[str, Any]) -> dict[str, Any]:
    ledger = root / LEDGER_FILENAME
    prior = _load_ledger(ledger)
    event["preceding_event_hash"] = prior[-1]["event_hash"] if prior else GENESIS_HASH
    event["event_hash"] = _event_hash(event)
    root.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(event) + "\n")
    return event


def _base_event(protocol: StudyProtocol, event_type: str, **extra: Any) -> dict[str, Any]:
    event = {
        "format_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "protocol_hash": protocol.hash,
        "status": extra.pop("status", "recorded"),
    }
    event.update(extra)
    return event


def initialize(protocol_path: str | Path, study_root: str | Path) -> Path:
    protocol = validate_protocol(protocol_path)
    root = Path(study_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / PROTOCOL_FILENAME
    if destination.exists():
        existing = validate_protocol(destination)
        if existing.hash != protocol.hash:
            raise ValueError("study root already contains a different frozen protocol")
    else:
        destination.write_text(canonical_json(protocol.to_dict()) + "\n", encoding="utf-8")
    _append(root, _base_event(protocol, "protocol_initialized", protocol_path=str(destination), protocol_sha256=protocol.hash)) if not (root / LEDGER_FILENAME).exists() else None
    return destination


def record(
    protocol_path: str | Path,
    study_root: str | Path,
    run_id: str,
    paths: Iterable[str | Path],
    *,
    status: str = "completed",
    event_type: str = "run_recorded",
    deviation_reason: str | None = None,
    evidence_tier: str = "confirmatory",
    amendment_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    protocol = validate_protocol(protocol_path)
    protocol.validate_run_config(run_id, {}, amendment_paths=amendment_paths)
    root = Path(study_root)
    if not (root / PROTOCOL_FILENAME).exists():
        initialize(protocol_path, root)
    source_paths = [Path(path) for path in paths]
    if not source_paths:
        raise ValueError("record requires at least one artifact path")
    artifacts: list[dict[str, Any]] = []
    for source in source_paths:
        if source.is_dir():
            artifacts.extend(_artifact(root, child) for child in sorted(source.rglob("*")) if child.is_file())
        else:
            artifacts.append(_artifact(root, source))
    if not artifacts:
        raise ValueError("record found no files")
    run_manifest = next((item for item in artifacts if Path(item["path"]).name in {"run_manifest.json", "colab_run_manifest.json"}), None)
    provenance: dict[str, Any] = {}
    if run_manifest:
        manifest_path = root / run_manifest["path"]
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                provenance = {key: loaded[key] for key in ("model_config", "dataset_fingerprint", "code_commit", "device", "device_name", "validation", "optimizer_steps") if key in loaded}
        except (OSError, json.JSONDecodeError):
            pass
    repo = Path(__file__).resolve().parents[5]
    event = _base_event(protocol, event_type, run_id=run_id, status=status, evidence_tier=evidence_tier, artifacts=artifacts, provenance=provenance, code={"commit": _git(repo, "rev-parse", "HEAD"), "dirty": bool(_git(repo, "status", "--short"))}, environment={"python": platform.python_version(), "platform": platform.platform()}, cli_args=list(sys.argv), deviation_reason=deviation_reason)
    return _append(root, event)


def amend(protocol_path: str | Path, study_root: str | Path, amendment_id: str, rationale: str, classification: str, expected_impact: str, changes: Mapping[str, Any]) -> Path:
    protocol = validate_protocol(protocol_path)
    if classification not in {"engineering", "exploratory", "confirmatory"}:
        raise ValueError("classification must be engineering, exploratory, or confirmatory")
    root = Path(study_root)
    prior_hash = protocol.hash
    amendment = {"format_version": 1, "amendment_id": amendment_id, "created_at": datetime.now(timezone.utc).isoformat(), "preceding_protocol_hash": prior_hash, "rationale": rationale, "expected_impact": expected_impact, "classification": classification, "changes": dict(changes)}
    directory = root / "amendments"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{amendment_id}.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(canonical_json(amendment) + "\n", encoding="utf-8")
    _append(root, _base_event(protocol, "protocol_amended", amendment_id=amendment_id, amendment_path=str(destination.relative_to(root)), amendment_hash=protocol_hash(amendment), classification=classification, status="recorded"))
    return destination


def verify(protocol_path: str | Path, study_root: str | Path) -> dict[str, Any]:
    protocol = validate_protocol(protocol_path)
    root = Path(study_root)
    copied = root / PROTOCOL_FILENAME
    if not copied.exists() or validate_protocol(copied).hash != protocol.hash:
        raise ValueError("study-root protocol hash does not match supplied protocol")
    events = _load_ledger(root / LEDGER_FILENAME)
    previous = GENESIS_HASH
    checked_artifacts = 0
    for index, event in enumerate(events, 1):
        if event.get("preceding_event_hash") != previous:
            raise ValueError(f"ledger chain broken at event {index}")
        if event.get("event_hash") != _event_hash(event):
            raise ValueError(f"ledger event hash mismatch at event {index}")
        if event.get("protocol_hash") != protocol.hash:
            raise ValueError(f"protocol hash mismatch at event {index}")
        for artifact in event.get("artifacts", []):
            path = Path(artifact["path"])
            actual = sha256_file(path if path.is_absolute() else root / path)
            if actual != artifact["sha256"]:
                raise ValueError(f"artifact hash mismatch: {artifact['path']}")
            checked_artifacts += 1
        if event.get("event_type") == "protocol_amended":
            amendment_path = event.get("amendment_path")
            if not isinstance(amendment_path, str):
                raise ValueError(f"amendment path missing at event {index}")
            amendment_file = root / amendment_path
            if not amendment_file.is_file():
                raise ValueError(f"amendment file missing: {amendment_path}")
            amendment = json.loads(amendment_file.read_text(encoding="utf-8"))
            if protocol_hash(amendment) != event.get("amendment_hash"):
                raise ValueError(f"amendment hash mismatch: {amendment_path}")
        previous = event["event_hash"]
    return {"protocol_hash": protocol.hash, "events": len(events), "artifacts": checked_artifacts, "valid": True}


def report(protocol_path: str | Path, study_root: str | Path, output_dir: str | Path | None = None) -> dict[str, Path]:
    protocol = validate_protocol(protocol_path)
    root = Path(study_root)
    verification = verify(protocol_path, root)
    events = _load_ledger(root / LEDGER_FILENAME)
    invalid = [event for event in events if event.get("status") in {"failed", "invalid", "excluded"}]
    confirmatory = [event for event in events if event.get("evidence_tier") == "confirmatory" and event.get("status") == "completed"]
    required = list(protocol.payload["evidence_requirements"])
    covered = {str(event.get("run_id")) for event in confirmatory}
    missing = [item for item in required if isinstance(item, str) and item not in covered]
    if missing:
        raise ValueError(f"cannot compile final report; missing confirmatory evidence: {missing}")
    if invalid:
        raise ValueError("cannot compile final report while failed, invalid, or excluded runs are present")
    output = Path(output_dir) if output_dir else root
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"format_version": 1, "study_id": protocol.payload["study_id"], "protocol_hash": protocol.hash, "verification": verification, "confirmatory_run_ids": sorted(covered), "event_count": len(events), "claim_language": protocol.payload["claim_language"], "prohibited_inferences": protocol.payload["prohibited_inferences"]}
    manifest_path = output / "MANUSCRIPT_MANIFEST.json"
    report_path = output / "STUDY_REPORT.md"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    lines = [f"# {protocol.payload['title']}", "", f"Protocol SHA-256: `{protocol.hash}`", "", "## Evidence status", "", f"- Verified ledger events: {len(events)}", f"- Confirmatory runs: {', '.join(sorted(covered))}", "- All required confirmatory evidence is present.", "", "## Claim-evidence table", "", "| Claim area | Evidence |", "| --- | --- |"]
    for item in protocol.payload["evidence_requirements"]:
        lines.append(f"| {item} | Recorded confirmatory event |")
    lines.extend(["", "## Boundaries", "", *[f"- {item}" for item in protocol.payload["prohibited_inferences"]], ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"report": report_path, "manifest": manifest_path}


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage C frozen protocol and append-only experiment ledger")
    parser.add_argument("command", choices=("validate", "initialize", "record", "amend", "verify", "report"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-amendment", type=Path, action="append", default=[])
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--status", default="completed")
    parser.add_argument("--evidence-tier", default="confirmatory")
    parser.add_argument("--amendment-id")
    parser.add_argument("--rationale")
    parser.add_argument("--classification")
    parser.add_argument("--expected-impact")
    parser.add_argument("--changes", default="{}")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "validate":
        result = {"protocol_hash": validate_protocol(args.protocol).hash, "valid": True}
    elif args.command == "initialize":
        result = {"protocol": str(initialize(args.protocol, args.study_root))}
    elif args.command == "record":
        if not args.run_id:
            parser.error("record requires --run-id")
        result = record(
            args.protocol,
            args.study_root,
            args.run_id,
            args.artifact,
            status=args.status,
            evidence_tier=args.evidence_tier,
            amendment_paths=args.protocol_amendment,
        )
    elif args.command == "amend":
        values = (args.amendment_id, args.rationale, args.classification, args.expected_impact)
        if any(value is None for value in values):
            parser.error("amend requires --amendment-id, --rationale, --classification, and --expected-impact")
        result = {"amendment": str(amend(args.protocol, args.study_root, args.amendment_id, args.rationale, args.classification, args.expected_impact, json.loads(args.changes)))}
    elif args.command == "verify":
        result = verify(args.protocol, args.study_root)
    else:
        result = {key: str(value) for key, value in report(args.protocol, args.study_root, args.output_dir).items()}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


main = _cli


if __name__ == "__main__":
    raise SystemExit(main())
