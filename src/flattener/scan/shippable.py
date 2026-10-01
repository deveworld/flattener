"""Ship gate: which dewarp weights may be the scanner's default model.

Weights are shippable when their whole lineage is own release-track training that
started from scratch on release data. ``shippable`` walks the lineage record of a
checkpoint (the ``lineage`` list that ``flattener.dewarp.train`` writes into every
checkpoint and sidecar: a start entry, then training steps; a ``kind: "checkpoint"``
start holds the parent's own lineage) and requires:

* the record's ``track``, every parent checkpoint's ``track`` and every training step's
  ``track`` are ``release``;
* the chain starts from ``scratch``. UVDoc, SyntheticDoc or other outside weights anywhere
  in the chain fail the gate. ``train`` lets a release run start from the sha256-matched
  UVDoc weights, so the track alone does not settle it;
* the data spec of every training step passes ``require_track(spec, "release")``. Specs
  are matched by their content sha256 (``DataSpec.sha256``): the record's own spec comes
  from the record, a parent checkpoint start may carry its own (``data_spec``), and
  ``specs`` supplies the others. A step whose spec is not at hand fails the gate
  (``unverified_data_spec``);
* the license manifest (``data_sources``, when the record has one) lists release sources
  whose license classes are all on ``RELEASE_LICENSE_KEYS``.

A ``kind: "soup"`` step (``tools/soup.py``: weights blended from two checkpoints) names
its ``parents`` as ``kind: "checkpoint"`` entries with their own lineage and data spec;
each parent goes through the same rules as a checkpoint start.

Accepted records: a checkpoint payload or sidecar, ``Dewarper.info`` (it has no data
spec content, so pass the spec through ``specs``), a run-info file (``init`` dict,
``data_spec_content``; a ``checkpoint`` init keeps the parent lineage as
``lineage_records``) and a training ``config.json`` (its start only; without spec
content it never passes).

The verdict's ``category`` is ``release`` (shippable), ``research`` (own weights that
fail a rule, such as models trained further from the UVDoc weights) or ``external``
(UVDoc MIT, the SyntheticDoc checkpoints and any other weights without a lineage record).
``reasons`` holds reason codes from ``REASONS``; ``initialization`` names the chain's
start as ``evidence.initialization`` does. A file is gated before it is deserialized.

``gate_verdict`` applies the same rules to the provenance record of a page gate
(``regions.context``, ``region-context-1``), whose data is a list of sources with SPDX
licenses rather than a data spec. OrientNet records go through ``shippable`` itself
(``scan.orient.release_problems``).

``load_weights`` refuses weights that fail the gate unless ``research=True``. With it,
the weights load and ``info["ship_gate"]`` carries ``research: true``, so every result
made with them is stamped.

``python -m flattener.scan.shippable PATH...`` prints one verdict per path: a ``.json``
path is read as a record, any other path is gated by its sidecar (``gate_file``). The
exit status is 0 when every path is shippable, 1 when any is not and 2 on a usage or
input error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import FlattenerError, Status
from ..dewarp.checkpoint import sidecar_path
from ..dewarp.data import SPEC_SCHEMA, DataSpec, license_key, release_license_ok, require_track
from ..dewarp.model import UVDOC_WEIGHTS_SHA256, sha256_file
from ..dewarp.predict import Dewarper, load_dewarper

GATE_SCHEMA = "flattener-ship-gate-1"
CATEGORIES = ("release", "research", "external")
# Start or record kinds that name weights trained outside Flattener.
EXTERNAL_KINDS = frozenset({"uvdoc", "syntheticdoc", "external", "external_pretrained"})
OWN_PROVENANCES = frozenset({"own_checkpoint", "own_previous_model"})
MAX_DEPTH = 64  # parent checkpoints followed before the gate gives up
# License terms a page gate's data may carry besides RELEASE_LICENSE_KEYS: the OFL-1.1
# fonts that Flattener's own scene generators draw. Data Flattener generates itself is
# under Apache or OFL.
GATE_EXTRA_LICENSE_KEYS = frozenset({"ofl"})
# Known outside weights files by sha256: UVDoc best_model.pkl (MIT) and the SyntheticDoc
# checkpoints original_31x45.ckpt and highres_121x177.ckpt.
EXTERNAL_WEIGHTS_SHA256 = {
    UVDOC_WEIGHTS_SHA256: "uvdoc",
    "557af82579157c000f3cbd27c79d02b2a7cfea177849dbb04b9499ff88c81b1f": "syntheticdoc",
    "a9b69148fb3a8bf309757106f489df17a339a03f298f59d882f5ff51ba65a91d": "syntheticdoc",
}
REASONS = {
    "external_weights": "the weights were trained outside Flattener",
    "no_lineage_record": "the weights have no lineage record (no checkpoint sidecar)",
    "sidecar_mismatch": "the sidecar sha256 differs from the file",
    "external_init": "the chain starts from outside weights",
    "unknown_init": "the chain start is missing or of an unknown kind",
    "unknown_lineage_entry": "a lineage has no training step or an entry of unknown kind",
    "not_release_track": "the record, a parent checkpoint or a training step does not "
    "record track release",
    "research_data": "a data spec fails require_track(spec, 'release') or is invalid",
    "unverified_data_spec": "no data spec with a training step's sha256 is at hand",
    "data_spec_mismatch": "the record's data spec does not hash to its data_spec_sha256",
    "data_license_not_release": "the license manifest lists a research source or a license "
    "class off RELEASE_LICENSE_KEYS",
    "lineage_mismatch": "the record's fields disagree with its lineage",
    "lineage_too_deep": f"the chain has more than {MAX_DEPTH} parent checkpoints",
}


@dataclass(frozen=True, slots=True)
class ShipVerdict:
    """The gate's answer for one set of weights (see the module docstring)."""

    shippable: bool
    category: str
    reasons: tuple[str, ...] = ()
    sha256: str | None = None
    track: str | None = None
    initialization: str | None = None
    chain: tuple[dict[str, Any], ...] = ()

    @property
    def research(self) -> bool:
        """True when results made with these weights carry the research stamp."""
        return not self.shippable

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GATE_SCHEMA,
            "shippable": self.shippable,
            "research": self.research,
            "category": self.category,
            "reasons": list(self.reasons),
            "sha256": self.sha256,
            "track": self.track,
            "initialization": self.initialization,
            "chain": [dict(node) for node in self.chain],
        }


def _spec_verdict(content: Any) -> tuple[str | None, str]:
    """(sha256, "release" or "research") of a data spec; sha256 is None for an invalid one."""
    try:
        spec = content if isinstance(content, DataSpec) else DataSpec.from_dict(content)
    except (FlattenerError, TypeError, ValueError):
        return None, "research"
    try:
        require_track(spec, "release")
    except FlattenerError:
        return spec.sha256, "research"
    return spec.sha256, "release"


def _spec_content(item: Any) -> Any:
    """The data spec an item holds: a spec itself, or a record's ``data_spec`` or
    ``data_spec_content``; None otherwise."""
    if isinstance(item, DataSpec):
        return item
    if not isinstance(item, Mapping):
        return None
    if item.get("schema") == SPEC_SCHEMA:
        return item
    for key in ("data_spec", "data_spec_content"):
        if isinstance(item.get(key), Mapping):
            return item[key]
    return None


def _embedded_specs(lineage: Any, depth: int = 0) -> Iterator[Any]:
    """Data specs carried by parent checkpoint starts (``data_spec``). A spec counts for a
    training step only when its content hashes to that step's ``data_spec_sha256``."""
    if not isinstance(lineage, list) or depth > MAX_DEPTH:
        return
    for entry in lineage:
        if isinstance(entry, Mapping) and entry.get("kind") == "checkpoint":
            if isinstance(entry.get("data_spec"), Mapping):
                yield entry["data_spec"]
            yield from _embedded_specs(entry.get("lineage"), depth + 1)
        elif isinstance(entry, Mapping) and entry.get("kind") == "soup":
            yield from _embedded_specs(entry.get("parents"), depth + 1)


def _spec_index(items: Iterable[Any]) -> dict[str, str]:
    index: dict[str, str] = {}
    for item in items:
        content = _spec_content(item)
        if content is None:
            continue
        digest, verdict = _spec_verdict(content)
        if digest is not None:
            index[digest] = verdict
    return index


def _record_sha256(record: Mapping[str, Any]) -> str | None:
    digest = record.get("sha256")
    if isinstance(digest, str):
        return digest
    result = record.get("result")  # run-info: the checkpoint the run picked
    best = result.get("best_checkpoint") if isinstance(result, Mapping) else None
    if isinstance(best, Mapping) and isinstance(best.get("sha256"), str):
        return best["sha256"]
    return None


def _external_kind(record: Mapping[str, Any], digest: str | None) -> str | None:
    """The kind of outside weights a record describes; None for own or unknown records."""
    if digest in EXTERNAL_WEIGHTS_SHA256:
        return EXTERNAL_WEIGHTS_SHA256[digest]
    kind = record.get("kind") if isinstance(record.get("kind"), str) else None
    if record.get("track") == "external" or record.get("provenance") == "external_pretrained":
        return kind if kind is not None and kind != "checkpoint" else "external"
    if kind in EXTERNAL_KINDS and not isinstance(record.get("lineage"), list):
        return kind
    return None


@dataclass(slots=True)
class _Own:
    """An own record in one shape: the lineage list, its spec and manifest."""

    track: Any
    lineage: Any
    spec: Any
    spec_sha256: Any
    sources: Any
    reasons: list[str]


def _run_info_start(init: Mapping[str, Any]) -> dict[str, Any]:
    if init.get("kind") != "checkpoint":
        return dict(init)
    return {
        "kind": "checkpoint",
        "sha256": init.get("sha256"),
        "track": init.get("track"),
        "data_spec_sha256": init.get("data_spec_sha256"),
        "lineage": init.get("lineage_records", init.get("lineage")),
    }


def _own_record(record: Mapping[str, Any]) -> _Own | None:
    """``record`` as an own lineage record, or None when it has none."""
    track, digest = record.get("track"), record.get("data_spec_sha256")
    if isinstance(record.get("lineage"), list):  # checkpoint payload, sidecar, Dewarper.info
        spec = record.get("data_spec")
        return _Own(track, record["lineage"], spec, digest, record.get("data_sources"), [])
    step = {"kind": "train", "track": track, "data_spec_sha256": digest}
    init = record.get("init")
    if isinstance(init, Mapping):  # run-info
        reasons = []
        config = record.get("config")
        if isinstance(config, Mapping) and (
            config.get("init", init.get("kind")) != init.get("kind")
            or config.get("track", track) != track
        ):
            reasons.append("lineage_mismatch")
        lineage = [_run_info_start(init), step]
        return _Own(track, lineage, record.get("data_spec_content"), digest, None, reasons)
    if isinstance(init, str) and "output_dir" in record:  # training config.json
        start = {"kind": init}
        if init == "checkpoint":
            start["path"] = record.get("init_path")
        return _Own(track, [start, step], None, None, None, [])
    return None


class _Walk:
    """Collects reasons and chain nodes while following a lineage to its start."""

    def __init__(self, specs: dict[str, str], reasons: list[str]) -> None:
        self.specs = specs
        self.reasons = reasons
        self.chain: list[dict[str, Any]] = []
        self.start_kind: str | None = None

    def fail(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)

    def lineage(self, lineage: Any, depth: int, node: dict[str, Any], spec_sha256: Any) -> None:
        """Check the training steps of the weights ``node`` names, then their start."""
        self.chain.append(node)
        if not isinstance(lineage, list) or not lineage or not isinstance(lineage[0], Mapping):
            self.fail("unknown_init")
            return
        steps = lineage[1:]
        if not steps:
            self.fail("unknown_lineage_entry")
        node["training"] = [self.step(step, depth) for step in steps]
        trained = [step for step in node["training"] if step["kind"] == "train"]
        last = trained[-1]["data_spec_sha256"] if trained else None
        if isinstance(spec_sha256, str) and last != spec_sha256:
            self.fail("lineage_mismatch")
        self.start(lineage[0], depth + 1)

    def step(self, step: Any, depth: int) -> dict[str, Any]:
        if isinstance(step, Mapping) and step.get("kind") == "soup":
            return self.soup(step, depth)
        if not isinstance(step, Mapping) or step.get("kind") != "train":
            self.fail("unknown_lineage_entry")
            kind = step.get("kind") if isinstance(step, Mapping) else None
            return {"kind": kind if isinstance(kind, str) else None}
        track, digest = step.get("track"), step.get("data_spec_sha256")
        if track != "release":
            self.fail("not_release_track")
        data = self.specs.get(digest, "unverified") if isinstance(digest, str) else "unverified"
        if data == "unverified":
            self.fail("unverified_data_spec")
        elif data != "release":
            self.fail("research_data")
        return {"kind": "train", "track": track, "data_spec_sha256": digest, "data": data}

    def soup(self, step: Mapping[str, Any], depth: int) -> dict[str, Any]:
        """A blend of checkpoints: its track, then each parent as a checkpoint start."""
        if step.get("track") != "release":
            self.fail("not_release_track")
        parents = step.get("parents")
        if not isinstance(parents, list) or len(parents) < 2:
            self.fail("unknown_lineage_entry")
            return {"kind": "soup", "parents": []}
        for parent in parents:
            if not isinstance(parent, Mapping) or parent.get("kind") != "checkpoint":
                self.fail("unknown_lineage_entry")
                continue
            self.start(parent, depth + 1)
        hashes = [parent.get("sha256") for parent in parents if isinstance(parent, Mapping)]
        return {"kind": "soup", "parents": hashes}

    def start(self, start: Mapping[str, Any], depth: int) -> None:
        kind = start.get("kind")
        if kind == "checkpoint":
            if depth > MAX_DEPTH:
                self.fail("lineage_too_deep")
                self.chain.append({"depth": depth, "kind": kind, "sha256": start.get("sha256")})
                return
            track = start.get("track")
            if track != "release":
                self.fail("not_release_track")
            node = {"depth": depth, "kind": kind, "sha256": start.get("sha256"), "track": track}
            self.lineage(start.get("lineage"), depth, node, start.get("data_spec_sha256"))
            return
        self.start_kind = kind if isinstance(kind, str) else None
        node = {"depth": depth, "kind": self.start_kind}
        if self.start_kind in EXTERNAL_KINDS:
            self.fail("external_init")
            node.update(sha256=start.get("sha256"), category="external")
        elif kind != "scratch":
            self.fail("unknown_init")
        self.chain.append(node)


def _sources_release(sources: Any) -> bool:
    """True when every source of a ``data_sources`` manifest is release-clean."""
    if not isinstance(sources, Mapping):
        return False
    for split in sources.values():
        for source in split if isinstance(split, list) else [None]:
            if not isinstance(source, Mapping) or source.get("track") != "release":
                return False
            classes = source.get("license_classes")
            if not isinstance(classes, Mapping) or not all(
                isinstance(name, str) and release_license_ok(name) for name in classes
            ):
                return False
    return True


def _initialization(start_kind: str | None) -> str:
    if start_kind == "scratch":
        return "scratch"
    return "external_pretrained" if start_kind in EXTERNAL_KINDS else "unknown"


def shippable(record: Mapping[str, Any], *, specs: Iterable[Any] = ()) -> ShipVerdict:
    """The ship gate verdict of the weights ``record`` describes (see the module docstring).

    ``specs`` adds data specs for parent checkpoints: spec dicts, ``DataSpec`` objects or
    records that hold one (sidecars, run-info files). Only specs whose content hashes to
    a training step's ``data_spec_sha256`` count, so extra entries do no harm.
    """
    if not isinstance(record, Mapping):
        raise FlattenerError(Status.INPUT_ERROR, "invalid_record", "record must be a mapping")
    digest = _record_sha256(record)
    external = _external_kind(record, digest)
    if external is not None:
        node = {"depth": 0, "kind": external, "sha256": digest, "category": "external"}
        return ShipVerdict(
            False, "external", ("external_weights",), digest, "external", None, (node,)
        )
    own = _own_record(record)
    if own is None:
        # Own weights without a lineage record (the earlier coordinate model) stay research.
        provenance = record.get("provenance")
        mine = record.get("kind") == "checkpoint" or (
            isinstance(provenance, str) and provenance in OWN_PROVENANCES
        )
        kind = record.get("kind") if isinstance(record.get("kind"), str) else None
        node = {"depth": 0, "kind": kind, "sha256": digest}
        category = "research" if mine else "external"
        return ShipVerdict(False, category, ("no_lineage_record",), digest, None, None, (node,))
    reasons = own.reasons
    walk = _Walk(_spec_index([own.spec, *_embedded_specs(own.lineage), *specs]), reasons)
    if own.track != "release":
        walk.fail("not_release_track")
    if own.spec is not None:
        spec_sha256, _ = _spec_verdict(own.spec)
        if spec_sha256 is None:
            walk.fail("research_data")
        elif isinstance(own.spec_sha256, str) and spec_sha256 != own.spec_sha256:
            walk.fail("data_spec_mismatch")
    if own.sources is not None and not _sources_release(own.sources):
        walk.fail("data_license_not_release")
    node = {"depth": 0, "kind": "checkpoint", "sha256": digest, "track": own.track}
    walk.lineage(own.lineage, 0, node, own.spec_sha256)
    ok = not reasons
    track = own.track if isinstance(own.track, str) else None
    return ShipVerdict(
        ok,
        "release" if ok else "research",
        tuple(reasons),
        digest,
        track,
        _initialization(walk.start_kind),
        tuple(walk.chain),
    )


def _license_release(text: Any) -> bool:
    """True when every term of an SPDX license expression is release data."""
    if not isinstance(text, str) or not text.strip():
        return False
    terms = re.split(r"\s+(?:AND|OR|WITH)\s+", text.replace("(", " ").replace(")", " ").strip())
    return all(
        release_license_ok(term) or license_key(term) in GATE_EXTRA_LICENSE_KEYS for term in terms
    )


def _gate_data_reason(record: Mapping[str, Any]) -> str | None:
    """Why a page gate record's data is not release data (``None`` when it is): a
    ``data_spec`` (dewarp schema) must pass ``require_track``; a ``data`` list needs every
    entry release-track (when it says) with a release license."""
    spec = record.get("data_spec")
    if isinstance(spec, Mapping):
        return None if _spec_verdict(spec)[1] == "release" else "research_data"
    data = record.get("data")
    if not isinstance(data, list) or not data:
        return "unverified_data_spec"
    for entry in data:
        if not isinstance(entry, Mapping):
            return "data_license_not_release"
        if entry.get("track", "release") != "release" or not _license_release(entry.get("license")):
            return "data_license_not_release"
    return None


def gate_verdict(provenance: Mapping[str, Any], *, sha256: str | None = None) -> ShipVerdict:
    """The ship gate verdict of a page gate from its ``provenance`` record.

    Shippable when ``track`` is release, ``lineage`` starts from scratch (through parent
    checkpoints that are release-track and carry their own release data record), every
    training step in it is release-track, and the data (``_gate_data_reason``) is release
    data: SmartDoc CC BY and the own generators (Apache-2.0, OFL-1.1) pass."""
    if not isinstance(provenance, Mapping):
        return ShipVerdict(False, "external", ("no_lineage_record",), sha256)
    reasons: list[str] = []

    def fail(reason: str | None) -> None:
        if reason is not None and reason not in reasons:
            reasons.append(reason)

    track = provenance.get("track")
    if track != "release":
        fail("not_release_track")
    fail(_gate_data_reason(provenance))
    chain: list[dict[str, Any]] = [{"depth": 0, "kind": "page_gate", "sha256": sha256}]
    lineage, start_kind = provenance.get("lineage"), None
    for depth in range(MAX_DEPTH + 2):
        if depth > MAX_DEPTH:
            fail("lineage_too_deep")
            break
        if not isinstance(lineage, list) or not lineage or not isinstance(lineage[0], Mapping):
            fail("unknown_init")
            break
        for step in lineage[1:]:
            if not isinstance(step, Mapping) or step.get("kind") != "train":
                fail("unknown_lineage_entry")
            elif step.get("track") != "release":
                fail("not_release_track")
        start = lineage[0]
        kind = start.get("kind")
        if kind != "checkpoint":
            start_kind = kind if isinstance(kind, str) else None
            if start_kind in EXTERNAL_KINDS:
                fail("external_init")
            elif kind != "scratch":
                fail("unknown_init")
            chain.append({"depth": depth + 1, "kind": start_kind})
            break
        if start.get("track") != "release":
            fail("not_release_track")
        fail(_gate_data_reason(start))
        chain.append(
            {
                "depth": depth + 1,
                "kind": kind,
                "sha256": start.get("sha256"),
                "track": start.get("track"),
            }
        )
        lineage = start.get("lineage")
    ok = not reasons
    category = "release" if ok else ("external" if "external_init" in reasons else "research")
    return ShipVerdict(
        ok,
        category,
        tuple(reasons),
        sha256,
        track if isinstance(track, str) else None,
        _initialization(start_kind),
        tuple(chain),
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise FlattenerError(Status.IO_ERROR, "unreadable_record", f"{path}: {error}") from error
    if not isinstance(value, dict):
        raise FlattenerError(Status.INPUT_ERROR, "invalid_record", f"{path}: not a JSON object")
    return value


def parent_sidecars(lineage: Any, depth: int = 0) -> Iterator[dict[str, Any]]:
    """Sidecars of parent checkpoints whose recorded path exists on this machine."""
    if not isinstance(lineage, list) or depth > MAX_DEPTH:
        return
    for entry in lineage:
        if not isinstance(entry, Mapping) or entry.get("kind") != "checkpoint":
            continue
        path = entry.get("path")
        if isinstance(path, str) and sidecar_path(Path(path)).is_file():
            try:
                yield _read_json(sidecar_path(Path(path)))
            except FlattenerError:
                pass
        yield from parent_sidecars(entry.get("lineage"), depth + 1)


def gate_file(path: str | Path, *, specs: Iterable[Any] = ()) -> ShipVerdict:
    """Gate a weights file by its sha256 and sidecar (``<file>.json``) without loading it.

    Sidecars of parent checkpoints found at their recorded paths add their data specs.
    """
    path = Path(path)
    if not path.is_file():
        raise FlattenerError(Status.IO_ERROR, "missing_checkpoint", f"{path} does not exist")
    digest = sha256_file(path)
    sidecar = sidecar_path(path)
    if digest in EXTERNAL_WEIGHTS_SHA256 or not sidecar.is_file():
        return shippable({"sha256": digest})
    record = _read_json(sidecar)
    if record.get("sha256") != digest:
        node = {"depth": 0, "kind": "checkpoint", "sha256": digest}
        track = record.get("track") if isinstance(record.get("track"), str) else None
        return ShipVerdict(False, "research", ("sidecar_mismatch",), digest, track, None, (node,))
    return shippable(record, specs=[*specs, *parent_sidecars(record.get("lineage"))])


def load_weights(
    path: str | Path,
    *,
    research: bool = False,
    device: str = "cpu",
    specs: Iterable[Any] = (),
) -> Dewarper:
    """Load dewarp weights through the ship gate.

    Weights that fail the gate raise ``FlattenerError`` (``model_error``,
    ``not_shippable``, the verdict in ``metadata["ship_gate"]``) unless ``research`` is
    true. The loaded ``Dewarper.info["ship_gate"]`` holds the verdict; its ``research``
    field stamps every result made with non-shippable weights.
    """
    verdict = gate_file(path, specs=specs)
    if not verdict.shippable and not research:
        raise FlattenerError(
            Status.MODEL_ERROR,
            "not_shippable",
            f"{path}: {verdict.category} weights fail the ship gate ({', '.join(verdict.reasons)});"
            " load them with research=True for comparisons",
            metadata={"ship_gate": verdict.to_dict()},
        )
    dewarper = load_dewarper(path, device=device)
    if dewarper.info.get("sha256") != verdict.sha256:
        raise FlattenerError(
            Status.IO_ERROR, "checksum_mismatch", f"{path} changed while it was loaded"
        )
    dewarper.info["ship_gate"] = verdict.to_dict()
    return dewarper


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m flattener.scan.shippable",
        description="Print the ship gate verdict of weights files or lineage records.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="weights files (gated by their sidecar) or .json records "
        "(checkpoint sidecars, run-info or config files)",
    )
    parser.add_argument(
        "--spec",
        action="append",
        type=Path,
        default=[],
        help="JSON data spec, or a record holding one, for a parent checkpoint (repeatable)",
    )
    args = parser.parse_args(argv)
    verdicts = []
    try:
        specs = [_read_json(path) for path in args.spec]
        for path in args.paths:
            if path.suffix == ".json":
                verdict = shippable(_read_json(path), specs=specs)
            else:
                verdict = gate_file(path, specs=specs)
            verdicts.append(verdict)
            print(json.dumps({"path": str(path), **verdict.to_dict()}, sort_keys=True))
    except FlattenerError as error:
        failure = {"status": str(error.status), "reason_code": error.reason_code}
        print(json.dumps({**failure, "message": error.message}), file=sys.stderr)
        return 2
    return 0 if all(verdict.shippable for verdict in verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
