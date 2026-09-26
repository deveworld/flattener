"""Step 4: our own train/val split, disjoint in document group AND mesh group.

The official SyntheticDoc splits share meshes (memo 3.4), so the pool (official
train + val, status ``success``) is re-split:

1. Document groups (arXiv paper, OpenStax/DOAB book, IMSLP work, RealKIE
   document, single page for the synthetic sources) are stratified by
   (document source, document license class). Within each stratum groups are
   taken in a salted SHA-256 order until ~``val_fraction`` of its samples.
2. ``val`` = every pool sample whose document group was chosen.
3. ``train`` = pool samples whose document group is not chosen AND whose mesh
   group (``<family>/<mesh id>``) is not used by any val sample.
   The rest (other documents rendered on a val mesh) is ``excluded_overlap``.
4. Official test samples are ``official_test`` and never used for training;
   ``test_overlap`` flags (1 = mesh, 2 = document group) mark pool samples
   that share a group with the official test split.

Deterministic for a given index and salt, so re-running never reshuffles.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from sdcommon import Layout, State, log, stable_unit, utcnow, write_json_atomic

SPLIT_SALT = "flattener-syntheticdoc-split-v1"


def choose_val_groups(
    strata: dict[tuple[str, str], Counter], val_fraction: float, salt: str
) -> set[str]:
    chosen: set[str] = set()
    for key in sorted(strata):
        groups = strata[key]
        target = val_fraction * sum(groups.values())
        taken = 0
        for group in sorted(groups, key=lambda g: (stable_unit(g, salt), g)):
            if taken >= target:
                break
            if taken + groups[group] > 1.5 * target:
                continue  # a group far larger than the stratum's quota; try the next one
            chosen.add(group)
            taken += groups[group]
    return chosen


def assign(
    rows: list[tuple[str, str, str, str, str, str | None, str | None]],
    *,
    val_fraction: float = 0.02,
    salt: str = SPLIT_SALT,
) -> tuple[dict[str, tuple[str, int]], dict[str, Any]]:
    """rows: (uid, official split, status, doc_source, doc_class, doc_group, mesh_group)."""
    pool = [r for r in rows if r[1] in ("train", "val") and r[2] == "success" and r[5] and r[6]]
    strata: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for _, _, _, source, doc_class, group, _ in pool:
        strata[(source or "", doc_class or "")][group] += 1
    val_groups = choose_val_groups(strata, val_fraction, salt)
    val_meshes = {r[6] for r in pool if r[5] in val_groups}
    test = [r for r in rows if r[1] == "test"]
    test_meshes = {r[6] for r in test if r[6]}
    test_docs = {r[5] for r in test if r[5]}
    pool_ids = {r[0] for r in pool}
    out: dict[str, tuple[str, int]] = {}
    for uid, split, _, _, _, doc, mesh in rows:
        flags = (1 if mesh in test_meshes else 0) | (2 if doc in test_docs else 0)
        if split == "test":
            out[uid] = ("official_test", 0)
        elif uid not in pool_ids:
            out[uid] = ("excluded_invalid", flags)
        elif doc in val_groups:
            out[uid] = ("val", flags)
        elif mesh in val_meshes:
            out[uid] = ("excluded_overlap", flags)
        else:
            out[uid] = ("train", flags)
    by_uid = {r[0]: r for r in rows}
    train_docs = {by_uid[u][5] for u, (s, _) in out.items() if s == "train"}
    train_meshes = {by_uid[u][6] for u, (s, _) in out.items() if s == "train"}
    val_docs = {by_uid[u][5] for u, (s, _) in out.items() if s == "val"}
    leaks = {
        "doc_groups_train_and_val": len(train_docs & val_docs),
        "mesh_groups_train_and_val": len(train_meshes & val_meshes),
    }
    if any(leaks.values()):
        raise AssertionError(f"split leaks: {leaks}")
    info = {
        "val_doc_groups": len(val_groups),
        "val_mesh_groups": len(val_meshes),
        "doc_groups_total": sum(len(g) for g in strata.values()),
        "strata": {
            f"{k[0]}|{k[1]}": {
                "groups": len(v),
                "samples": sum(v.values()),
                "val_groups": sum(1 for g in v if g in val_groups),
                "val_samples": sum(n for g, n in v.items() if g in val_groups),
            }
            for k, v in sorted(strata.items())
        },
        "leak_check": leaks,
        "official_test_mesh_groups": len(test_meshes),
        "official_test_doc_groups": len(test_docs),
    }
    return out, info


def resplit(state: State, layout: Layout, *, val_fraction: float = 0.02) -> dict[str, Any]:
    rows = state.query(
        "SELECT uid, split, status, doc_source, doc_class, doc_group, mesh_group FROM samples"
    )
    out, info = assign(rows, val_fraction=val_fraction)
    updates = [(split, flags, uid) for uid, (split, flags) in out.items()]
    for k in range(0, len(updates), 50000):
        state.executemany(
            "UPDATE samples SET our_split=?, test_overlap=? WHERE uid=?", updates[k : k + 50000]
        )

    def table(sql: str) -> dict[str, dict[str, int]]:
        res: dict[str, dict[str, int]] = defaultdict(dict)
        for a, b, n in state.query(sql):
            res[str(a)][str(b)] = n
        return {k: dict(sorted(v.items())) for k, v in sorted(res.items())}

    report = {
        "schema": "flattener-syntheticdoc-split-report-1",
        "track": "research",
        "note": "Split is defined over all classes; the release track uses its clean subset.",
        "generated_at": utcnow(),
        "method": __doc__.strip().splitlines()[0],
        "params": {
            "val_fraction": val_fraction,
            "salt": SPLIT_SALT,
            "pool": "official train + val, status success",
        },
        **info,
        "our_split_x_class": table(
            "SELECT our_split, lic_class, COUNT(*) FROM samples GROUP BY our_split, lic_class"
        ),
        "our_split_x_official_split": table(
            "SELECT our_split, split, COUNT(*) FROM samples GROUP BY our_split, split"
        ),
        "our_split_x_family": table(
            "SELECT our_split, mesh_family, COUNT(*) FROM samples GROUP BY our_split, mesh_family"
        ),
        "our_split_x_source": table(
            "SELECT our_split, doc_source, COUNT(*) FROM samples GROUP BY our_split, doc_source"
        ),
        "test_overlap_flags_x_split": table(
            "SELECT our_split, test_overlap, COUNT(*) FROM samples GROUP BY our_split, test_overlap"
        ),
    }
    write_json_atomic(layout.reports / "split-report.json", report)
    state.put(
        "split",
        {
            "at": utcnow(),
            "params": report["params"],
            "counts": dict(Counter(s for s, _ in out.values())),
        },
    )
    log("split", dict(Counter(s for s, _ in out.values())))
    return report
