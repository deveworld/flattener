"""Step 3: license tables from the asset share and per-sample classification.

Tables (asset share https://libdrive.ethz.ch/index.php/s/Gp5rdkRS5dXHExP, DOI
10.3929/ethz-c-000804058) are copied verbatim into ``index/license-tables``
with their ETag, size and SHA-256:

* ``documents/sources_{arxiv,doab,openstax,imslp}.json``: per paper/book/work
  license and attribution, with the page image names each one produced.
* ``backgrounds/backgrounds.zip``: the ``license``/``authors`` fields of all
  5,789 MatSynth material ``metadata.json`` members (range reads only).
* the README files of the dataset and asset shares (terms as published).

Classes (worst component wins, precedence nc > unknown > sa > clean):

==============  =========================================================  =======
component       rule                                                       class
==============  =========================================================  =======
mesh            SyntheticDoc meshes, CC BY 4.0                             clean
background      MatSynth ``license``: CC0, CC-BY -> clean;                 per item
                CC BY-NC-ND 2.0 (Deschaintre 2020) -> nc; missing -> unknown
arxiv           per paper: cc-by-4.0 / cc0 -> clean                        per item
openstax        per book: CC BY 4.0 -> clean; BY-NC-SA -> nc               per item
doab            per book: BY -> clean; BY-SA -> sa; BY-NC(-SA) -> nc       per item
imslp           CC BY-SA 4.0 (whole source)                                sa
RealKIE         CC BY-NC 4.0 (whole source)                                nc
inv3d           CC BY-NC-SA 4.0 (whole source)                             nc
CoSyn-400K      ODC-BY 1.0                                                 clean
res_txt2img     FLUX.2 [klein] outputs released CC BY 4.0 (memo 3.2)       clean
anything else   no license information                                      unknown
==============  =========================================================  =======

Track: ``release`` = clean only; ``research`` = every class.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sdcommon import (
    ASSET_DOI,
    ASSET_TOKEN,
    DATASET_DOI,
    DATASET_TOKEN,
    ConsistencyError,
    Http,
    Layout,
    RemoteFile,
    State,
    extract_member,
    get_file,
    get_ranges,
    head,
    log,
    member_spans,
    propfind,
    read_central_directory,
    sha256_file,
    utcnow,
    write_bytes_atomic,
    write_json_atomic,
)

CLASSES = ("clean", "sa", "nc", "unknown")
_PRECEDENCE = {"clean": 0, "sa": 1, "unknown": 2, "nc": 3}
SOURCE_FILES = ("arxiv", "doab", "openstax", "imslp")
ASSET_TEXT_FILES = (
    "README.md",
    "doi.md",
    "documents/README.md",
    "backgrounds/README.md",
    "meshes/README.md",
) + tuple(f"documents/sources_{s}.json" for s in SOURCE_FILES)
DATASET_TEXT_FILES = ("README.md", "DOI.md", "read_sample.py")

SYNTHETICDOC_CITATION = (
    "Daniel Woortmann, Tanguy Magne, and Olga Sorkine-Hornung. SyntheticDoc: A Large Synthetic "
    "Dataset for Document Unwarping and Illumination Correction. ECCV 2026. Dataset "
    f"https://doi.org/{DATASET_DOI} (CC BY 4.0); assets https://doi.org/{ASSET_DOI} (CC BY 4.0)."
)
WHOLE_SOURCE = {
    "imslp": ("CC-BY-SA-4.0", "https://creativecommons.org/licenses/by-sa/4.0/", "sa"),
    "RealKIE": ("CC-BY-NC-4.0", "https://creativecommons.org/licenses/by-nc/4.0/", "nc"),
    "inv3d": ("CC-BY-NC-SA-4.0", "https://creativecommons.org/licenses/by-nc-sa/4.0/", "nc"),
    "CoSyn-400K": ("ODC-BY-1.0", "https://opendatacommons.org/licenses/by/1-0/", "clean"),
    "res_txt2img": ("CC-BY-4.0", "https://creativecommons.org/licenses/by/4.0/", "clean"),
}
SOURCE_ATTRIBUTION = {
    "CoSyn-400K": (
        "Yue Yang, Ajay Patel, Matt Deitke, Tanmay Gupta, Luca Weihs, Andrew Head, Mark Yatskar, "
        "Chris Callison-Burch, Ranjay Krishna, Aniruddha Kembhavi, and Christopher Clark. Scaling "
        "Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation. ACL "
        "2025. CoSyn-400K https://huggingface.co/datasets/allenai/CoSyn-400K (ODC-BY 1.0)."
    ),
    "res_txt2img": (
        "SyntheticDoc text-to-image magazine pages generated with FLUX.2 [klein] 9B and released "
        "by the SyntheticDoc authors under CC BY 4.0 (FLUX Non-Commercial License s.2(d): outputs "
        "usable for any purpose except training a competing FLUX model)."
    ),
    "RealKIE": (
        "Benjamin Townsend, Madison May, Katherine Mackowiak, and Christopher Wells. RealKIE: "
        "Five Novel Datasets for Enterprise Key Information Extraction. arXiv:2403.20101, 2024 "
        "(CC BY-NC 4.0)."
    ),
    "inv3d": (
        "Felix Hertlein, Alexander Naumann, and Patrick Philipp. Inv3D: A High-Resolution 3D "
        "Invoice Dataset for Template-Guided Single-Image Document Unwarping. IJDAR 26(3), 2023 "
        "(CC BY-NC-SA 4.0)."
    ),
    "imslp": (
        "International Music Score Library Project (IMSLP), https://imslp.org/ (CC BY-SA 4.0)."
    ),
}
MESH_ATTRIBUTION = {
    "license": "CC-BY-4.0",
    "citation": SYNTHETICDOC_CITATION,
    "note": "Meshes simulated with ARCSim (Narain et al. 2012, 2013); released meshes CC BY 4.0.",
}
MATSYNTH_CITATION = (
    "Giuseppe Vecchio and Valentin Deschaintre. MatSynth: A Modern PBR Materials Dataset. "
    "CVPR 2024."
)


def combine(*classes: str) -> str:
    return max(classes, key=lambda c: _PRECEDENCE[c])


def cc_license(url: str, label: str = "") -> tuple[str, str]:
    """Normalize a Creative Commons URL/label to (SPDX-like id, class)."""
    text = f"{url} {label}".lower()
    match = re.search(r"licenses/(by(?:-nc)?(?:-sa|-nd)?(?:-nc)?)[/-](\d\.\d)", text)
    if match:
        code, version = match[1], match[2]
    else:
        words = re.sub(r"[^a-z0-9.]+", " ", label.lower()).split()
        if not words or words[0] != "cc":
            return ("unknown", "unknown")
        parts = [w for w in words[1:] if w in ("by", "nc", "sa", "nd")]
        code = "-".join(parts)
        version = next((w for w in words if re.fullmatch(r"\d\.\d", w)), "4.0")
    code_parts = set(code.split("-"))
    spdx = "CC-" + "-".join(p.upper() for p in code.split("-")) + f"-{version}"
    if "nc" in code_parts:
        return spdx, "nc"
    if "nd" in code_parts:
        return spdx, "unknown"  # ND pages are excluded upstream; flag if one appears
    if "sa" in code_parts:
        return spdx, "sa"
    if code == "by":
        return spdx, "clean"
    return spdx, "unknown"


def background_class(license_text: str | None) -> tuple[str, str]:
    text = (license_text or "").strip().upper().replace("_", " ")
    if text in ("CC0", "CC0 1.0", "CC-0"):
        return "CC0-1.0", "clean"
    if text in ("CC-BY", "CC BY", "CC BY 4.0", "CC-BY-4.0"):
        return "CC-BY", "clean"
    if "NC" in text:
        return text.replace(" ", "-"), "nc"
    return (text or "missing"), "unknown"


# --------------------------------------------------------------------------- fetch tables


def fetch_tables(client: Http, state: State, layout: Layout, *, connections: int = 1) -> dict:
    """Download the license tables (idempotent: files whose ETag is unchanged are kept)."""
    tables = layout.tables
    tables.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"fetched_at": utcnow(), "files": {}}
    listing = {}
    for token, directory in (
        (ASSET_TOKEN, ""),
        (ASSET_TOKEN, "documents"),
        (ASSET_TOKEN, "backgrounds"),
        (ASSET_TOKEN, "meshes"),
        (DATASET_TOKEN, ""),
    ):
        for row in propfind(client, token, directory):
            listing[(token, row.path)] = row
    previous = state.get("license_tables", {}).get("files", {})
    for token, share, names in (
        (ASSET_TOKEN, "assets", ASSET_TEXT_FILES),
        (DATASET_TOKEN, "dataset", DATASET_TEXT_FILES),
    ):
        for name in names:
            row = listing.get((token, name))
            if row is None:
                raise ConsistencyError(f"{share}:{name} missing from share listing")
            local = tables / share / name
            key = f"{share}/{name}"
            old = previous.get(key)
            if not (
                old
                and old["etag"] == row.etag
                and local.is_file()
                and sha256_file(local) == old["sha256"]
            ):
                resp = get_file(client, RemoteFile(token, name), expect_size=row.size)
                if resp.headers.get("etag") not in (None, row.etag):
                    raise ConsistencyError(f"{key}: ETag changed while listing")
                write_bytes_atomic(local, resp.body)
            record["files"][key] = {
                "etag": row.etag,
                "size": row.size,
                "last_modified": row.last_modified,
                "sha256": sha256_file(local),
                "local": str(local.relative_to(layout.research)),
            }
    # backgrounds: license fields of every material
    remote = RemoteFile(ASSET_TOKEN, "backgrounds/backgrounds.zip")
    info = head(client, remote)
    old = previous.get("backgrounds/backgrounds.zip")
    bg_json = tables / "assets" / "backgrounds" / "materials-metadata.json"
    if not (old and old["etag"] == info.etag and bg_json.is_file()):
        entries, cd_offset, _ = read_central_directory(
            client, remote, info.size, info.etag, tail_bytes=8 << 20
        )
        spans = member_spans(entries, cd_offset)
        metas = [e for e in entries if e.name.endswith("/metadata.json")]
        log(f"backgrounds.zip: {len(entries)} members, {len(metas)} material metadata files")
        got = get_ranges(client, remote, [spans[e.name] for e in metas], total=info.size)
        materials = {}
        for entry in metas:
            name = entry.name.rstrip("/").rsplit("/", 2)[-2]
            materials[name] = json.loads(extract_member(got[spans[entry.name]], 0, entry))
        if head(client, remote).etag != info.etag:
            raise ConsistencyError("backgrounds.zip changed while reading")
        write_json_atomic(bg_json, materials, indent=None)
    record["files"]["backgrounds/backgrounds.zip"] = {
        "etag": info.etag,
        "size": info.size,
        "last_modified": info.last_modified,
        "oc_checksum": info.checksum,
        "extracted": str(bg_json.relative_to(layout.research)),
        "sha256": sha256_file(bg_json),
    }
    load_tables(state, layout)
    state.put("license_tables", record)
    return record


# --------------------------------------------------------------------------- registry


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _stem(name: str) -> str:
    return name[:-4] if name.lower().endswith(".png") else name


class Registry:
    """Name -> document reference lookups built from the sources_*.json tables."""

    def __init__(self, layout: Layout) -> None:
        base = layout.tables / "assets" / "documents"
        self.docs: dict[str, dict[str, Any]] = {}
        self.by_name: dict[str, dict[str, str]] = {s: {} for s in SOURCE_FILES}
        self.fallback: dict[str, dict[str, str]] = {s: {} for s in SOURCE_FILES}
        load = lambda s: json.loads((base / f"sources_{s}.json").read_text(encoding="utf-8"))  # noqa: E731
        for paper in load("arxiv"):
            ref = f"arxiv:{paper['paper_id']}"
            spdx, cls = {
                "cc-by-4.0": ("CC-BY-4.0", "clean"),
                "cc0": ("CC0-1.0", "clean"),
            }.get(str(paper.get("license", "")).lower(), (str(paper.get("license")), "unknown"))
            self.docs[ref] = {
                "source": "arxiv",
                "license_id": spdx,
                "license_url": {
                    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
                    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
                }.get(spdx, ""),
                "license_class": cls,
                "title": paper.get("title"),
                "authors": paper.get("authors"),
                "url": paper.get("arxiv_url"),
                "citation": paper.get("citation_text"),
            }
            self.fallback["arxiv"][paper["paper_id"]] = ref
            for image in paper["source_of_images"]:
                self.by_name["arxiv"][_nfc(_stem(image))] = ref
        for book in load("openstax"):
            ref = f"openstax:{book['id']}"
            lic = book.get("license") or {}
            spdx, cls = cc_license(str(lic.get("url", "")), str(lic.get("name", "")))
            if cls == "unknown" and "attribution license" in str(lic.get("name", "")).lower():
                spdx, cls = "CC-BY-4.0", "clean"
            self.docs[ref] = {
                "source": "openstax",
                "license_id": spdx,
                "license_url": lic.get("url"),
                "license_class": cls,
                "title": book.get("title"),
                "authors": book.get("authors"),
                "url": book.get("url"),
                "citation": (
                    f"{', '.join(book.get('authors') or [])}. {book.get('title')}. OpenStax, "
                    f"{book.get('url')} ({spdx}). Access for free at openstax.org."
                ),
            }
            for image in book["source_of_images"]:
                stem = _nfc(_stem(image))
                self.by_name["openstax"][stem] = ref
                self.fallback["openstax"][re.sub(r"_\d+$", "", stem)] = ref
        for book in load("doab"):
            handle = str(book.get("doab_url", "")).rstrip("/").rsplit("/", 1)[-1]
            ref = f"doab:{handle}"
            spdx, cls = cc_license(str(book.get("license", "")), str(book.get("license_type", "")))
            if cls == "unknown" and str(book.get("license_type", "")).upper() in (
                "CC BY",
                "CC BY 4.0",
            ):
                spdx, cls = "CC-BY-4.0", "clean"
            authors = book.get("authors") or []
            self.docs[ref] = {
                "source": "doab",
                "license_id": spdx,
                "license_url": book.get("license"),
                "license_class": cls,
                "title": book.get("title"),
                "authors": authors,
                "url": book.get("doab_url"),
                "citation": (
                    f"{'; '.join(authors)}. {book.get('title')}. {book.get('publisher')}. "
                    f"{book.get('doab_url')} ({spdx})."
                ),
                "extra": {"language": book.get("language"), "pdf_url": book.get("pdf_url")},
            }
            self.fallback["doab"][handle] = ref
            for image in book["source_of_images"]:
                self.by_name["doab"][_nfc(_stem(image))] = ref
        for k, work in enumerate(load("imslp")):
            ref = f"imslp:{k:04d}"
            self.docs[ref] = {
                "source": "imslp",
                "license_id": "CC-BY-SA-4.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "license_class": "sa",
                "title": work.get("work_title"),
                "authors": None,
                "url": work.get("imslp_url"),
                "citation": f"{work.get('work_title')}. IMSLP, {work.get('imslp_url')} "
                "(CC BY-SA 4.0; see the work page for additional terms).",
            }
            for image in work["source_of_images"]:
                stem = _nfc(_stem(image))
                self.by_name["imslp"][stem] = ref
                self.fallback["imslp"][stem.split("_")[1] if "_" in stem else stem] = ref
        for source, (spdx, url, cls) in WHOLE_SOURCE.items():
            if source == "imslp":
                continue
            self.docs[f"source:{source}"] = {
                "source": source,
                "license_id": spdx,
                "license_url": url,
                "license_class": cls,
                "title": None,
                "authors": None,
                "url": None,
                "citation": SOURCE_ATTRIBUTION[source],
            }

    def resolve(self, source: str, name: str) -> tuple[str | None, str, str, str]:
        """Return (doc_ref, license_id, class, doc_group) for one document path."""
        stem = _nfc(_stem(name or ""))
        ref = None
        if source in self.by_name:
            ref = self.by_name[source].get(stem)
            if ref is None:
                fb = self.fallback[source]
                if source == "arxiv":
                    match = re.match(r"^page_\d+_(.+)_p\d+$", stem)
                    ref = fb.get(match[1]) if match else None
                elif source == "doab":
                    match = re.search(r"_(\d+)_\d+$", stem)
                    ref = fb.get(match[1]) if match else None
                elif source == "openstax":
                    ref = fb.get(re.sub(r"_\d+$", "", stem))
                elif source == "imslp":
                    ref = fb.get(stem.split("_")[1]) if "_" in stem else None
            if ref is not None:
                doc = self.docs[ref]
                return ref, doc["license_id"], doc["license_class"], ref
            if source == "imslp":
                return None, "CC-BY-SA-4.0", "sa", f"imslp:?{stem}"
            return None, "unmatched", "unknown", f"{source}:?{stem}"
        if source in WHOLE_SOURCE:
            spdx, _, cls = WHOLE_SOURCE[source]
            if source == "RealKIE":
                group = "realkie:" + re.sub(r"_page_\d+$", "", stem)
            else:
                group = f"{source.lower()}:{stem}"
            return f"source:{source}", spdx, cls, group
        return None, "unknown-source", "unknown", f"{source or '?'}:{stem}"


def load_backgrounds(layout: Layout) -> dict[str, dict[str, Any]]:
    path = layout.tables / "assets" / "backgrounds" / "materials-metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_tables(state: State, layout: Layout) -> None:
    """Mirror the attribution registry and background licenses into the index DB."""
    reg = Registry(layout)
    rows = [
        (
            ref,
            d["source"],
            d["license_id"],
            d.get("license_url"),
            d["license_class"],
            d.get("title"),
            json.dumps(d.get("authors"), ensure_ascii=False),
            d.get("url"),
            d.get("citation"),
            json.dumps(d.get("extra"), ensure_ascii=False),
        )
        for ref, d in reg.docs.items()
    ]
    materials = load_backgrounds(layout)
    bg_rows = []
    for name, meta in materials.items():
        spdx, cls = background_class(meta.get("license"))
        bg_rows.append(
            (
                name,
                spdx,
                cls,
                json.dumps(meta.get("authors"), ensure_ascii=False),
                meta.get("source"),
                json.dumps({k: meta.get(k) for k in ("name", "link", "category", "license")}),
            )
        )
    with state.lock:
        state.db.execute("DELETE FROM docs")
        state.db.executemany("INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        state.db.execute("DELETE FROM backgrounds")
        state.db.executemany("INSERT INTO backgrounds VALUES (?,?,?,?,?,?)", bg_rows)
        state.db.commit()
    log(f"registry: {len(rows)} document references, {len(bg_rows)} background materials")


# --------------------------------------------------------------------------- classify


def classify(state: State, layout: Layout, *, batch: int = 50000) -> dict[str, Any]:
    reg = Registry(layout)
    backgrounds = {
        name: (spdx, cls)
        for name, spdx, cls in state.query("SELECT name, license, license_class FROM backgrounds")
    }
    if not backgrounds:
        raise RuntimeError("background license table is empty; run the licenses step first")
    rows = state.query(
        "SELECT uid, doc_source, doc_name, bg_name, mesh_family, mesh_id, status FROM samples"
    )
    updates = []
    counts: Counter = Counter()
    for uid, source, name, bg, family, mesh_id, status in rows:
        if status != "success":
            ref, doc_spdx, doc_cls, group = None, "n/a", "unknown", None
            bg_spdx, bg_cls = "n/a", "unknown"
            cls = "unknown"
        else:
            ref, doc_spdx, doc_cls, group = reg.resolve(source or "", name or "")
            bg_spdx, bg_cls = backgrounds.get(bg or "", ("missing", "unknown"))
            cls = combine(doc_cls, bg_cls, "clean")  # mesh component is always clean
        mesh_group = f"{family}/{mesh_id}" if family and mesh_id else None
        updates.append((doc_cls, bg_cls, cls, ref, doc_spdx, bg_spdx, group, mesh_group, uid))
        counts[cls] += 1
        if len(updates) >= batch:
            _flush(state, updates)
            updates = []
    _flush(state, updates)
    state.put("classified", {"at": utcnow(), "counts": dict(counts)})
    log("classified", dict(counts))
    return dict(counts)


def _flush(state: State, updates: list[tuple]) -> None:
    if updates:
        state.executemany(
            "UPDATE samples SET doc_class=?, bg_class=?, lic_class=?, doc_ref=?, doc_license=?, "
            "bg_license=?, doc_group=?, mesh_group=? WHERE uid=?",
            updates,
        )


def license_report(state: State, layout: Layout) -> dict[str, Any]:
    def table(sql: str) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = defaultdict(dict)
        for key, cls, n in state.query(sql):
            out[str(key)][cls] = n
        return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}

    total = dict(state.query("SELECT lic_class, COUNT(*) FROM samples GROUP BY lic_class"))
    success = state.query("SELECT COUNT(*) FROM samples WHERE status='success'")[0][0]
    by_split = table("SELECT split, lic_class, COUNT(*) FROM samples GROUP BY split, lic_class")
    by_source = table(
        "SELECT doc_source, lic_class, COUNT(*) FROM samples GROUP BY doc_source, lic_class"
    )
    by_doc_license = table(
        "SELECT doc_source || ' ' || doc_license, doc_class, COUNT(*) FROM samples "
        "GROUP BY doc_source, doc_license, doc_class"
    )
    by_bg = table(
        "SELECT bg_license, bg_class, COUNT(*) FROM samples GROUP BY bg_license, bg_class"
    )
    by_family = table(
        "SELECT mesh_family, lic_class, COUNT(*) FROM samples GROUP BY mesh_family, lic_class"
    )
    doc_x_bg = table(
        "SELECT doc_class || ' doc', bg_class, COUNT(*) FROM samples GROUP BY doc_class, bg_class"
    )
    unknown = state.query(
        "SELECT uid, doc_source, doc_name, bg_name, doc_license, bg_license FROM samples "
        "WHERE lic_class='unknown' LIMIT 50"
    )
    clean_pool = dict(
        state.query("SELECT split, COUNT(*) FROM samples WHERE lic_class='clean' GROUP BY split")
    )
    report = {
        "schema": "flattener-syntheticdoc-license-report-1",
        "track": "research",
        "note": "Covers every class (research track). The release track uses class 'clean' only.",
        "generated_at": utcnow(),
        "dataset": {"doi": DATASET_DOI, "license": "CC-BY-4.0", "assets_doi": ASSET_DOI},
        "rules": {
            "precedence": "nc > unknown > sa > clean (worst component wins)",
            "components": "document (per paper/book/work or whole source), background material "
            "(MatSynth license field), mesh (CC BY 4.0, always clean)",
            "release": "clean only (owner's decision: ShareAlike samples are left out for now)",
            "research": "all classes; weights never released",
        },
        "tables": state.get("license_tables"),
        "samples_total": sum(total.values()),
        "samples_success": success,
        "class_counts": total,
        "clean_fraction_of_success": round(total.get("clean", 0) / max(success, 1), 4),
        "clean_by_split": clean_pool,
        "by_split": by_split,
        "by_document_source": by_source,
        "by_document_license": by_doc_license,
        "by_background_license": by_bg,
        "by_mesh_family": by_family,
        "document_class_x_background_class": doc_x_bg,
        "unknown_examples": [
            dict(
                zip(("uid", "doc_source", "doc_name", "bg_name", "doc_license", "bg_license"), row)
            )
            for row in unknown
        ],
    }
    write_json_atomic(layout.reports / "license-report.json", report)
    return report


def attribution_for(state: State, doc_ref: str | None, bg_name: str | None) -> dict[str, Any]:
    """Per-sample attribution block (document, background, mesh, dataset)."""
    doc = None
    if doc_ref:
        rows = state.query(
            "SELECT source, license_id, license_url, title, authors, url, citation FROM docs "
            "WHERE doc_ref=?",
            (doc_ref,),
        )
        if rows:
            s, spdx, url, title, authors, link, citation = rows[0]
            doc = {
                "ref": doc_ref,
                "source": s,
                "license": spdx,
                "license_url": url,
                "title": title,
                "authors": json.loads(authors) if authors else None,
                "url": link,
                "citation": citation,
            }
    bg = None
    if bg_name:
        rows = state.query(
            "SELECT license, authors, source, meta FROM backgrounds WHERE name=?", (bg_name,)
        )
        if rows:
            spdx, authors, source, meta = rows[0]
            meta = json.loads(meta)
            bg = {
                "material": bg_name,
                "license": spdx,
                "authors": json.loads(authors),
                "source": source,
                "link": meta.get("link"),
                "matsynth_name": meta.get("name"),
                "citation": MATSYNTH_CITATION,
            }
    return {
        "dataset": {
            "name": "SyntheticDoc",
            "doi": DATASET_DOI,
            "license": "CC-BY-4.0",
            "citation": SYNTHETICDOC_CITATION,
        },
        "document": doc,
        "background": bg,
        "mesh": MESH_ATTRIBUTION,
    }


def tables_present(layout: Layout) -> bool:
    base: Path = layout.tables
    return (
        all((base / "assets" / "documents" / f"sources_{s}.json").is_file() for s in SOURCE_FILES)
        and (base / "assets" / "backgrounds" / "materials-metadata.json").is_file()
    )
