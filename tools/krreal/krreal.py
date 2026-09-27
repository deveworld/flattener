"""KR-Real v1 pilot kit: shot list, ingest, draft labels, transcripts, checks and manifest.

The photos are private evaluation data. They stay under a git-ignored data root
(default ``outputs/krreal/v1``); every command that writes refuses a root that git
would track. Nothing here collects or uploads photos.

  shotlist     write the 60-shot plan (shotlist.csv) to fill in while copying photos
  ingest       copy the phone files named in shotlist.csv into photos/ under kit names
  draft        write drafts/<stem>.json for labeler.html (provider hook: --provider)
  transcripts  write transcript templates for the photos picked per stratum (--ocr hook)
  check        check labels/ (and drafts/ with --drafts) and transcripts/
  manifest     write manifest.json: SHA-256 per photo, split, label and transcript state
  sync-labeler copy kit.json into labeler.html (after editing kit.json)

Exit status: 0 done, 1 problems found (listed on stderr), 2 usage error.
See tools/krreal/README.md for the whole workflow and shooting-guide.md for shooting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import drafts  # noqa: E402
import kitlib  # noqa: E402
from labelcheck import (  # noqa: E402
    check_label,
    check_transcript,
    errors,
    load_json,
    load_kit,
    load_schema,
    parse_transcript,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "outputs" / "krreal" / "v1"


def _fail(problems: list[str]) -> int:
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


def _kit_photos(root: Path, kit: dict, only: list[str] | None) -> tuple[list, list[str]]:
    names, problems = [], []
    folder = root / "photos"
    files = sorted(p.name for p in folder.glob("*") if p.is_file()) if folder.is_dir() else []
    wanted = set(only) if only else None
    for file in files:
        if file.endswith(".part") or (wanted is not None and file not in wanted):
            continue
        try:
            names.append(kitlib.parse_name(file, kit))
        except ValueError as error:
            problems.append(str(error))
    if wanted is not None:
        problems += [f"{file}: not in {folder}" for file in sorted(wanted - set(files))]
    return names, problems


def cmd_shotlist(args: argparse.Namespace) -> int:
    rows = kitlib.shotlist_rows(load_kit())
    if args.print:
        for row in rows:
            print(f"{row['phone']} {int(row['shot']):>3} {row['stratum']:<8} {row['description']}")
        return 0
    kitlib.ensure_ignored(args.root)
    path = args.root / "shotlist.csv"
    if path.exists() and not args.overwrite:
        return _fail([f"{path} exists; --overwrite replaces it (and the file names in it)"])
    kitlib.write_shotlist(path, rows)
    print(f"wrote {path} ({len(rows)} shots)")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    kitlib.ensure_ignored(args.root)
    shotlist = args.shotlist or args.root / "shotlist.csv"
    try:
        rows = kitlib.read_shotlist(shotlist)
    except (OSError, ValueError) as error:
        return _fail([str(error)])
    done, problems = kitlib.ingest(args.root, rows, kit=load_kit(), dry_run=args.dry_run)
    for line in done:
        print(("would copy " if args.dry_run else "copied ") + line)
    print(f"{len(done)} photos {'to copy' if args.dry_run else 'copied'}, {len(problems)} problems")
    return _fail(problems)


def cmd_draft(args: argparse.Namespace) -> int:
    kitlib.ensure_ignored(args.root)
    kit = load_kit()
    try:
        provider = drafts.label_provider(args.provider)
    except (ImportError, ValueError, OSError) as error:
        return _fail([str(error)])
    names, problems = _kit_photos(args.root, kit, args.photos)
    written = kept = 0
    for name in names:
        target = args.root / "drafts" / f"{name.stem}.json"
        if target.exists() and not args.overwrite:
            kept += 1
            continue
        try:
            info = kitlib.photo_info(args.root / "photos" / name.file)
        except ValueError as error:
            problems.append(str(error))
            continue
        try:
            proposal = provider(args.root / "photos" / name.file, info | {"name": name})
        except Exception as error:  # a hook failure keeps the draft without geometry
            problems.append(f"{name.file}: provider failed ({error!r})")
            proposal = None
        label, dropped = drafts.build_draft(
            info, name, provider_name=args.provider, proposal=proposal, kit=kit
        )
        problems += dropped
        kitlib.write_json(target, label)
        written += 1
    print(f"drafts: {written} written, {kept} kept (use --overwrite to replace)")
    return _fail(problems)


def cmd_transcripts(args: argparse.Namespace) -> int:
    kitlib.ensure_ignored(args.root)
    kit = load_kit()
    try:
        provider = drafts.text_provider(args.ocr)
    except (ImportError, ValueError, OSError) as error:
        return _fail([str(error)])
    names, problems = _kit_photos(args.root, kit, args.photos)
    if not args.photos:
        names = drafts.pick_for_transcripts(names, kit)
    written = 0
    for name in names:
        target = args.root / "transcripts" / f"{name.stem}.txt"
        if target.exists():
            try:
                status = parse_transcript(target.read_text(encoding="utf-8")).header.get("status")
            except (OSError, ValueError):
                status = None
            if not args.overwrite or status != "draft":
                print(f"kept {target.name} (status {status})")
                continue
        photo = args.root / "photos" / name.file
        try:
            info = kitlib.photo_info(photo)
        except ValueError as error:
            problems.append(str(error))
            continue
        try:
            lines = provider(photo, info | {"name": name})
        except Exception as error:  # an OCR failure leaves an empty template
            problems.append(f"{name.file}: text provider failed ({error!r})")
            lines = []
        text = drafts.render_transcript(
            photo=name.file, sha256=info["sha256"], page=1, draft=args.ocr, lines=lines
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        written += 1
    print(f"transcripts: {written} written for {len(names)} photos")
    return _fail(problems)


def cmd_check(args: argparse.Namespace) -> int:
    kit, schema = load_kit(), load_schema()
    shas: dict[str, str] = {}

    def photo_sha(file: str) -> str | None:
        path = args.root / "photos" / file
        if file not in shas and path.is_file():
            shas[file] = kitlib.sha256_file(path)
        return shas.get(file)

    problems: list[str] = []
    warnings: list[str] = []
    folders = ["labels", "drafts"] if args.drafts else ["labels"]
    checked = 0
    for folder in folders:
        for path in sorted((args.root / folder).glob("*.json")):
            checked += 1
            file = path.stem + ".jpg"
            try:
                label = load_json(path)
            except (OSError, ValueError) as error:
                problems.append(f"{folder}/{path.name}: {error}")
                continue
            issues = check_label(label, kit=kit, schema=schema, file_name=file)
            sha = photo_sha(file)
            if sha is None:
                warnings.append(f"{folder}/{path.name}: no photo {file}")
            elif not errors(issues) and label["photo"]["sha256"] != sha:
                problems.append(f"{folder}/{path.name}: made for other bytes than {file}")
            for issue in issues:
                line = f"{folder}/{path.name}: {issue.where}: {issue.message}"
                (problems if issue.level == "error" else warnings).append(line)
    for path in sorted((args.root / "transcripts").glob("*.txt")):
        checked += 1
        file = path.stem + ".jpg"
        try:
            transcript = parse_transcript(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as error:
            problems.append(f"transcripts/{path.name}: {error}")
            continue
        issues = check_transcript(
            transcript,
            photo_file=file,
            photo_sha256=photo_sha(file),
            markers=kit["transcript_markers"],
        )
        for issue in issues:
            line = f"transcripts/{path.name}: {issue.where}: {issue.message}"
            (problems if issue.level == "error" else warnings).append(line)
    for line in warnings:
        print(f"warning: {line}")
    print(f"checked {checked} files: {len(problems)} errors, {len(warnings)} warnings")
    return _fail(problems)


def cmd_manifest(args: argparse.Namespace) -> int:
    kitlib.ensure_ignored(args.root)
    manifest = kitlib.build_manifest(args.root)
    out = args.out or args.root / "manifest.json"
    kitlib.write_json(out, manifest)
    summary = manifest["summary"]
    print(f"{manifest['set']} manifest: {out}")
    print(f"photos {summary['photos']}/{summary['planned']}, by phone {summary['by_phone']}")
    print(f"devices {summary['devices']}")
    for code, count in summary["by_stratum"].items():
        print(f"  {code:<8} {count['have']:>3}/{count['planned']}")
    print(f"split {summary['by_split']}")
    print(f"labels {summary['labels']}, transcripts {summary['transcripts']}")
    print(f"M1 complete: {'yes' if summary['m1_complete'] else 'no'}")
    for item in summary["m1_missing"]:
        print(f"  - {item}")
    for warning in manifest["warnings"]:
        print(f"warning: {warning}")
    if args.strict:
        return _fail(manifest["problems"])
    for problem in manifest["problems"]:
        print(f"problem: {problem}", file=sys.stderr)
    return 0


def cmd_sync_labeler(args: argparse.Namespace) -> int:
    html = kitlib.LABELER_PATH.read_text(encoding="utf-8")
    kit_text = (Path(__file__).resolve().parent / "kit.json").read_text(encoding="utf-8")
    updated = kitlib.embed_kit(html, kit_text)
    if updated == html:
        print("labeler.html already carries kit.json")
        return 0
    kitlib.LABELER_PATH.write_text(updated, encoding="utf-8")
    print("labeler.html now carries kit.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, handler, help_text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text, description=help_text)
        sub.add_argument(
            "--root",
            type=Path,
            default=DEFAULT_ROOT,
            help="data root (git-ignored); default %(default)s",
        )
        sub.set_defaults(handler=handler)
        return sub

    sub = command("shotlist", cmd_shotlist, "write <root>/shotlist.csv, the 60-shot plan")
    sub.add_argument("--overwrite", action="store_true", help="replace an existing shotlist.csv")
    sub.add_argument("--print", action="store_true", help="print the plan to the terminal")

    sub = command("ingest", cmd_ingest, "copy phone files listed in shotlist.csv into photos/")
    sub.add_argument("--shotlist", type=Path, help="default <root>/shotlist.csv")
    sub.add_argument("--dry-run", action="store_true", help="list the copies without copying")

    sub = command("draft", cmd_draft, "write drafts/<stem>.json for the labeler")
    sub.add_argument(
        "--provider",
        default="none",
        help="none, or a hook module:function / file.py:function that proposes geometry",
    )
    sub.add_argument("--photos", nargs="+", metavar="FILE", help="only these photo files")
    sub.add_argument("--overwrite", action="store_true", help="replace existing drafts")

    sub = command("transcripts", cmd_transcripts, "write transcript templates")
    sub.add_argument(
        "--photos",
        nargs="+",
        metavar="FILE",
        help="only these photo files (default: the pick per stratum)",
    )
    sub.add_argument(
        "--ocr",
        default="none",
        help="text draft: none, tesseract (kor+eng), or a hook module:function",
    )
    sub.add_argument(
        "--overwrite", action="store_true", help="replace templates still in status draft"
    )

    sub = command("check", cmd_check, "check labels and transcripts")
    sub.add_argument("--drafts", action="store_true", help="check drafts/ as well")

    sub = commands.add_parser(
        "sync-labeler",
        help="copy kit.json into labeler.html after editing kit.json",
        description="copy kit.json into labeler.html after editing kit.json",
    )
    sub.set_defaults(handler=cmd_sync_labeler)

    sub = command("manifest", cmd_manifest, "write manifest.json with SHA-256 per photo")
    sub.add_argument("--out", type=Path, help="default <root>/manifest.json")
    sub.add_argument("--strict", action="store_true", help="exit 1 when problems are found")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
