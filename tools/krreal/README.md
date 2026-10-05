# KR-Real v1 pilot kit

Tools to shoot, label and transcribe the first 60 photos of KR-Real v1, a Korean real-photo evaluation set, and to build their list (manifest). The photos are private evaluation material.

- Photos, labels, transcripts and the manifest all live in `outputs/krreal/v1/` (a folder git ignores). The kit's commands refuse folders that git tracks.
- The photos are not used for training or checkpoint selection. The dev split (40% of sources) is for tuning thresholds, and the test split (60%) is reported once per milestone.
- Only photos of public-domain printed matter may be published. The manifest marks a photo `public_candidate` only if its source has `public` set to `yes` in the optional file `outputs/krreal/v1/sources.csv` (columns: `source_id,kind,public,note`) and the photo has no GPS data. All others are `private`.

## Files

| File | Contents |
|---|---|
| `kit.json` | Strata, shot list per phone, flags, kinds of text regions and reference lines, file naming rule. All other files use these values |
| `label.schema.json` | Label JSON Schema (`krreal-label-1`) |
| `labeler.html` | Labeler that runs in a browser. It is a single file and uses no external resources |
| `transcript-template.txt` | Transcript file template and transcription rules |
| `krreal.py` | Commands: `shotlist`, `ingest`, `draft`, `transcripts`, `check`, `manifest`, `sync-labeler` |
| `kitlib.py`, `labelcheck.py`, `drafts.py` | Modules used by the commands: photos and manifest, label and transcript checks, drafts and hooks |

After editing `kit.json`, run `krreal.py sync-labeler` to update the copy inside the labeler. `tests/test_krreal_kit.py` checks that the schema, the labeler and `kit.json` agree.

## Workflow

Run commands from the repository root as `.venv/bin/python tools/krreal/krreal.py <command>`. The default for `--root` is `outputs/krreal/v1`.

| Step | Command or tool | Result |
|---|---|---|
| 1. Shot list | `shotlist` | `shotlist.csv` (30 rows per phone) |
| 2. Shooting | Phone camera | 60 phone photos |
| 3. Transfer | `ingest` | `photos/kr1-….jpg` (original bytes unchanged) |
| 4. List | `manifest` | `manifest.json`, missing counts and warnings |
| 5. Drafts | `draft [--provider …]` | `drafts/<name>.json` |
| 6. Labels | `labeler.html` | `labels/<name>.json` |
| 7. Check | `check [--drafts]` | Error list, exit code |
| 8. Transcription | `transcripts [--ocr …]`, text editor | `transcripts/<name>.txt` |
| 9. Wrap-up | `manifest --strict` | `pilot complete: yes` |

Exit codes: 0 means done, 1 means there are problems (listed on standard error), 2 means a usage error.

**Pilot completion criteria** (the pilot is complete when `summary.m1_missing` in `manifest` is empty)

- At least the planned count in every stratum, and 60 photos or more in total
- Two phones with different EXIF models
- Every photo has a `labeled` or `checked` label with no check errors
- At least 20 transcribed pages confirmed by two people (`checked`)

## Labels

Each photo has one JSON label file, which follows `label.schema.json`. `check` enforces the rules that the schema cannot express.

**Coordinates.** Pixels of the photo with the EXIF orientation applied. (0, 0) is the center of the top-left pixel; x grows to the right and y grows downward. Normalized flattener coordinates are `g = 2 * (p + 0.5) / size - 1`. When the page extends past the photo, corners may be placed outside the photo (up to 50% of the photo size).

| Field | Rule |
|---|---|
| `quad` | The 4 page corners: TL, TR, BR, BL in the page's reading orientation. They must run clockwise on screen and form a convex shape |
| `page_rotation_deg`, `page_quarter` | Computed from the quad: the clockwise angle from the image x axis to the sum of the top edge (TL→TR) and bottom edge (BL→BR) directions, and the number of quarter turns (0–3) closest to that angle. Rotating the photo counterclockwise by `90 × page_quarter` degrees makes the page upright |
| `curved_edges` | Only for edges that curve. At least 7 points including the corners. Direction: top: TL→TR, bottom: BL→BR, left: TL→BL, right: TR→BR. The end points equal the quad corners |
| `text_polygons` | Text region polygons. They also cover headers and page numbers. Kinds `text`, `table`, `formula` |
| `reference_lines` | 3–5 lines that are straight on the flat page: `baseline` (text line baseline), `table_rule` (table rule), `vertical_line` (vertical line) |
| `page_count`, `pages` | 0 for no page, 2 for a two-page spread. A two-page spread has a quad for each page, and `reading_order` 1 is the left page |
| `flags` | `occluded`, `cut_off`, `neighbor_page`, `vertical_text`, `no_page`, `two_page`, `exif_wrong`, `glare`, `shadow`, `blur`. `no_page`, `two_page` and `cut_off` follow from the page count and `cut_off_sides` |
| `cut_off_sides` | Sides where the page extends past the photo (`top`, `right`, `bottom`, `left`, relative to the photo) |
| `photo`, `capture` | File name, SHA-256, size, EXIF orientation, device and capture settings. GPS is not copied |
| `review` | `draft` → `labeled` (author, time) → `checked` (another person, time). Time spent in `seconds_spent` |
| `draft` | The provider that made the draft and the fields it filled. Kept after a person edits the label |

`labeled` and `checked` labels must pass every rule. For this milestone a two-page spread (`spread`) may have quads only. A `draft` is checked only against the schema and the geometry of shapes already drawn.

## Labeler (`labeler.html`)

Open the file in a browser (Chrome, Edge, Firefox). Photos are read only inside the browser and are never sent anywhere.

1. Use **Open folder** to pick `outputs/krreal/v1`. It reads `photos/`, `drafts/`, `labels/` and `manifest.json` together. You can also pick photos and JSON files with **Open files**.
2. Enter your name in the **Name** field at the top.
3. For each photo, mark the 4 corners (Q), curved edges (E), text regions (T), 3–5 reference lines (L) and flags. When the **Check** panel on the right is empty, save (Ctrl+S). If errors remain, save it as a draft (`draft`).
4. Where labels are saved
   - Chrome, Edge: pick the `labels/` folder with **Connect labels folder**, and labels are written straight into it.
   - Other browsers: `<name>.json` appears in the downloads folder. Move it to `labels/`.
5. Another person reviews the label and then presses **Save as checked** under their own name.

- Unsaved work is kept in browser storage automatically and is restored when you open that photo again.
- When a photo opens, the labeler computes its SHA-256 and compares it with the values in the draft, label and manifest. It warns if the size with EXIF orientation applied differs from the record.
- The thick yellow line and the arrow mark the top of the page. If you placed the corners in a different order, fix it with **Rotate corners** (O) or **Swap left/right** (M).
- Shortcuts are listed under **Help** (?). The target is 1.5–2 minutes per photo.

## Drafts and hooks

`draft` writes `drafts/<name>.json` for each photo. The default provider `none` fills in only photo information (SHA-256, size, EXIF, device). It does not touch `labels/`.

To attach a function that prefills geometry, use `--provider module:function` or `--provider path/to/file.py:function`.

```python
def propose(photo: Path, info: dict) -> dict:
    # info: kitlib.photo_info(photo) plus name (PhotoName)
    # Corners are TL, TR, BR, BL. Everything except quad is optional.
    page = {
        "quad": [[x, y], ...],
        "curved_edges": {"top": [...]},
        "text_polygons": [...],
        "reference_lines": [...],
    }
    return {"pages": [page], "flags": [...]}
```

The coordinate convention is the same as for labels. Curve end points are replaced by the quad corners. A proposal that fails the checks is discarded, and only the photo information is kept. The scanner provider is `--provider tools/krreal/scan_provider.py:propose`. It turns the scanner's final grid into a quad and curved edges with `drafts.pages_from_grid`. If a scan record for the same photo exists in `scan-quarter/` or `scan-default/`, it uses that grid; otherwise it runs the scanner. Photos for which the scanner returned the original are left without geometry. It does not fill in text regions or reference lines yet.

## Transcripts

```bash
.venv/bin/python tools/krreal/krreal.py transcripts                 # 24 photos picked by stratum
.venv/bin/python tools/krreal/krreal.py transcripts --ocr tesseract # Tesseract kor+eng drafts
.venv/bin/python tools/krreal/krreal.py transcripts --photos kr1-p1-curved-b03-007.jpg
```

- For each stratum, it picks the number of photos set by `transcribe` in `kit.json`, in a fixed order (curved 8, small print 4, tables and formulas 4, rotation 3, flat 3, background 2).
- A file has a header (`# key: value`), a `---` line and the body, in that order. The rules are in the template: write each text line of the photo on its own line, in reading order, exactly as printed; separate table cells with tabs; write formulas as `[[formula]]`, figures as `[[figure]]` and unreadable characters as `[[?]]`.
- `draft` → `transcribed` (first transcriber) → `checked` (a different checker). Compare against the photo zoomed to 100%.
- `--ocr tesseract` needs Tesseract's `kor` language data. `--ocr module:function` takes a `(photo, info) -> list[str]` function.
- `--overwrite` rewrites only files that are still `draft`.

## manifest

`manifest.json` (`krreal-manifest-1`) records for each photo the file name, SHA-256, byte count, phone, stratum, shot-list variant, source, split, public status, size, EXIF orientation, device, capture time, and the SHA-256 and status of its label and transcript. Splits are made per source, in a fixed order, so that dev gets close to 40% of the photos. Once a source has a split, it keeps it when the manifest is rebuilt. `problems` lists naming rule violations, photos with identical bytes, duplicate numbers, and label and transcript errors. `warnings` lists missing EXIF, GPS present, non-JPEG files, and any single source above 15%.
