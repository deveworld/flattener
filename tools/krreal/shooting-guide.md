# KR-Real v1 pilot shooting guide

You will take 60 real photos with two phones to evaluate the scanner. The photos are private evaluation material. They are not used for training, checkpoint selection or public sharing. Each photo takes about a minute, and the whole session takes about 1 hour 30 minutes.

## 1. Preparation

**Two phones.** Use two phones of different models. Name one `p1` and the other `p2`, and write this down.

**Camera settings.** Set these on both phones before shooting.

| Item | Setting |
|---|---|
| Format | JPEG. iPhone: Settings → Camera → Formats → Most Compatible. Galaxy: Camera settings → Advanced picture options → High efficiency pictures off |
| Location tags | Off. iPhone: Settings → Privacy & Security → Location Services → Camera → Never. Galaxy: Camera settings → Location tags off |
| Shooting mode | Default photo mode, 1x, default resolution |
| Turn off | Document scan mode and document scan suggestions (Scene optimizer), filters, portrait mode, flash |
| Leave on | HDR, autofocus, auto exposure |

**Books and documents.** Prepare at least 8 books and at least 4 single-sheet documents.

- Give each source a number and write the numbers on paper: books `b01`, `b02` …, single-sheet documents `d01`, `d02` …, and `x00` for photos with no document.
- No source may account for more than 9 of the 60 photos (15%). The evaluation splits dev and test by source, so more sources is better.
- Mostly use books with Korean body text. Mix thicknesses (thin books, thick books of 300 pages or more) and page sizes.
- Do not photograph documents with resident registration numbers, addresses, bank accounts or signatures, or other people's faces.

**Shot list.** Make it on the laptop.

```bash
.venv/bin/python tools/krreal/krreal.py shotlist          # outputs/krreal/v1/shotlist.csv
.venv/bin/python tools/krreal/krreal.py shotlist --print  # show on screen
```

Each phone has 30 rows, numbered 1–30. Print the list or keep it open on a phone, and shoot in order. After each shot, write the source number on that row.

## 2. Strata and counts

| Stratum | Code | Pilot (p1 + p2) | v1 final target |
|---|---|---|---|
| Curved Korean book | `curved` | 12 (6 + 6) | 60 |
| Small print (8 pt or less) | `small` | 6 (3 + 3) | 30 |
| Tables and formulas | `table` | 8 (4 + 4) | 40 |
| Large rotation (90/180/270°, ±30°, wrong EXIF) | `rotate` | 8 (4 + 4) | 40 |
| Cluttered background | `clutter` | 6 (3 + 3) | 30 |
| Small page (under 20% of the photo area) | `tiny` | 5 (3 + 2) | 25 |
| Already flat page | `flat` | 6 (3 + 3) | 30 |
| No page, occluded, cut off, neighboring page | `hard` | 5 (2 + 3) | 25 |
| Two-page spread (labels only) | `spread` | 4 (2 + 2) | 20 |
| Total | | 60 (30 + 30) | 300 |

The pilot is 1/5 of the final target and covers every stratum. The shot list for each stratum is in `tools/krreal/kit.json`, and `shotlist` uses that list as is.

## 3. Common rules

- Hold the phone in your hand, as you would with a scanning app. Do not use a tripod.
- Hold the phone in portrait orientation and place the page upright. Only the `rotate` stratum changes the orientation.
- Get the whole page in the photo. The only exception is cut-off photos (`cut_off` in `hard`).
- Check the focus after each shot. If the photo is blurry, shoot again and delete the blurry one.
- Photograph each page only once per phone.
- Do not edit the photos. Cropping, rotating, filters and adjustments all change the original bytes and the EXIF data.
- Follow the order of the list. If you reshoot, keep the last photo and delete the earlier ones.

## 4. How to shoot each stratum

### Curved Korean book (`curved`, 6 per phone)

Open the book on a desk and aim at one page. Keep all four page corners and a little of the desk around them in the frame. Do not press the page flat with your hand; leave it curved toward the gutter.

1. `thick_left`: left page of a thick book
2. `thick_right`: right page of a thick book
3. `thin`: one page of a thin book or a magazine
4. `lifted`: page edge lifted up
5. `oblique`: at an oblique angle of 30–45°
6. `dim`: dark room with a single light

### Small print of 8 pt or less (`small`, 3 per phone)

Pages with small print, such as footnotes, dictionaries and terms and conditions. Shoot from the closest distance that still fits the whole page, and hold the phone steady with both hands.

1. `footnote`: footnotes, references, index
2. `dictionary`: dictionary or glossary
3. `fineprint`: terms and conditions, manuals, newspapers

### Tables and formulas (`table`, 4 per phone)

Pages with long table rules and pages with many formulas. Make sure to include one table on a curved book page.

1. `table_flat`: single-sheet document with a table
2. `table_book`: book page with a table (curved)
3. `formula`: textbook or workbook with many formulas
4. `grid`: page that mixes tables and formulas, or a form with many boxes

### Large rotation (`rotate`, 4 per phone)

Hold the phone upright as usual, and rotate the book instead.

| Variant | How | Top of the page in the photo |
|---|---|---|
| `r90` (p1) | Book rotated 90° clockwise | Right |
| `r180` (p1, p2) | Book rotated 180° | Bottom |
| `r270` (p2) | Book rotated 90° counterclockwise | Left |
| `skew_pos` (p1) | Book rotated about 30° clockwise | Top, tilted to the right |
| `skew_neg` (p2) | Book rotated about 30° counterclockwise | Top, tilted to the left |
| `exif` (p1, p2) | Hold the phone flat, parallel to the desk, and shoot straight down; turn the phone 90° just before taking the shot | Differs by phone |

In the `exif` pose the phone easily records the wrong orientation. Do not fix the orientation in the gallery; leave it as it is. Whoever labels the photo looks at it and decides whether to set the `wrong EXIF orientation` flag.

### Cluttered background (`clutter`, 3 per phone)

- p1: `pattern` (patterned fabric, bedding), `wood` (wood-grain desk), `mess` (desk with a lot of paper, pens and cups)
- p2: `pattern`, `dark` (dark desk), `hands` (holding the book by hand, with fingers over the page edge)

### Small page (`tiny`, 3 on p1, 2 on p2)

Shoot so that the page covers less than 20% of the photo area. The page's width and height should each be less than half of the photo's. The text does not need to be readable.

- `far`: an A4 sheet or a book page from far away
- `card`: business card, receipt, note paper
- `corner` (p1 only): small page pushed into a corner of the photo

### Already flat page (`flat`, 3 per phone)

Lay a single sheet flat on the desk. These photos check that the scanner does not bend a flat page.

- `topdown`: straight from above, facing the page
- `oblique`: at an angle (perspective only)
- `form`: form document with tables or boxes

### No page, occluded, cut off, neighboring page (`hard`, 2 on p1, 3 on p2)

These photos check that the scanner rejects or warns when it should.

- `none` (p2): photo with no document (empty desk, room). The source number is `x00`.
- `occluded` (p1, p2): a finger, pen or cup covers part of the page and the text
- `cut_off` (p1): one side of the page runs out of the photo (10–30% of the page)
- `neighbor` (p2): aim at one page of an open book, with 1/3 or more of the neighboring page visible

### Two-page spread (`spread`, 2 per phone)

Get both pages of an open book into the photo. This milestone only makes labels for them; evaluation comes in the next milestone.

- `spread_top`: from straight above
- `spread_oblique`: at an angle

## 5. What to avoid

| Avoid | Why |
|---|---|
| Sending photos through a messenger such as KakaoTalk | It recompresses the photos and strips EXIF |
| Cloud uploads with storage saving (compression) | The original bytes change |
| HEIC format | The kit and the scanner read only JPEG |
| Screenshots, editing, filters, document scan mode | The original changes, or the photo is already corrected |
| Location tags, personal information, other people's faces | Keep them out even though the material is private |
| Putting photos in git, public share links | Book photos are under copyright, so they stay private |

## 6. File names

Inside the kit, photos are named `kr1-<phone>-<stratum>-<source>-<number>.jpg`.

```text
kr1-p1-curved-b03-007.jpg
    │  │      │   └── shot list number (from 001 on each phone)
    │  │      └────── source: b01… book, d01… single-sheet document, x00 no document
    │  └───────────── stratum code (table in section 2)
    └──────────────── phone p1, p2
```

You do not need to rename files by hand. `ingest` in section 7 reads the shot list and copies the photos under these names.

## 7. Moving photos to the laptop

Keep the photos in `outputs/krreal/v1/` inside the repository. Git ignores `outputs/`, so the photos are never committed. The kit's commands refuse folders that git tracks.

```text
outputs/krreal/v1/
  incoming/p1/  incoming/p2/   originals transferred from the phones
  shotlist.csv                 shot list
  photos/                      photos copied under kit names
```

1. Transfer the originals from the phones.
   - Android: connect a USB cable and choose "File transfer" in the phone's notification. In a file manager, copy the photos from `DCIM/Camera` to `incoming/p1/` (or `p2/`).
   - iPhone: send the originals with LocalSend (open source, runs on iPhone, Android and Linux). Move the received photos to `incoming/p2/` (or `p1/`).
2. Check: `git check-ignore -v outputs/krreal/v1/incoming/p1/IMG_0001.jpg` must print `.gitignore:outputs/`.
3. In `shotlist.csv`, enter the phone's file name (`IMG_1234.jpg`) in the `file` column and the source number in the `source` column.
4. Copy the photos. The original bytes are copied unchanged and verified with SHA-256. Photos that already exist are not overwritten.

   ```bash
   .venv/bin/python tools/krreal/krreal.py ingest --dry-run
   .venv/bin/python tools/krreal/krreal.py ingest
   ```

5. Build the manifest and look at the missing counts and warnings.

   ```bash
   .venv/bin/python tools/krreal/krreal.py manifest
   ```

   If "no EXIF" appears, the photo was re-saved during the transfer. Transfer the original again. If "GPS" appears, turn off location tags and reshoot.
6. Do not delete the originals in `incoming/`. Keep backups only on a private external disk.

## 8. Checklist before shooting

- [ ] Both phones: JPEG, location tags off, flash off, document scan mode off
- [ ] At least 8 books and at least 4 single-sheet documents, each with a source number
- [ ] Shot list (30 rows per phone)
- [ ] A bright spot and a dark spot, patterned fabric, a wood-grain desk, a dark desk
