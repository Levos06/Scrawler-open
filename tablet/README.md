# A line from a tablet

If your handwriting already lives in a note app, the pen strokes are in its files and no tracer is needed: the reference line is taken exactly as you wrote it.

The steps are the same for every app:

1. Get the notebook out of the app **in the app's own format**, not as a PDF or a picture (those keep the look of the ink, not the pen path).
2. Find the page and the line, and look at the picture to check:

   ```bash
   python tablet/app_line.py notes.goodnotes
   python tablet/app_line.py notes.goodnotes my_line.json --page 3 --line 2 --svg check.svg
   ```

3. Make the reference and write:

   ```bash
   python make_ref.py my_line.json my_line.npy
   python write_line.py "Any text you like." out_line --ref my_line.npy --ref-text "what the line says"
   python write_pages.py notes.txt out --ref my_line.npy
   ```

Pick a line of ordinary prose, 30-50 characters, without formulas, underlines or corrections. The surest way is to write one such line on a new empty page: then `--page N` alone takes the whole page and nothing has to be split.

## By app

| App | File to export | Command |
|---|---|---|
| GoodNotes 5 and 6 (iPad) | the notebook as a Goodnotes document, `.goodnotes` | `python tablet/app_line.py notes.goodnotes ...` |
| Notewise (iPad, Android) | the note as a Notewise file, `.notewise` | `python tablet/app_line.py notes.notewise ...` |
| Huawei Notes | the note as a note file, `.hinote` | `python tablet/app_line.py notes.hinote ...` |
| Kilonotes (Android) | the notebook in the app's own export, a `.zip` | `python tablet/app_line.py notes.zip ...` |
| Samsung Notes | the note as a Samsung Notes file, `.sdocx` | two steps, see below |
| any other app | a picture of the page | the photo route, see below |

The names of the export menus change between versions of the apps, so they are not listed here: look for "export" or "share" and choose the format named in the table.

### Samsung Notes

The `.sdocx` format is read by the npm package [@twango/sdocx](https://github.com/twangodev/sdocx) (GPL-3.0), which is installed separately and is not part of this repository. It needs Node.js.

```bash
(cd tablet/samsung && npm ci)
node tablet/samsung/sdocx2ink.mjs pages notes.sdocx          # writes pages/notes_p00.json, notes_p01.json, ...
python tablet/app_line.py pages/notes_p00.json my_line.json --line 2 --svg check.svg
```

### Any other app

Apple Notes, Notability, OneNote, Xournal++ and the rest are not read here. Use the photo route: export the page as a picture or take a screenshot, crop it to one line and run the [tracer](../trace/README.md). Ink from a screen is clean, with no paper, shadow or ruling, which is the easy case for it.

```bash
python download_weights.py --tracer
pip install -r trace/requirements.txt
python trace/scrawler_trace.py my_line.png --out my_line.json --svg check.svg
python make_ref.py my_line.json my_line.npy
```

### Your own strokes

If you can get the strokes yourself, write them as a JSON list of strokes, each a list of `[x, y]` points, y pointing down, in writing order, in any units. That is all `make_ref.py` needs.

## What to check

- **The picture.** `--svg check.svg` draws the page with the lines numbered and the chosen one in red. The page is split into lines by where the ink is densest from top to bottom. That works on ordinary rows of text and can go wrong on crowded pages, formulas and drawings.
- **The order of the strokes.** The models read the line in writing order. Strokes come out in the order the app stores them, which is the writing order unless you moved or pasted ink afterwards.
- **Highlighter and shapes** are skipped: only pen strokes are taken.

## What this was tested on

The readers are reverse-engineered: none of these formats has a public description. Each one was run on real notebooks and read all of them: GoodNotes 5 and 6 (56 notebooks), Notewise (27), Huawei Notes (41 exports from two versions of the app, 34 of them with ink), Kilonotes (8 exports), Samsung Notes (31 notes, through @twango/sdocx). A newer version of an app may write something they do not know; if `app_line.py` finds no strokes or draws nonsense, use the photo route.

The files are only read as data. Nothing in them is executed and nothing is unpacked to disk.
