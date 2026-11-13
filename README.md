# Scrawler

Handwriting from **one line** of the writer's hand, with no fine-tuning: whole pages of notes with LaTeX formulas, or single lines the way [synthink.dev](https://synthink.dev) writes them. The output is pen strokes, not pixels, so the PDF and SVG are sharp at any zoom.

![A page written by a fine-tuned version of the models](examples/demo_authors_hand/page_05.png)

*The page above is in the author's hand, written by personal fine-tuned models that are **not** released. For what the released weights give out of the box, see [this page](examples/demo_released_weights/page_05.png).*

This repository holds the code. The weights are on [Hugging Face](https://huggingface.co/levossadtchi/Scrawler-open), and `download_weights.py` fetches them. Training code and data are not included.

## What to use

| You want | Run | Model | Languages | What it cannot do |
|---|---|---|---|---|
| Pages of notes with LaTeX formulas | `write_pages.py` | pages model | Russian prose and mathematics | the hand is copied only loosely |
| A line, or a few lines, in someone's hand, as on [the site](https://synthink.dev/generate/) | `write_line.py` | site model | Russian | no formulas, no page layout |
| The same in English or Russian | `write_line.py --model bilingual` | bilingual model | English and Russian | no formulas, no page layout; **non-commercial research only** |

English pages with formulas are not possible yet: `write_pages.py` runs the pages model only. English prose through the pages model was not measured.

Every route needs one line of the writer's hand as the style, the **reference line**. It can come from a photo, from a note app on a tablet, or from your own stroke data: see [Your reference line](#your-reference-line). Without one, a traced line of the open HWR200 dataset is used.

## Quick start

Requires Python; for formulas also Node.js.

```bash
git clone https://github.com/Levos06/Scrawler-open.git
cd Scrawler-open
pip install -r requirements.txt
python download_weights.py                 # the pages model, about 80 MB
(cd tools/mathjax && npm ci)               # formula layout
python write_pages.py examples/physics_oscillator.txt out
```

For the other routes, fetch their weights first. The flags can be combined.

| Flag | Adds | Size |
|---|---|---|
| `--site` | site model, for `write_line.py` | about 75 MB |
| `--bilingual` | bilingual model, for `write_line.py --model bilingual` | about 100 MB |
| `--tracer` | the tracer, to take a reference line from a photo | about 60 MB |

```bash
python download_weights.py --site
python write_line.py "Мороз и солнце; день чудесный!" out_line
```

The code selects CUDA, MPS or CPU by itself; set `SCRAWLER_DEVICE` to force one. Results depend on the device: the same seed gives the same output only on the same device. Tested on Apple Silicon (MPS and CPU); the CUDA path has not been run with this code.

## Your reference line

One line of ordinary prose, 30-50 characters, without formulas, underlines or corrections. Whatever the source, it ends as an `.npy` file from `make_ref.py`, which every script takes as `--ref`.

### From a photo or scan

Crop the photo to **one line** and run it through the [tracer](trace/README.md), which recovers the pen strokes from the image.

```bash
python download_weights.py --tracer
pip install -r trace/requirements.txt
python trace/scrawler_trace.py my_line.png --out my_line.json --svg my_line.svg
python make_ref.py my_line.json my_line.npy
```

- Crop tightly to one line. Parts of the lines above and below can get into the trace.
- Use dark pen on light paper. Colour photos, dark backgrounds and low contrast work poorly.
- Look at `my_line.svg` before going on: it shows the traced strokes.

![A scan, its trace and new text in that style](examples/from_scan.png)

*Top: a real scan on squared paper (HWR200, open data). Middle: the trace. Bottom: new text and formulas in that style. One example, not a measurement.*

The tracer weights have their own licence caveat, see [its training data](trace/TRAINING_DATA.md).

### From a tablet

If the line was written in a note app, its pen strokes are in the app's file and no tracer is needed. `tablet/app_line.py` reads GoodNotes, Notewise, Huawei Notes, Kilonotes and Samsung Notes. How to export from each app, and what to do with the others, is in [tablet/README.md](tablet/README.md).

```bash
python tablet/app_line.py notes.goodnotes my_line.json --page 3 --line 2 --svg check.svg
python make_ref.py my_line.json my_line.npy
```

### Your own stroke data

Write the strokes as a JSON list of strokes, each a list of `[x, y]` points with y pointing down, in writing order, in any units. That is what `make_ref.py` takes.

## Pages with formulas: `write_pages.py`

```bash
python write_pages.py notes.txt out --ref my_line.npy
```

The input:

```text
# A heading
A line of text with an inline formula $\omega_0 = \sqrt{k/m}$ in it.
$x(t) = A\cos(\omega_0 t + \varphi)$

An empty line above makes a paragraph gap.
```

Long lines are wrapped at about 42 characters (`--wrap`), never inside `$...$`. Formulas use the LaTeX that MathJax understands. `python write_pages.py --help` lists the other options.

`out/` (the folder must not exist yet) gets:

- `pages.pdf`: vector, every stroke a smooth curve
- `page_NN.svg` and `page_NN.png` (300 dpi)
- `strokes.json`: the strokes; `python render_pages.py out/strokes.json thick --pen 1.4` redraws them without the models
- `report.json`: per-line timing and what wrote each glyph

The example of the quick start is 11 pages and took under 7 minutes on an Apple M5 Max (MPS). The CPU is much slower: a 4-page test with many formulas took 74 minutes.

How a page is written:

| On the line | Written by |
|---|---|
| text only | **Text model**: a diffusion denoiser over pen points, with a duration planner for each character |
| formula only | **Formula module**: MathJax places every glyph, a diffusion **glyph model** draws each one, and a classifier picks the most recognisable variant |
| text + `$formula$` | The text model writes the line, then the formula spans are replaced by the formula module |

The style vector comes from a line-level style encoder. Fraction bars, radicals, arrows and `|x|` are drawn as pen lines by code, not by a model, so they do not take the writer's style.

## Lines in a hand: `write_line.py`

```bash
python write_line.py "Мороз и солнце; день чудесный!" out_line --ref my_line.npy --ref-text "what your line says"
python write_line.py lines.txt out_line --ref my_line.npy --ref-text "..."                 # one line of text per row
python write_line.py "One line, and the rest is written by your hand." out_line --model bilingual --ref my_line.npy --ref-text "..."
```

`out_line/` gets `line.svg`, `line.png` and `line.pdf` with the chosen attempt of every line, `attempts.png` with all attempts, and `strokes.json`. A line takes 5-10 seconds on an Apple M5 Max (MPS).

Give `--ref-text`, what is written in the reference line, when you know it. It sets how wide the writer writes; without it the recogniser reads the reference, which gets most letters right but not all.

### The recipe

The same for both models, in `scrawler/site_model.py`:

1. The planner gives the number of pen points for every character. These are multiplied by the writer's spread: the length of the reference line in points over the length the planner gives for the text of that line, clipped to [0.6, 1.4].
2. The text model writes the line four times, with guidance 1.5 and 20 steps. It attends to the pen points of the reference line itself, not only to its style vector.
3. A recogniser keeps the attempts whose character error rate is within 0.02 of the lowest, and of those a writer classifier picks the one closest to the reference.

The bilingual model changes two things, both measured on it: an English line is chosen by its English recogniser, and a Russian line that starts with a capital letter gets for that letter the length the planner gives it inside a phrase (without this the first word often gets a stray stroke).

### Site model

This is what [synthink.dev](https://synthink.dev/generate/) runs in the browser, as ONNX converted from these weights. The code was checked against the ONNX files the site serves: from the same noise the pen points agree to 0.0003 of the line height and the same attempt is chosen.

| On the site | Here | What it does |
|---|---|---|
| `den.onnx`, `refpts.onnx` | `ckpt_dl/u3ref/fp16/step060000_fp16.pt` | text model, with attention to the reference points |
| `planner.onnx` | `ckpt_dl/planner7/best.pt` | pen points per character |
| `semb.onnx` | `ckpt_semb3_all/best.pt` | style vector of the reference line |
| `rec.onnx` | `ckpt_ctc4/ctc_fp16.pt` | recogniser: how legible an attempt is |
| `scls.onnx` | `ckpt_stylecls/embed_fp16.pt` | writer classifier without its class layer: how close an attempt is to the reference |
| `fix.onnx`, `tenc.onnx`, `tstep.onnx` | `trace/weights/` | ink restorer and tracer, for photos |

Its text model is the text model of the pages model trained for 60 000 more steps, with traced lines of 190 HWR200 writers added to the data and attention to the reference points added to the network.

### Bilingual model

**Licence.** It was trained on [BRUSH](https://dsd.cs.brown.edu/) (Brown University), whose terms allow non-commercial research use only, and its weights inherit that: `ckpt_dl/u4s2/`, `ckpt_semb4/` and `ckpt_ctc6/` are for non-commercial research only, on top of CC BY-NC 4.0.

Made from the site model in three steps:

1. **Text model:** 60 000 more steps on the site model's data, with underlines removed from the tablet lines (they made the first word of a line come out underlined), plus English pen lines of 158 BRUSH writers. 12 more BRUSH writers were kept out of all training to measure on.
2. **Style encoder:** trained further to tell BRUSH writers apart, while staying close to its old vectors on Russian lines.
3. **Text model:** 20 000 more steps with the new style encoder.

Its English recogniser is the site's recogniser trained further on the same mix.

## Weights

| Model | Path | What | Size |
|---|---|---|---|
| all | `ckpt_dl/planner7/best.pt` | duration planner | 14 MB |
| pages, site | `ckpt_semb3_all/best.pt` | style encoder | 7 MB |
| pages | `ckpt_dl/b4styc/fp16/step003000_fp16.pt` | text model | 45 MB |
| pages | `ckpt_glyph_dn_u2s/fp16/step050000.pt` | glyph model | 14 MB |
| pages | `ckpt_dl/glyph_cls2.pt` | glyph classifier | 1 MB |
| site | `ckpt_dl/u3ref/fp16/step060000_fp16.pt` | text model | 51 MB |
| site, bilingual | `ckpt_ctc4/ctc_fp16.pt` | recogniser (Russian lines) | 20 MB |
| site, bilingual | `ckpt_stylecls/embed_fp16.pt` | writer classifier without its class layer | 4 MB |
| bilingual | `ckpt_dl/u4s2/fp16/step020000_fp16.pt` | text model; non-commercial research only | 51 MB |
| bilingual | `ckpt_semb4/best.pt` | style encoder; non-commercial research only | 7 MB |
| bilingual | `ckpt_ctc6/ctc_fp16.pt` | recogniser (English lines); non-commercial research only | 20 MB |
| photos | `trace/weights/` | tracer and ink restorer | 62 MB |

The folder names are those of the training runs. `download_weights.py` fetches each group, see [Quick start](#quick-start).

## Code

| Path | What |
|---|---|
| `write_pages.py` | text file -> pages: wrapping, page layout, the order of the steps |
| `write_line.py` | lines of text in the hand of a reference line, the way the site does it |
| `make_ref.py` | your strokes -> a reference line |
| `scrawler/text_model.py` | text model of the pages model and the duration planner |
| `scrawler/site_model.py` | text model of the site and bilingual models, recognisers, writer classifier, the recipe |
| `scrawler/style_encoder.py` | one line -> style vector |
| `scrawler/glyph_model.py` | glyph model and the classifier that picks variants |
| `scrawler/formula.py` | formula layout, pen-line marks, inserting formulas into text |
| `scrawler/render.py`, `render_pages.py` | strokes -> PDF, SVG, PNG |
| `trace/` | the tracer, one file |
| `tablet/` | a line out of a note app: readers of GoodNotes, Notewise, Huawei Notes, Kilonotes and Samsung Notes files |
| `tools/mathjax/layout.mjs` | LaTeX -> glyph boxes |
| `data/glyph_stroke_counts.json` | per glyph class, how many real glyphs had 1, 2, ... strokes; counts only |

## Evaluation

Four measures are used below. They are on different scales, so numbers of different measures cannot be compared.

- **Judge:** an LLM judge (Gemini 4 Argon) lists errors in each line; "clean" means no error of severity 2 or higher.
- **Legible:** the recogniser's character error rate is at most 0.05 (`ckpt_ctc4` for Russian, `ckpt_ctc6` for English).
- **Russian classifier:** cosine between a generated line and the writer's real line, by the writer classifier `ckpt_stylecls` (trained on Russian tablet writers).
- **Bilingual classifier:** the same kind of classifier trained on both languages (not released). "Own writer found" means the generated line is closer to its own reference than to the other 19.

### Pages model

240 lines of real notes from writers the generator never saw, each written from one reference line by the same writer, scored by the judge. Pairs are re-sampled for the 95% intervals.

| Run | Clean, % [95%] | Prose | Mixed | Formula |
|---|---|---|---|---|
| Real handwriting of the same lines | 45.0 [38.8, 51.2] | 47.0 | 35.7 | 51.4 |
| Text model alone | 47.1 [40.8, 53.3] | 63.0 | 41.4 | 30.0 |
| **Pages model: text model + formula module** | **61.3 [55.0, 67.5]** | 63.0 | 54.3 | 65.7 |
| Same, with a library of real glyphs (not released) | 65.4 [59.2, 71.2] | 62.0 | 58.6 | 77.1 |

- The judge rewards legibility. Scoring above real notes means the output is tidier than hurried human writing, not that it looks more human.
- Prose is the same set of generated lines in every run but the first, so 62-63 shows judge noise of a few points.
- The private library variant scores 4.2 points higher, 95% interval [-9.6, +1.2] for the difference. That is within the noise, but a small loss from removing it is possible.

Style, by the style encoder `ckpt_semb3_all` (a third scale, used only here): cosine to the writer's real line 0.59-0.68 for generated lines, against 0.55-0.70 between real lines of the same writers, on 3 unseen writers.

### Site model against the pages model

Traced lines of 9 writers the models never saw. Russian classifier: 0.71 to the writer's own line for the site model, 0.47 for the text model of the pages model; 0.46 to other writers for both. Judge on the 240 lines above: 47.1% clean, the same as "Text model alone". By eye the gain in likeness is modest.

### Bilingual model against the site model

20 hands per language, one reference line each, best of four attempts. English: 12 BRUSH writers the models never saw and 8 the bilingual model saw. Measured on a GPU; the same seed gives other lines on a Mac.

| | English: own writer found (bilingual classifier) | English: cosine own minus others (bilingual classifier) | English: legible | Russian: cosine to own line (Russian classifier) | Russian: legible | Russian: stray stroke on the first word |
|---|---|---|---|---|---|---|
| Site model | 5% | 0.06 | 95% | 0.64 | 95% | 50% |
| **Bilingual model** | **65%** | **0.29** | **95%** | **0.67** | **90%** | **5%** |
| Another real line of the same writer | 85% | 0.49 | | | | |

The English hands are much closer to their writers than with the site model, but still well short of real lines. Without the best of four, English lines often lose spaces between words: 82% legible over all attempts.

## Known weaknesses

- All text models sometimes swap, drop or add a letter. There is no spell check.
- The site model often adds a stray stroke to the first word of a line, an underline or a strike-through: in the chosen attempt for 10 of 20 test hands. The bilingual model does it for 1 of 20.
- Pages: a one-letter formula inside a sentence can collide with the neighbouring word.
- Pages: about 2% of glyphs are the best of the variants the classifier rejected, and may be misshapen.
- Pages: rare symbols without their own class are drawn as a similar one (for example `\simeq` as `\approx`).
- Pages: a heading of one short word may come out smaller than the other lines.
- Pages: a line of several formulas with almost no plain text between them is written by the text model alone instead of the formula module (`fallback_text_model` in `report.json`).

## Training data

- **Pages model:** tablet handwriting of 13 writers who agreed to this use, about 63 000 lines and 86 000 glyphs. Writer names are removed from the weights. The notes themselves are not published.
- **Site model:** the same, plus traced lines of 190 writers of [HWR200](https://huggingface.co/datasets/AntiplagiatCompany/HWR200) (Apache-2.0).
- **Bilingual model:** the same, plus English pen lines of 158 writers of [BRUSH](https://dsd.cs.brown.edu/) (non-commercial research use only). Its style encoder and English recogniser were trained further on the same lines.
- **Style encoder:** the same kind of tablet data, plus traced lines of HWR200.
- **Recogniser:** the tablet lines above, the same lines rendered to images and traced back, and traced lines of HWR200.
- **Writer classifier:** tablet lines of 15 writers: 12 of the 13 above and three more.
- **Tracer:** a mix of tablet notes and open sets, some with non-commercial or ShareAlike terms. See [trace/TRAINING_DATA.md](trace/TRAINING_DATA.md), including the open question about ShareAlike.

## Responsible use

This is for notes, experiments and research. Do not use it to forge signatures, documents or exam work, or to imitate someone's handwriting without their consent.

## License

Code: Apache-2.0 ([LICENSE](LICENSE)). The weights, downloaded separately from Hugging Face, are under CC BY-NC 4.0. The bilingual model is for non-commercial research only (BRUSH). The tracer weights carry further conditions from their training data, see [trace/TRAINING_DATA.md](trace/TRAINING_DATA.md). Third-party components: [NOTICE](NOTICE).
