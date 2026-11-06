# Training data of the tracer and its terms

Checked 2026-11-06 against the licence files in our local copies and the sources' pages. This is an engineer's reading, not legal advice.

Licence of this release: code Apache-2.0, weights CC BY-NC 4.0, chosen with the open question below in view.

## trace8 (`weights/trace8_fp16.pt`): data/trace_mix_v1, 220 000 lines

| source | share | stated terms | with CC BY-NC 4.0 on the weights |
|---|---|---|---|
| our tablet notebooks (data/ink_lines_v6, 13 writers, about 45k lines) | 28% | private, not published; the writers agreed to this use | fine: the data is not published |
| Google Fonts handwriting: 237 fonts passed our filter, 225 OFL, 12 Apache-2.0 | 22% | OFL / Apache-2.0 | fine |
| pseudo-script (scripts/pseudo_script.py, our generator) | 16% | ours | fine |
| BRUSH (Brown), 170 writers | 13% | "may only be used for non-commercial research purposes"; requests a citation of the paper. Says nothing about derived models. Other uses: contact the authors (Prof. James Tompkin) | compatible on NC, but its "research" is narrower than CC BY-NC. Cite it and state the restriction on the card |
| Make Me a Hanzi medians (graphics.txt) | 8% | Arphic Public License (font licence; requires derivatives of the Font to stay under it) | no conflict seen: weights are not a font. Credit it |
| MathWriting (Google) | 6% | **CC BY-NC-SA 4.0** | **ShareAlike: conflict if the weights count as based on the data** |
| korean-handwriting-strokes (hangul.ink, handwriting by Jo YW) | 4% | **CC BY-SA 4.0**. "Use for education, art, research, and products is welcome"; attribution string is given | **ShareAlike: same conflict, and NC is an added restriction** |
| OUHD-L (Urdu) | 3% | CC BY 4.0 (Zenodo 10.5281/zenodo.20642162), consent covered per its card | fine, with attribution |

The text shown in the font and Hanzi lines came from the transcriptions of IAM, RIMES-2011, NorHand-v3, KHATT, CASIA-HWDB2, Korean Handwriting Strokes, OUHD-L and our own labels (scripts/ext_texts.py). Only the words were used, no images. The terms of those transcription sets were **not checked**.

## inkfix1 (`weights/inkfix1.pt`): data/ink_lines_v7

Synthetic pairs rendered from our own tablet lines only (65k lines, 14 writers, 3 of them held out). No open dataset and no HWR200 crops (those went into inkfix1r, which is not released).

## The ShareAlike question

Creative Commons' AI-training guide says that if models based on SA content are shared publicly, they would need the same licence as the original works. The same guide does not say a trained model is adapted material, so whether SA applies at all is unsettled.

If it does apply, there are two consequences:
- CC BY-NC 4.0 on the weights does not satisfy MathWriting (BY-NC-SA) or the Korean set (BY-SA).
- No single licence satisfies both, because BY-SA adaptations cannot go under BY-NC-SA. Changing the weights to BY-NC-SA would fix MathWriting only.

We publish the weights under CC BY-NC 4.0 and state this openly: the question is not resolved, and we have not asked the rights holders. If you hold rights in one of these sets and object, please open a discussion on this repository.

## Attribution

- BRUSH: Atsunobu Kotani, Stefanie Tellex, James Tompkin, "Generating Handwriting via Decoupled Style Descriptors", ECCV 2020; non-commercial research use only.
- MathWriting: Google LLC, CC BY-NC-SA 4.0.
- *Korean Handwriting Strokes by Jonathan Baudanza (hangul.ink), handwriting by Jo YW, CC BY-SA 4.0; transcripts partly from Korean Wikipedia (CC BY-SA 4.0).*
- OUHD-L: Zenodo 10.5281/zenodo.20642162, CC BY 4.0.
- Make Me a Hanzi (Shaunak Kishore), graphics derived from Arphic PL fonts under the Arphic Public License.
- Google Fonts handwriting families, OFL / Apache-2.0.

Because of BRUSH, treat the tracer weights as non-commercial research use, which is narrower than CC BY-NC.

## Example image

`example_open_lines.png` uses only:
- Korean Handwriting Strokes (CC BY-SA 4.0) and OUHD-L (CC BY 4.0), rendered from the benchmark's held-out lines;
- Digital Peter (hand of Peter I; the Hugging Face card says MIT; the competition repository is MIT and does not state dataset terms separately). This set is not in the training mix.

The image itself is under CC BY-SA 4.0 because of the Korean render.
