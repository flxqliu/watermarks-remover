# research/paper/ -- arXiv v1 skeleton (gaps C1/C2)

Status: **skeleton only** -- placeholder prose + TODOs, **no real
experiment numbers**. The generated tables/figures do not exist yet.

## Files

| File | Contents | Source |
| --- | --- | --- |
| `main.tex` | Article-class skeleton; sections 1--10 (02 §5); table/figure placeholders (02 §6); ACL style drop-in note | `research/02-paper-outline.md` |
| `refs.bib` | 32 verified references, 03 A--E (@misc + eprint for arXiv; venue labels in `note`; `% re-verify at submission`) | `research/03-related-work.md` |
| `abstract.tex` | ~150-word abstract draft (02 §2); TODO: recheck numbers after the run | `research/02-paper-outline.md` §2 |
| `ethics.tex` | Ethics statement (04 §3 adapted to a paragraph; disclosure note 04 §5) | `research/04-ethics-and-legal.md` |
| `acknowledgments.tex` | MarkLLM / THU-BPM (Generative-Watermark-Toolkits) + HF models | `research/04-ethics-and-legal.md` §5, 03 §D |

## Build

Requires a TeX distribution with `natbib` (pdflatex + bibtex). From
this directory:

```bash
pdflatex main
bibtex  main
pdflatex main
pdflatex main
```

`main.tex` will **not compile until the generated tables and figures
exist** (`tables/t1.tex`--`tables/t7.tex`,
`figures/f1.pdf`--`figures/f6.pdf`). The dev environment here has no
pdflatex; syntax is checked with a brace-balance script instead
(research/tests/test_paper_skeleton.py).

## Missing before submission (see research/05-arxiv-readiness.md)

- **Tables/figures (gap C3)**: run `research/scripts/make_tables.py`
  and `research/scripts/make_figures.py` (to be implemented; gaps
  B3/C3) to emit `tables/` and `figures/` from the run results --
  they are the real-number inputs referenced by `main.tex`.
- **Real numbers**: replace every TODO in §4--§6 with the actual run
  output (gaps A1--A7, B1--B3, C4); abstract numbers re-checked (W2).
- **ACL style**: drop in `acl2024.sty` and switch the preamble to the
  ACL template at submission (gap C1); swap
  `\bibliographystyle{plainnat}` if the venue requires its own style.
- **Citations**: re-verify every ID/venue at submission (03 header;
  `% re-verify at submission` in refs.bib); every bib entry must be
  cited (or pruned) before upload.
- **Reproducibility**: manifest, pins, release URLs (gaps A7/E3) and
  the data-availability / reproducibility statements (01 §7--8).

## Notes

- `research/` is gitignored; use `git add -f` if you want the paper
  tracked.
- Keep `ethics.tex` in sync with `research/04-ethics-and-legal.md`
  and the abstract with `research/02-paper-outline.md` §2.
