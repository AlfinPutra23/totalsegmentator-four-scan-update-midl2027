# Updating TotalSegmentator with four labelled scans: code of five registered looks

Code for the MIDL 2027 submission (Clinical Translation and Validation track) that updates TotalSegmentator 2.13.0 (task `total`) with four labelled scans per draw and compares it, paired on the same scans, with the unmodified model in five registered single looks.

**Research code. Not a medical device. Not for clinical use.**

## Registered looks and their code

| Look | Registration | Code |
|---|---|---|
| WORD test | https://osf.io/c5fyj | `scripts/vccor/final/word_*.py`, `scripts/vccor/final/word_a1/` |
| FLARE-T50 | https://osf.io/ajfrc | `scripts/vccor/final/flaret50/` |
| LCTSC test | https://osf.io/d6ayj | `scripts/vccor/final/lctsc/` |
| WORD-N | https://osf.io/7jser | `scripts/vccor/final/wordn/` |
| LCTSC-T23 | https://osf.io/69kwv | `scripts/vccor/final/lctsc_t23/` |

Post hoc analyses (not registered): `scripts/vccor/final/t50_posthoc/`. Figures and fact tables of the paper: `figures/` (they read stored result files; no image or label is opened). Shared code that the looks import from the development phase sits at its original path under `scripts/`.

Comments and docstrings were edited for this release; the code is unchanged. `FILE_HASHES.tsv` gives each file's sha256 as it ran and, where the file was frozen before a look, the freeze record that lists the same hash (80 of 107 code files). The other files are the figure scripts, the post hoc analyses and helpers that were not part of a freeze. `EDITED_FILES.tsv` lists the 72 edited files with both hashes; with docstrings blanked, each parses to the same syntax tree as the version that ran. All 80 frozen files are in the OSF deposits of the registrations above, as they ran. `derive_t23.py` checks its LCTSC sources against the LCTSC freeze record, so it runs only on those versions.

## What is not here

No images, labels, predictions or model weights. The datasets come from their own sources under their own terms: WORD (Luo et al. 2022), FLARE22 (MICCAI FLARE 2022), LCTSC (TCIA, https://doi.org/10.7937/K9/TCIA.2017.3R3FVZ08). Paths in the code are those of the original machine and must be adapted.

The three LCTSC-adapted models are released separately: https://huggingface.co/alfinwilliam30/totalsegmentator-2.13-lctsc-4scan-update

## Requirements

`requirements.txt` lists the versions the study ran with (Python 3.10).

## Licence

Apache-2.0 (`LICENSE`).
