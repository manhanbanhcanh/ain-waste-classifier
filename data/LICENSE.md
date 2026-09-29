# TrashNet dataset — source, license, and checksum

This project's optional full dataset comes from **one** source only:

- Repository: <https://github.com/garythung/trashnet>
- Pinned commit: `6fa2b878c6c1b4304b91109070ce0edf9279bb31` (2023-06-02, "Change dataset link to HuggingFace")
- Dataset file inside the repo at that commit: `data/dataset-resized.zip`
- Classes (six, TrashNet's own): `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`
- Reported image counts at that commit: 403 cardboard, 501 glass, 410 metal, 594 paper, 482 plastic, 137 trash (2527 total)
- Direct file URL used by `scripts/prepare_dataset.py`:
  `https://raw.githubusercontent.com/garythung/trashnet/6fa2b878c6c1b4304b91109070ce0edf9279bb31/data/dataset-resized.zip`

Do not substitute a mirror, a Kaggle re-upload, or the Hugging Face copy linked from the repo's newer README. `scripts/prepare_dataset.py` only ever downloads the URL above.

## License

MIT License, Copyright (c) 2017 Gary Thung (from the repository's `LICENSE` file at the pinned commit).

```text
MIT License

Copyright (c) 2017 Gary Thung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Checksum (pinned; archive is not shipped)

The student zip does **not** contain `dataset-resized.zip`. The digest below was
recorded from an instructor preflight download of the pinned URL on 2026-09-19
(42,834,870 bytes). `scripts/prepare_dataset.py --confirm` verifies against this
value by default.

```text
sha256  0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2
size    42834870
url     https://raw.githubusercontent.com/garythung/trashnet/6fa2b878c6c1b4304b91109070ce0edf9279bb31/data/dataset-resized.zip
```

Re-verify if you download it yourself:

```bash
curl -sL -o dataset-resized.zip \
  "https://raw.githubusercontent.com/garythung/trashnet/6fa2b878c6c1b4304b91109070ce0edf9279bb31/data/dataset-resized.zip"
shasum -a 256 dataset-resized.zip
```

## Why this file ships without downloading anything

The instructor kit must be releasable even if GitHub, this specific repository, or
the network is unavailable. `data/sample/` (see `README.md` in this folder) ships a
small set of original synthetic images so every project group can do offline
smoke tests, build `starter/student_core.py`, and run `tests/test_project_11_sanity.py`
without ever running `scripts/prepare_dataset.py`.
