# Visual Waste Sorting Assistant

A complete waste classification system with a Streamlit web UI and unified CLI pipeline. Built with PyTorch, featuring three CNN architectures (Lite, Standard, Smart) for the TrashNet dataset.

## Python Version

**Python 3.11+** (tested on 3.11, 3.12, 3.13)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r ../requirements.txt
```

## Run Application (Streamlit Dashboard)

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Train Model on Full TrashNet Dataset

```bash
# 1. Download and split TrashNet (run once)
python ../scripts/prepare_dataset.py --confirm

# 2. Train model
python ../scripts/train_model.py --epochs 20 --lr 0.001 --depth 4 --base-filters 32 --dropout 0.5 --augmentation basic
```

Model weights saved to `../models/waste_classifier.pth`, history to `../models/training_history.pkl`.

## Run Ablation Experiments

```bash
# Run all ablation studies (depth, filters, dropout, lr, augmentation)
python ../scripts/run_ablation.py --param all --epochs 15

# Run specific ablation
python ../scripts/run_ablation.py --param depth --epochs 15
python ../scripts/run_ablation.py --param filters --epochs 15
python ../scripts/run_ablation.py --param dropout --epochs 15
python ../scripts/run_ablation.py --param lr --epochs 15
python ../scripts/run_ablation.py --param augmentation --epochs 15
```

Results saved to `../ablation_results/` as JSON and PNG plots.

## Run Tests

```bash
python -m pytest ../tests/test_project_11_sanity.py -v
```

## Project Structure (for Submission)

```
code/
├── README.md                 # This file
├── requirements.txt          # Pinned dependencies
├── starter/
│   ├── __init__.py
│   ├── student_core.py       # CNN: build_model, train, predict
│   └── app.py                # Streamlit dashboard
├── scripts/
│   ├── prepare_dataset.py    # TrashNet download & split
│   ├── make_sample_images.py # Synthetic sample generator
│   ├── train_model.py        # Training script with ablation params
│   └── run_ablation.py       # Ablation experiment runner
├── tests/
│   └── test_project_11_sanity.py
└── data/
    ├── LICENSE.md
    ├── README.md
    └── sample/               # ≤20 synthetic images (offline smoke tests)
        ├── cardboard/
        ├── glass/
        ├── metal/
        ├── paper/
        ├── plastic/
        └── trash/
```

## 📄 License

TrashNet dataset: MIT License (Gary Thung, Yangjie Li)
Code: MIT License

## 🔗 Links

- **GitHub Repository**: https://github.com/manhanbanhcanh/ain-waste-classifier/tree/final
- **TrashNet Dataset**: https://github.com/garythung/trashnet
- **Streamlit Docs**: https://docs.streamlit.io
- **PyTorch Docs**: https://pytorch.org/docs
