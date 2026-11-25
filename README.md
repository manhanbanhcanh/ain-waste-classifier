# Visual Waste Sorting Assistant

## Project description

This project is for our lecture class about Artificial Intelligence.

The project requirement are: build, train, and evaluate a CNN-based waste classifier capable of recognizing six types of recyclable/trash materials under realistic image conditions.

For additional information (original repository's README.md), see [INFOMATION.md](INFORMATION.md)

## Provided materials & resources

### File structure

```text
project-11-waste-classifier/
├── data/
│   ├── README.md                 # Data dictionary, class definitions, and split procedures
│   ├── LICENSE.md                # Dataset provenance, pinned commit, MIT license, and SHA-256 hash
│   └── sample/                   # Offline smoke-test images (≥1 per class, including edge cases)
├── scripts/
│   ├── prepare_dataset.py        # Automated download and train/val/test splitting utility for full TrashNet
│   └── make_sample_images.py     # Offline synthetic image generation utility
├── starter/
│   ├── student_core.py           # Core functions: build_model, train, predict
│   └── app.py                    # Streamlit dashboard with real-time classification
├── tests/
│   ├── train/
│   │   ├── train.py              # Training script with progress bars
│   │   └── GUIDE.md              # Training guide
│   └── test_project_11_sanity.py # Unit tests
├── models/                       # Trained model weights (git-ignored)
│   ├── best_model_lite.pt
│   ├── best_model_standard.pt
│   └── best_model_smart.pt
└── requirements.txt              # Python dependencies
```

### Starter infrastructure and our implementation

1. `scripts/prepare_dataset.py` handles optional downloading and partitioning of the full 2,527-image TrashNet corpus into train, validation, and test splits (git-ignored).
2. `starter/app.py` provides the upload and confidence-visualization frontend. Students implement the neural network pipeline in `starter/student_core.py`:
   - **`build_model(input_shape, num_classes)`** — defines the CNN architecture (convolutional, pooling, dropout, and fully connected layers) terminating in a Softmax output.
   - **`train(model, train_dir, validation_dir, epochs)`** — runs the training loop with categorical cross-entropy loss and an adaptive optimizer (Adam or SGD with momentum).
   - **`predict(model, image_path)`** — returns the predicted class label and confidence probability for a single image.

We must build and train our own CNN. Shipping downloaded pre-trained solution weights (`.h5`, `.pt`, `.ckpt`, `.pkl`) or calling hosted vision/LLM APIs is strictly prohibited.

### Installation and Execution

```bash
git clone https://github.com/manhanbanhcanh/ain-waste-classifier
pip install -r requirements.txt
python scripts/prepare_dataset.py --confirm
```

### Training Models

```bash
# Quick test (sample data)
python tests/train/train.py --variant standard --epochs 5

# Full training (recommended)
python tests/train/train.py --variant standard --full --epochs 30 --progress

# Maximum accuracy (Smart variant)
python tests/train/train.py --variant smart --full --epochs 40 --progress
```

See `tests/train/GUIDE.md` for complete training guide.

### Running the App

```bash
streamlit run starter/app.py
```

---

For license problem please contact me through my social media at [ducmanh.space](https://ducmanh.space/)
