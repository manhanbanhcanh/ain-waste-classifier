# Visual Waste Sorting Assistant

**By**: Duc Manh, Trang Quynh, Bao Chi (Group no.9 - Topic no.11) <br>
**Class**: 66FIT3AIN.TT02 <br>
**Lecturer**: Nguyen Thanh Vinh _[(github)](https://github.com/vinhnt21)_

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
│   ├── student_core.py           # Algorithmic stubs: build_model, train, and predict
│   └── app.py                    # Streamlit image upload and real-time classification dashboard
├── tests/
│   └── test_project_11_sanity.py # Unit tests verifying dataset contracts and stub signatures
└── requirements.txt              # Pinned Python package dependencies (including streamlit, pillow)
```

### Starter infrastructure and our implementation

1. `scripts/prepare_dataset.py` handles optional downloading and partitioning of the full 2,527-image TrashNet corpus into train, validation, and test splits (git-ignored).
2. `starter/app.py` provides the upload and confidence-visualization frontend. Students implement the neural network pipeline in `starter/student_core.py`:
   - **`build_model(input_shape, num_classes)`** — defines the CNN architecture (convolutional, pooling, dropout, and fully connected layers) terminating in a Softmax output.
   - **`train(model, train_dir, validation_dir, epochs)`** — runs the training loop with categorical cross-entropy loss and an adaptive optimizer (Adam or SGD with momentum).
   - **`predict(model, image_path)`** — returns the predicted class label and confidence probability for a single image.

We must build and train their our CNN. Shipping downloaded pre-trained solution weights (`.h5`, `.pt`, `.ckpt`, `.pkl`) or calling hosted vision/LLM APIs is strictly prohibited.

### Installation and Execution

```bash
git clone https://github.com/manhanbanhcanh/ain-waste-classifier
pip install requirements.txt
python scripts/prepare_dataset.py --confirm
```

---

For license problem please contact me through my social media at [ducmanh.space](https://ducmanh.space/)
