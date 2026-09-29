# Project 11 — Visual Waste Sorting Assistant

## 1. Problem Description

Computer Vision (CV) and Convolutional Neural Networks (CNNs) have revolutionized autonomous visual perception within Artificial Intelligence (AI). CNNs apply parameterized spatial convolution kernels, non-linear activations (ReLU), and spatial pooling layers to extract hierarchical invariant features from raw pixel tensors, outperforming hand-crafted image descriptors on object recognition benchmarks.

This project builds an automated visual classifier to support waste sorting at campus recycling stations. The model must categorize photographed discarded items into one of six canonical categories from the benchmark TrashNet dataset:
$$\mathcal{Y} = \{\text{cardboard}, \text{glass}, \text{metal}, \text{paper}, \text{plastic}, \text{trash}\}$$

The visual domain presents substantial challenges: specular reflections on glass, crumpling deformability of plastic and paper, and variable illumination conditions. Students must design and train a CNN architecture from first principles using PyTorch or TensorFlow/Keras, establish naive baselines, analyze training/validation learning curves to diagnose overfitting, and benchmark performance across degraded and challenging test images.

---

## 2. Provided Materials & Starter Resources

The project repository supplies an offline smoke-test image suite, a dataset staging script for the full TrashNet corpus, license documentation, and a Streamlit upload dashboard.

### File Structure
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

### Starter Infrastructure vs. Student Implementation
`scripts/prepare_dataset.py` handles optional downloading and partitioning of the full 2,527-image TrashNet corpus into train, validation, and test splits (git-ignored). `starter/app.py` provides the upload and confidence-visualization frontend. Students implement the neural network pipeline in `starter/student_core.py`:

- **`build_model(input_shape, num_classes)`** — defines the CNN architecture (convolutional, pooling, dropout, and fully connected layers) terminating in a Softmax output.
- **`train(model, train_dir, validation_dir, epochs)`** — runs the training loop with categorical cross-entropy loss and an adaptive optimizer (Adam or SGD with momentum).
- **`predict(model, image_path)`** — returns the predicted class label and confidence probability for a single image.

Students must build and train their own CNN. Shipping downloaded pre-trained solution weights (`.h5`, `.pt`, `.ckpt`, `.pkl`) or calling hosted vision/LLM APIs is strictly prohibited.

### Installation and Execution
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run starter/app.py
```

---

## 3. Requirements & Deliverables

### 3.1 Midterm Milestone — AI Core
- **Algorithmic Implementation**: Construct a CNN from first principles in `starter/student_core.py` using PyTorch or TensorFlow/Keras. Implement `build_model`, `train`, and `predict` as specified above. The architecture must include at least one convolutional block, pooling, dropout regularization, and a final Softmax classification head.
- **Mandatory Controlled Experiment**: Conduct an architectural ablation varying at least one design parameter: network depth, convolutional filter count, dropout probability (e.g., 0.2 vs. 0.5), data augmentation strategy (rotations, flips, zoom), or initial learning rate. Plot training and validation loss/accuracy curves across epochs, analyze overfitting onset, and report macro-averaged metrics on the held-out test split. Compare against a naive baseline (majority-class or color-histogram nearest-centroid classifier).
- **Deliverables**: Implemented `student_core.py`, training scripts or notebooks, and `presentation.pdf`.
- **Grading**: CNN architecture and training loop correctness (15 pts); ablation rigor, learning curves, and overfitting mitigation (10 pts); oral defense with live image classification demo (15 pts). **Total: 40 pts.**

### 3.2 Final Milestone — AI Product
- **Interactive Application**: Deliver a Streamlit application that enables users to: upload arbitrary JPG/PNG images of waste items; view instant visual previews; display the predicted TrashNet class with per-class confidence bars; surface educational disposal guidance per category; and highlight known classification failure modes.
- **Stress & Edge Scenarios**: Demonstrate live robustness across four required visual conditions: standard in-distribution images, complex or deformed items, low-light or blurred captures, and novel items photographed directly on campus (photos taken by the team count).
- **Deliverables**: Functional Streamlit application, clean GitHub repository, and `report.pdf` (IEEE format).
- **Grading**: Application interface, upload ergonomics, and inference responsiveness (25 pts); report mathematical rigor, convolutional derivations, and ablation findings (15 pts); oral defense against novel test images (10 pts). **Total: 50 pts.**

---

## 4. References

- [1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," *Proceedings of the IEEE*, vol. 86, no. 11, pp. 2278–2324, 1998, doi: 10.1109/5.726791.
- [2] G. Thung and M. Yang, "Classification of trash for recyclability status," *Stanford University Technical Report*, CS229 Machine Learning Final Projects, 2016.
- [3] S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. Hoboken, NJ, USA: Pearson, 2020, pp. 750–788.
