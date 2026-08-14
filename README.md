# CS 4/510: Computer Vision & Deep Learning — Programming Assignment #3

**Author:** Veniamin Velikoretskikh (veniamin@pdx.edu)
**Course:** CS 4/510 — Portland State University

Transfer learning assignment: uses a pretrained InceptionResNetV2 (trained on
ImageNet) as a frozen feature extractor for a cats-vs-dogs binary classifier,
then compares training a head on the full network vs. on a smaller sub-network
cut partway through.

## What it does

**Step 1 — Inspect the pretrained model.** Loads InceptionResNetV2 with
ImageNet weights (no top layers), prints the full architecture, counts its
trainable parameters, and visualizes the 32 filters of the first conv layer.

**Step 2 — Load the data.** Loads the cats/dogs training and test sets from
disk with `image_dataset_from_directory` (labels inferred from folder names),
resizes everything to 150x150, and preprocesses images for InceptionResNetV2.

**Step 3 — Build the classifier.** Stacks a new head (Flatten → Dense(256,
relu) → Dense(1, sigmoid)) on top of the pretrained base, then freezes the
base so only the new head's weights train.

**Step 4(i) — Baseline (untrained head).** Runs the untrained model over the
test set to get a baseline accuracy (expected ~50%, since the head is still
randomly initialized) and plots a confusion matrix.

**Step 4(ii) — Train the head.** Trains the head for 10 epochs on top of the
frozen full InceptionResNetV2 base, printing per-epoch train/test loss, then
re-evaluates test accuracy and confusion matrix.

**Step 4(iii) — Sub-network comparison.** Repeats the same setup but cuts the
pretrained network off early (at `block35_5_ac`, the end of the 5th
Inception-ResNet-A block) instead of using the full network, trains a fresh
head on top of that smaller sub-network, and compares its accuracy to the
full-network version.

## Files

- `CS410_Computer_vision_and_deep_learning_assignment_3.py` — main script
- `CS410 computer vision and deep learning assignment 3.pdf` — writeup
- `step1_first_layer_filters.png` — first-layer filter visualization
- `step4i_confusion_matrix.png` — confusion matrix, untrained head
- `step4ii_confusion_matrix.png` — confusion matrix, trained head (full network)
- `step4iii_confusion_matrix.png` — confusion matrix, trained head (sub-network)

## Requirements

```
numpy
tensorflow
keras
matplotlib
scikit-learn
```

## Data setup

The script expects a `cats_dogs_dataset/dataset/` folder next to it, structured as:

```
cats_dogs_dataset/dataset/
├── training_set/
│   ├── cats/
│   └── dogs/
└── test_set/
    ├── cats/
    └── dogs/
```

## Usage

```bash
python CS410_Computer_vision_and_deep_learning_assignment_3.py
```

Runs all four steps end-to-end: loads InceptionResNetV2, visualizes its first
layer, loads the dataset, evaluates the untrained head, trains it for 10
epochs, evaluates again, then repeats training/evaluation on the sub-network
cutoff. Saves all output images to the working directory. A GPU is strongly
recommended — this trains two separate models over the full dataset.

## Notes

- The pretrained base is frozen in both runs (`trainable = False`), so only
  the new head's ~few hundred thousand parameters are updated — not
  InceptionResNetV2's ~54M weights.
- The sub-network cutoff (`block35_5_ac`) was chosen as a clean architectural
  boundary — the end of a full Inception-ResNet-A block — rather than an
  arbitrary middle layer.
