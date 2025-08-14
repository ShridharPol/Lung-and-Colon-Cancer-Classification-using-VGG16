# Lung and Colon Cancer Classification with VGG16 & pHash Filtering

## Overview

This project implements a deep learning pipeline using **VGG16** to classify histopathological images of lung and colon tissues into their respective subtypes.
We worked with the [LC25000 dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images), which contains augmented images generated from a relatively small set of originals.
To reduce potential **data leakage** from near-duplicate images across splits, we applied **Perceptual Hashing (pHash)** to detect and remove visually similar images.

**Problem:** The LC25000 dataset contains augmented images with near-duplicates, risking data leakage and overestimated accuracy.  
**Solution:** Applied Perceptual Hashing (pHash) to remove visually similar images before training a VGG16 classifier.  
**Outcome:** Reduced data leakage risk and achieved more reliable performance metrics (accuracy: ~98% on a truly independent test set).

---

## Why pHash Filtering?

* **Problem:** LC25000’s augmentation strategy can cause identical or near-identical images to appear in both training and evaluation sets.
* **Risk:** This can lead to inflated accuracy due to memorization rather than true generalization.
* **Our approach:**

  * Compute a **256-bit pHash** for each image.
  * Identify and remove images with matching hashes across train/val/test splits.
  * Ensure that no near-duplicate exists between different data splits.

> We note that pHash filtering reduces — but does not completely eliminate — the effects of dataset augmentation. Overfitting is still possible due to limited unique originals.

---

## Dataset

* **Source:** [LC25000 dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
* **Classes:**

  1. Lung Adenocarcinoma
  2. Lung Benign Tissue
  3. Lung Squamous Cell Carcinoma
  4. Colon Adenocarcinoma
  5. Colon Benign Tissue
* **Image Format:** `.jpeg`
* **Labels:** Derived from folder structure (integer-encoded per class).
* **Resolution:** Resized to `224×224` before feeding into the network.

---

## Methodology

### Data Preparation

* Scan image folders to get file paths and class labels.
* Compute **pHash** for each image.
* Filter out duplicates across train/val/test sets.
* Resize and normalize images to `[0,1]` range.
* Apply standard data augmentation (rotation, flips, zoom, etc.).

### Model Architecture

* **Base model:** VGG16 (`imagenet` weights, without top layers).
* **Custom classifier head:**

  * Global Average Pooling
  * Dense(256, ReLU) + Dropout
  * Dense(#classes, Softmax)
* **Loss:** Categorical Crossentropy
* **Optimizer:** Adam

### Training Procedure

* **Phase 1:** Freeze all VGG16 convolutional layers, train only the top layers.
* **Phase 2:** Unfreeze top convolutional blocks and fine-tune with a reduced learning rate.
* **Callbacks:** EarlyStopping, ModelCheckpoint.

---

## Results

*(Replace with your actual metrics)*

| Metric   | Train | Val  | Test |
| -------- | ----- | ---- | ---- |
| Accuracy | 0.9972|0.9969| 0.984|
| Loss     | 0.0365|0.0383|0.0353|

> Despite pHash filtering, some overfitting is observed — likely due to the dataset’s limited number of unique original images.

---

## Key Takeaways

* pHash filtering helps reduce data leakage in LC25000, improving reliability of evaluation.
* The dataset’s inherent limitations mean results should be interpreted cautiously.
* For robust clinical AI, larger and more diverse datasets are needed.

---

## Reproducing this Work

This project can be reproduced entirely on **Kaggle** in a few clicks.

### Steps:
1. **Go to Kaggle** and create a new Notebook.
2. **Upload** the `LC25000-vgg16-classification-phash.ipynb` file.
3. In the right-hand panel, under **"Add Data"**, search for  
   **"LC25000 histopathological images"** by *andrewmvd* and add it as an input dataset.
4. **Run All Cells** — the notebook will:
   - Compute perceptual hashes (pHash) to check for near-duplicate images.
   - Prepare train/validation/test splits.
   - Train the VGG16 model.
   - Output accuracy, loss, and sample predictions.

> No manual data downloads or preprocessing are required when running on Kaggle.

---

## References

* [LC25000 Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
* [pHash Official Site](http://www.phash.org/)
* [VGG16 Paper](https://arxiv.org/abs/1409.1556)
