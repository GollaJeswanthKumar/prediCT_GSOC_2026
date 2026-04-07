# PrediCT - ML4SCI

## CAC Segmentation Pipeline

---

## Overview

This project focuses on building an end-to-end deep learning pipeline for segmenting cardiac structures from non-contrast CT (NCCT) scans, as a foundational step toward Coronary Artery Calcium (CAC) detection.

CAC is a critical biomarker for cardiovascular risk assessment, and automating its segmentation enables scalable and reproducible clinical analysis.

This work is part of the **PrediCT project under ML4SCI**.

---

## Problem Statement

CAC segmentation is challenging due to:

* Small, scattered high-intensity calcium regions
* Severe class imbalance (very few positive pixels)
* False positives from bones (ribs, vertebrae)
* Noise and variability in CT imaging

The goal is to design a **robust, efficient, and reproducible segmentation pipeline**.

---

##  Workflow

DICOM CT Scans
→ HU Conversion
→ Preprocessing (Windowing, Normalization, Resampling)
→ Heart Mask Generation (TotalSegmentator)
→ Slice-wise Dataset Creation
→ U-Net Training
→ Evaluation (Dice Score + Visualization)

---

## Current Progress

* DICOM loading and HU conversion implemented
* Preprocessing pipeline (windowing, normalization, resampling)
* Heart mask generation using TotalSegmentator (~50 patients)
* Optimized data pipeline using precomputed `.npy` slices
* Fast DataLoader with local storage (Colab optimization)
* 2D U-Net baseline implemented
* Training and evaluation pipeline completed

---

## Current Results

* Dice Score: **~0.37 (early training stage)**
* Updated Dice Score : **~0.484 (After training 2DU-net upon 30 Epochs)**
* Updated Dice Score : **~0.47 (After training 2.5DU-net upon 30 Epochs)**
* Model is learning meaningful regions but not yet converged

---

## Ongoing Improvements

* So for 2.5D unet I used [slice-1,slice,slice+1] slices so that It will learn neighbouring slices context as well
* Used K = 3 and Epochs = 30 
* May be 3 slices and 30 epochs are not sufficient so thinking to increase K = 5 and Epochs = 40 or 50

---

## Technologies Used

* Python
* PyTorch
* NumPy, Pandas
* pydicom, SimpleITK
* TotalSegmentator
* Google Colab (GPU)

---

## Dataset

* Stanford COCA Dataset
  (Cardiac CT scans with CAC annotations)

---

## Future Work

* Achieve Dice score > 0.80
* Implement CAC segmentation (beyond heart region)
* Compute Agatston score for clinical relevance
* Compare multiple segmentation architectures

---

## Key Highlights

* Efficient preprocessing pipeline (no repeated computation)
* GPU-optimized training workflow
* Clean separation of preprocessing and training stages
* Reproducible and scalable design

---
