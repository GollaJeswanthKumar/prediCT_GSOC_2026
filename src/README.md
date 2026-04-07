# PrediCT — Project 1: Heart Segmentation (COCA Dataset)

2D U-Net pipeline for GSoC ML4Sci: download → mask → preprocess → train → evaluate.

## Setup
pip install -r requirements.txt

## Run order
download_and_flatten.py -> Download COCA via azcopy; flatten nested DICOM folders 
generate_masks.py -> Run TotalSegmentator on 50 patients → `heart.nii.gz` 
preprocess.py -> HU window → resample → resize → save 2D .npy slices 
dataset.py -> Dataset/DataLoader (imported, not run directly) 
train.py -> Train U-Net, save best weights 
evaluate.py -> Dice scores, inference timing, visualisations 


**Preprocessing**
- HU window `[-100, 400]` covers soft tissue and calcium while suppressing lung
  air and bone extremes — standard for cardiac CT segmentation.
- Isotropic resampling to 1×1×1 mm ensures consistent voxel scale across
  scanners before spatial resize to 256×256.

**Augmentation**
- Horizontal/vertical flips + random 90° rotation are anatomy-safe for
  cardiac CT (heart is symmetric and rotation-invariant at coarse scale).
- Mild intensity jitter simulates scanner variation without distorting HU
  relationships.

**Model: 2D U-Net**
- Encoder-decoder with skip connections achieves >0.85 Dice on organ
  segmentation benchmarks consistently.
- 2D slice-by-slice inference is ~100× faster than TotalSegmentator's 3D
  nnU-Net and fits comfortably in 24 GB VRAM (GPU I have used for this task) at batch size 16.
- BatchNorm added for more stable training.

**Loss: Dice + BCE**
- BCE alone struggles with class imbalance (heart ≈ 10–20% of slice area).
  Adding Dice loss directly optimises the evaluation metric.


