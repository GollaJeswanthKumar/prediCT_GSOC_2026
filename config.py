import os

## Paths
DATA_ROOT = "/media/mohan/Data/Jeswanth/dataset/patients/patient"

# Where TotalSegmentator heart masks will be saved (one sub-folder per patient)
MASK_ROOT = "/media/mohan/Data/Jeswanth/dataset/heart_masks"

# Where preprocessed 2D .npy slices will be saved
SAVE_ROOT = "/media/mohan/Data/Jeswanth/dataset/preprocessed"

# Where trained model weights will be saved
MODEL_SAVE_PATH = "/media/mohan/Data/Jeswanth/dataset/checkpoints/unet_best.pth"

## TotalSegmentator

TOTALSEG_LICENSE = "aca_B96AWVHHUFROIU"   
TARGET_MASK_COUNT = 50          

## Preprocessing 

HU_MIN = -100        # cardiac soft-tissue window lower bound
HU_MAX = 400         # cardiac soft-tissue window upper bound
NEW_SPACING = (1, 1, 1)          # isotropic resampling (mm)
TARGET_SIZE = (256, 256)         # spatial resize per slice

## Split ratios for train/val/test sets

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
RANDOM_SEED = 42

## Dataloader parameters

BATCH_SIZE  = 32  
NUM_WORKERS = 8
EPOCHS      = 30
LR          = 1e-3
