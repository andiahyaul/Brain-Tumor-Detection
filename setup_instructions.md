# Setup Instructions for Brain Tumor Detection Project

## Missing Files
Due to GitHub file size limitations, the following files are not included in this repository:

### Model Files (50MB total):
- `models/brain_tumor_detector_best.h5` (17MB)
- `models/brain_tumor_best.h5` (17MB)
- `models/brain_tumor_finetuned_best.h5` (17MB)

### Dataset (8.8MB):
- `Data/yes/` - 155 tumor MRI images
- `Data/no/` - 98 no tumor MRI images

### Processed Data (16MB):
- `processed_data/train_images.npz` (12MB)
- `processed_data/test_images.npz` (2MB)
- `processed_data/val_images.npz` (1.9MB)
- Various .npy label files

## How to Setup:

1. **Get the dataset**: Place MRI images in `Data/yes/` and `Data/no/` folders
2. **Run preprocessing**: Execute `notebooks/Data_Preprocessing.ipynb` to generate processed data files
3. **Train models**: Run `notebooks/Baseline_Train.ipynb` to train the models
4. **Create ensemble**: The ensemble configuration is saved as `models/brain_tumor_ensemble.pkl`

## Alternative: Use Pre-trained Models
If you have the trained models, place them in the `models/` directory with the exact filenames listed above.

The ensemble model will automatically load these files when instantiated.