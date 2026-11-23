# ISIC 2019 Dataset Instructions

This directory contains the ground truth and metadata CSVs for the **ISIC 2019 Challenge (BCN20000 & HAM10000 & MSK)**.

---

## 1. Metadata Files Included in this Repository

The following ground truth and patient metadata files are tracked in `data/metadata/`:
- `ISIC_2019_Training_GroundTruth.csv`: Ground truth diagnostic categories (MEL, NV, BCC, AK, BKL, DF, VASC, SCC) for all 25,331 training images.
- `ISIC_2019_Training_Metadata.csv`: Patient metadata including lesion IDs, approximate age, anatomical site, and sex.
- `ISIC_2019_Test_GroundTruth.csv`: Ground truth for the held-out test evaluation set.
- `ISIC_2019_Test_Metadata.csv`: Metadata for the held-out test evaluation set.

---

## 2. High-Resolution Image Archives (Large Files)

Due to GitHub's **100 MB per-file upload limit**, the raw image archives cannot be hosted directly inside Git:
* `ISIC_2019_Training_Input.zip`: **9.77 GB** (25,331 dermatoscopic JPEG images)
* `ISIC_2019_Test_Input.zip`: **3.82 GB** (8,238 test dermatoscopic JPEG images)

### How to Download the Images:
You can download the raw image ZIPs directly from the official ISIC Archive:
1. **Official ISIC Challenge 2019 Portal:**
   * URL: [https://challenge.isic-archive.com/data/#2019](https://challenge.isic-archive.com/data/#2019)
   * Training Images: Download `ISIC_2019_Training_Input.zip` (9.77 GB)
   * Test Images: Download `ISIC_2019_Test_Input.zip` (3.82 GB)
2. **Kaggle Mirror (Optional Fast Download):**
   * Kaggle dataset: `cdeotte/isic2019-256x256` or official ISIC 2019 dataset via Kaggle API:
     ```bash
     kaggle datasets download -d andrewmvd/isic-2019
     ```

### Local Directory Layout for Running Experiments:
Place the downloaded archives or extracted images into:
```
data/
├── metadata/
│   ├── ISIC_2019_Training_GroundTruth.csv
│   ├── ISIC_2019_Training_Metadata.csv
│   ├── ISIC_2019_Test_GroundTruth.csv
│   └── ISIC_2019_Test_Metadata.csv
├── ISIC_2019_Training_Input.zip (or extracted images/)
└── ISIC_2019_Test_Input.zip
```
