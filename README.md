# ODIR-5K Normal vs. Disease Classification

Binary triage classifier (`0 = Normal`, `1 = Disease`) for retinal fundus photographs,
fine-tuned from a pretrained CNN backbone (ResNet50 / EfficientNet-B0 / DenseNet121),
built for automated screening triage — flagging images for clinician review rather
than standalone diagnosis.

## What's in the notebook

| Stage | What it does |
|---|---|
| Labeling | Builds binary labels from `full_df.csv` where available, falls back to patient-level `data.xlsx` metadata otherwise |
| Preprocessing | CLAHE contrast enhancement (fundus-specific), ImageNet normalization |
| Splitting | Patient-level, leakage-checked **5-fold `StratifiedGroupKFold`** — no patient's left/right eye ever appears in both train and val within a fold |
| Augmentation | Random crop, flip, rotation, color jitter, `RandomErasing` |
| Class balancing | `WeightedRandomSampler` on the minority class |
| Model | Pretrained backbone + custom binary head |
| Training | **Gradual, layer-by-layer backbone unfreezing** with discriminative learning rates, `ReduceLROnPlateau`, and early stopping — see "Training strategy" below |
| Metrics | Out-of-fold (OOF) ROC-AUC (primary), sensitivity, specificity, F1, confusion matrix, ROC curve |
| Explainability | Grad-CAM on the best-performing fold's checkpoint |
| Error analysis | Most confidently-wrong OOF predictions, visualized |
| Submission | 5-fold checkpoint ensemble → `submission.csv` on the unlabeled Testing Images |

## Setup

```bash
pip install -r requirements.txt
```

GPU strongly recommended — training runs 5 independent folds, each up to
`TOTAL_EPOCHS` (with early stopping usually cutting this shorter).

## Data setup

The dataset (~2GB: `full_df.csv`, `preprocessed_images/`, and an `ODIR-5K/` subfolder
with `Training Images/`, `Testing Images/`, `data.xlsx`) is **not committed to this repo**
— it's pulled from Kaggle at run time via `kagglehub` and cached locally, so it only
downloads once.

Source: [andrewmvd/ocular-disease-recognition-odir5k](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)

**One-time Kaggle auth setup, before running the notebook:**
1. Create a free Kaggle account if you don't have one.
2. Go to Kaggle → Settings → API → "Create New Token" — downloads `kaggle.json`.
3. Place it at `~/.kaggle/kaggle.json` (Linux/Mac) or `%USERPROFILE%\.kaggle\kaggle.json`
   (Windows) — or set the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables instead.

The **"Data setup"** cell near the top of the notebook then handles the rest:
downloads (or reuses the cached copy of) the dataset, auto-detects the actual folder
layout, and sets `DATA_ROOT` — every path used later in the notebook
(`PREPROC_DIR`, `TRAINING_DIR`, `TESTING_DIR`, `FULL_DF_PATH`, `PATIENT_META_PATH`) is
built from `DATA_ROOT`, so nothing needs to be hardcoded or edited manually. If your
local copy is laid out differently, override the individual path variables directly in
the config cell.

## Training strategy (why it's structured this way)

An earlier version of this pipeline unfroze the entire backbone at once after a short
head warm-up. That reliably overfit: training loss fell toward zero while validation
loss climbed for the back half of every fold's run, with the best checkpoint typically
landing well before the fixed epoch budget was used up.

The current version instead:

1. **Freezes the whole backbone** and trains only the classification head for
   `FREEZE_EPOCHS` epochs.
2. **Unfreezes one backbone group at a time**, output-side first (the group closest
   to the head opens first, the input stem opens last), every
   `EPOCHS_PER_UNFREEZE_GROUP` epochs.
3. Gives each newly-opened group a **smaller learning rate than the previous one**
   (`UNFREEZE_LR_DECAY` per step) — deeper, more task-specific layers move faster;
   shallow, generic layers move gently.
4. **Overrides only the stem's LR** (`FINAL_STAGE_LR`) once it opens in the last
   stage, since low-level filters (edge/color detectors) rarely benefit from
   aggressive updates — already-open groups keep the rate they'd earned via the
   normal decay, undisturbed.
5. Runs **`ReduceLROnPlateau`** (on val AUC) within each stage, and **early-stops**
   a fold if val AUC hasn't improved in `EARLY_STOP_PATIENCE` epochs.

Key config knobs (all near the top of the notebook):

```python
N_UNFREEZE_GROUPS = 4          # how many backbone chunks to stage the unfreeze across
EPOCHS_PER_UNFREEZE_GROUP = 3  # epochs between each unfreeze event
UNFREEZE_LR_DECAY = 0.5        # per-group LR decay (excludes the stem override)
FINAL_STAGE_LR = 5e-5          # explicit LR for the stem once it unfreezes
EARLY_STOP_PATIENCE = 6        # epochs without improvement before a fold stops
WEIGHT_DECAY = 3e-4
HEAD_DROPOUT = 0.4
```

If the val-AUC-per-epoch plot (printed at the end of training) still shows a drop
right when the stem group unfreezes, the simplest next step is
`N_UNFREEZE_GROUPS = 3` — i.e. never unfreeze the stem at all.

## Outputs

- `checkpoints/{backbone}_fold{i}.pt` — best checkpoint per fold (state_dict + val AUC + epoch)
- `submission.csv` — ensembled predictions (`filename`, `predicted_probability`, `predicted_label`) on the unlabeled Testing Images

## 📊 Model Performance Evaluation

### Out-of-Fold (5-Fold Cross-Validation) Results

| Metric | Overall (OOF) | Per-Fold Mean ± Std |
| :--- | :---: | :---: |
| **ROC-AUC (Primary Metric)** | **0.8218** | **0.8247 ± 0.0133** |
| **F1 Score** | **0.7605** | — |
| **Accuracy** | **0.7473** | — |
| **Sensitivity (Disease Recall)** | **0.7112** | — |
| **Specificity (Normal Recall)** | **0.7939** | — |

---

### Detailed Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | 0.68 | 0.79 | 0.73 | 3,052 |
| **Disease** | 0.82 | 0.71 | 0.76 | 3,948 |
| **Macro Average** | **0.75** | **0.75** | **0.75** | **7,000** |
| **Weighted Average** | **0.76** | **0.75** | **0.75** | **7,000** |

---

### Visual Diagnostic Evaluation

> **Left:** Out-of-Fold Confusion Matrix evaluated at decision threshold $t = 0.5$ ($N = 7,000$).  
> **Right:** Overall Out-of-Fold Receiver Operating Characteristic (ROC) curve yielding an AUC of **0.822**.

## Known limitations

- Metrics are out-of-fold on ODIR-5K only — there's no external, independently-labeled
  test set, so these numbers describe generalization within this dataset's distribution,
  not necessarily to other cameras, populations, or imaging conditions.
- Binary Normal/Disease collapses ODIR-5K's original multi-label diagnoses; a positive
  prediction doesn't indicate *which* disease is present.
- This is a screening/triage aid, not a diagnostic tool — sensitivity at the chosen
  operating threshold should be reported and weighed against the cost of missed disease
  in any real deployment discussion.
