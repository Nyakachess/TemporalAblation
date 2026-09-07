# Temporal Feature Ablation Platform

Web-based temporal feature ablation and redundancy-evaluation platform for the
CIS4517 MSc dissertation (Beryl Odingo, 26673207). Implemented in Python with
Streamlit, pandas and Altair.

## What it does

- Presents the canonical FINAL results of the block-wise study (Blocks A, B1, B2)
  across checkpoints Q1-Q3: seven-model performance matrix, ten-seed incremental
  gains with the 0.01/0.02 framework bands, and three-block SHAP attribution.
- **Apply the framework** tab: upload any results CSV (columns `checkpoint,
  model, roc_auc, pr_auc` with a baseline row `model = A` per checkpoint) and the
  platform computes incremental gains and classifies each configuration as
  negligible / marginal / meaningful / strong - the reusable redundancy
  framework described in the dissertation.

## New in v4

- **Flexible outcomes**: the outcome no longer has to be a 0/1 column. Pick any
  column: a numeric score gets a threshold ("at risk when below 60"), a letter
  grade or pass/fail label gets a picker for which values count as at risk
  (D and F, or Fail/Refer). The app derives the binary flag and reports the
  class balance.
- **Categorical features handled**: Yes/No columns become 1/0 and categories
  (Gender, Department, income level, ...) are one-hot encoded automatically -
  no data preparation needed.
- **Privacy guard**: name, email, ID and date columns are never offered as
  features.
- Works on cross-sectional datasets (no checkpoint column) as well as
  checkpointed ones.

## New in v3

- **Date checkpoints**: if the checkpoint column holds dates (or unix
  timestamps) instead of labels like q1/q2/q3, the app detects this and bins
  them into 2-6 equal-sized, time-ordered periods (P1, P2, ...) of your choice.
- **Plain-language verdicts**: after a run, the app states in one sentence per
  stream when behavioural data becomes redundant (e.g. "becomes redundant from
  Q2 onward - continued collection after that point is not justified").
- **Learners likely to struggle**: the app scores every learner's risk with the
  full model, bands them (low / watch / elevated / high), shows the 20
  highest-risk learners at the latest checkpoint, and offers the full ranked
  list as a download. Scores are decision-support estimates, not judgements.

## Train models inside the app

The **Run ablation** tab trains models in the app itself: upload a modelling
dataset CSV (e.g. `full_checkpoint_model_data_with_b2_FINAL.csv`), confirm the
auto-detected outcome column, checkpoint column and block assignment, and click
**Run ablation**. The platform trains the canonical XGBoost configuration for
every block combination at every checkpoint, then classifies the incremental
gains with the redundancy framework. Timestamp, identifier and `q4_*` columns
are excluded from features automatically to prevent outcome leakage.

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

Then open http://localhost:8501.

## Use the latest pipeline outputs (optional)

The app ships with the canonical FINAL results embedded. To have it read live
pipeline outputs instead, copy these files from Google Drive into `./data/`:

    FINAL_sevenmodel_xgb_results.csv
    FINAL_multiseed_gain_summary.csv
    FINAL_threeblock_shap_blocks.csv

## Deploy free (for the dissertation link)

1. Push this folder to a public GitHub repository.
2. At https://share.streamlit.io choose "Create app", select the repo and
   `app.py`, and deploy. The public URL goes in the dissertation appendix.

## Evidence for Chapter 4

Screenshot: (1) the Overview tab, (2) the Incremental value tab showing the
threshold bands, (3) the Apply-the-framework tab with a classified upload.
# TemporalAblation
# TemporalAblation
