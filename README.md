# Temporal Feature Ablation Platform

Artefact for the CIS4517 MSc dissertation *Evaluating the Incremental Predictive Value of Behavioural Telemetry Beyond Assessment Performance for Identifying At-Risk Learners in MOOCs* (Beryl Odingo, 26673207).

The artefact has two parts:

1. **Analysis pipeline**: `Ch2_aligned_rerun.ipynb` (Google Colab). It builds leakage-free feature blocks from MOOCCubeX, runs the block-wise ablation, statistical tests, SHAP attribution and interrupted time series, and exports every results table.
2. **Web platform**: `app.py` (Streamlit, pandas, Altair). It presents the final results and lets users apply the redundancy framework to their own results.

## Research question

Once assessment evidence exists, does behavioural telemetry (video: B1; discussion: B2) still add predictive value for identifying learners with low Q4 correctness? If not, from which checkpoint does it become redundant?

## Folder contents

| File / folder | What it is |
|---|---|
| `app.py`, `requirements.txt` | Streamlit web platform |
| `data/` | Final results read by the app (exported from `results_tables.xlsx`) |
| `Ch2_aligned_rerun.ipynb` | Final analysis notebook, saved with its outputs |
| `results_tables.xlsx` | Every results table (one sheet per table), exported by the notebook |
| `figures/` | SHAP summary plots, SHAP block shares, ITS plots and residual diagnostics |
| `docs/data_dictionary.md` | Definitions of every feature, the target and the row counts |
| `docs/decision_log.md` | Key design decisions and the reasons for them |
| `archive/` | Earlier exploratory notebooks and results (superseded; kept for transparency) |

## Web platform

### What it does

- **Overview:** the seven-configuration performance matrix (A, B1, B2, B1+B2, A+B1, A+B2, A+B1+B2) at checkpoints Q1–Q3.
- **Incremental value:** gains over Model A in ROC-AUC, PR-AUC, F1 and recall, from the hold-out set (with bootstrap 95% CIs and DeLong p-values) and from time-aware five-fold cross-validation. Each gain is classified with the framework bands: < 0.01 negligible, 0.01–0.02 marginal, 0.02–0.05 meaningful, ≥ 0.05 strong.
- **Attribution:** block-level SHAP shares (XGBoost and LightGBM) and behavioural feature-level shares.
- **Apply the framework:** upload a results CSV with the columns `checkpoint, model, roc_auc, pr_auc`, including a baseline row `model = A` for each checkpoint. The platform calculates the incremental gains and classifies each configuration. This is the reusable redundancy framework described in the dissertation.

### Run locally

    pip install -r requirements.txt
    streamlit run app.py

Then open http://localhost:8501.

### Deploy (public link for the dissertation)

1. Push this folder to a public GitHub repository.
2. At https://share.streamlit.io, choose **Create app**, select the repository and `app.py`, and deploy.
3. Put the public URL in Appendix C of the dissertation.

## Analysis pipeline

### How to run

1. Open `Ch2_aligned_rerun.ipynb` in Google Colab (High-RAM runtime if available).
2. Set `SMOKE_TEST = True` and choose Run all. This checks the environment on synthetic data in about 5 minutes.
3. Set `SMOKE_TEST = False`, check `DATA_DIR` and the four file names, and run all cells again. The full run takes roughly 1.5–3 hours.
4. Outputs are written to `MyDrive/ch2_rerun_outputs/`.

### Inputs (not included)

The four MOOCCubeX files are `user-problem.parquet`, `user-video.parquet`, `translated_comment.parquet` and `translated_reply.parquet`. MOOCCubeX (Yu et al., 2021) was released by its authors for academic research. Download it from the official source. The raw data are not redistributed here because of their size and licence.

### Method summary

- **Checkpoints:** each learner's assessment events are split into quarters by event count. Features are cumulative to the learner's cut-off at Q1, Q2 and Q3. Q4 is used only for the target (Q4 correctness < 0.60).
- **Blocks:**
  - A: eight assessment features, including current-quarter correctness and trend;
  - B1: eight video-engagement constructs;
  - B2: five discussion counts.
- **Modelling:** chronological 80/20 split (learners ordered by Q1 cut-off time), with z-score scaling for A, min–max scaling for B, and SMOTE fitted on training data only. The models are XGBoost, LightGBM and Logistic Regression.
- **Inference:** paired bootstrap (1,000 resamples), DeLong tests, and time-aware five-fold cross-validation.
- **Redundancy rule:** no gain of 0.02 or more in ROC-AUC, PR-AUC, F1 or recall, and a behavioural SHAP share below 5%.
- **ITS:** segmented OLS with HAC standard errors across 21 positions relative to each learner's first assessment.

### Output-to-dissertation map

| Sheet in `results_tables.xlsx` | Dissertation |
|---|---|
| `T1_verification_tests_stage1` | Table 9 |
| `T_data_summary` | Table 10 |
| `T5_holdout_xgboost` | Table 11 |
| `T_gains_holdout_stats` | Table 12 |
| `T6_cv_incremental_gains` | Table 13 |
| `T7_ensemble_verification` | Table 14 |
| `T8_shap_block_shares` | Table 15, Figure 4 |
| `T8b_shap_feature_level` | Table 16 |
| `T9_its_regressions`, `T9b_its_diagnostics` | Table 17, Figure 2 |
| `T_redundancy_decision` | Table 18 |
| `T_cv_fold_prevalence` | Table F1 (Appendix F) |

### Verification

Automated tests T1–T7 run inside the notebook: row counts, preservation of rows through the joins, no missing values, temporal inclusion of discussion events, no post-cut-off predictors, and one row per learner per checkpoint. All passed.

## Environment

Python 3 on Google Colab. Polars 1.35.2, pandas 2.2.3, NumPy 2.1.3, scikit-learn 1.6.1, XGBoost 3.4.1, LightGBM 4.6.0, SHAP 0.52.0, imbalanced-learn 0.14.2, statsmodels 0.15.0. Random seed: 42.

## Data handling and ethics

Only pseudonymised secondary data were used, and no learner was re-identified. Comment and reply text was never processed; only timestamps and counts were extracted. All outputs are aggregate statistics. The project received departmental ethical approval (dissertation Appendix E).
