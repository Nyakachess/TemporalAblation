"""
Temporal Feature Ablation Platform
==================================
Web-based temporal feature ablation and redundancy-evaluation platform for the
MSc dissertation:

  "Evaluating the Incremental Predictive Value of Video-Based and
   Discussion-Based Behavioural Telemetry Beyond Assessment Performance
   for Identifying At-Risk Learners in MOOCs"

Beryl Odingo - 26673207 - CIS4517, Edge Hill University.

The platform ships with the study's canonical FINAL results embedded, so it
runs with no data files. If CSVs exported by the pipeline are placed in ./data
(FINAL_sevenmodel_xgb_results.csv, FINAL_multiseed_gain_summary.csv,
FINAL_threeblock_shap_blocks.csv) they are loaded instead, so the platform
always reflects the latest pipeline run.

Run locally:   streamlit run app.py
"""

import os
import io
import pandas as pd
import altair as alt
import streamlit as st

# ----------------------------------------------------------------------------
# Colour assignments (validated categorical palette, light surface)
# Colour follows the entity: each block keeps its hue everywhere in the app.
# ----------------------------------------------------------------------------
COLOURS = {
    "A": "#0072B2",        # assessment  - blue
    "B1": "#D55E00",       # video       - vermillion
    "B2": "#009E73",       # discussion  - green
}
INK = "#22252D"
MUTED = "#5D6370"

CHECKPOINTS = ["q1", "q2", "q3"]

# ----------------------------------------------------------------------------
# The study's redundancy framework (defined in Chapter 3 of the dissertation)
# ----------------------------------------------------------------------------
def classify_gain(gain: float) -> str:
    """Classify an incremental held-out gain against the pre-specified bands."""
    if gain >= 0.05:
        return "strong"
    if gain >= 0.02:
        return "meaningful"
    if gain >= 0.01:
        return "marginal"
    return "negligible"


BAND_HELP = (
    "below 0.01 = negligible · 0.01-0.02 = marginal · "
    "0.02-0.05 = meaningful · 0.05+ = strong. "
    "A stream is treated as practically redundant at a checkpoint when its "
    "incremental ROC-AUC and PR-AUC gains over the assessment-only model "
    "are below 0.02."
)

# ----------------------------------------------------------------------------
# Canonical FINAL results (embedded fallback; ./data CSVs override these)
# ----------------------------------------------------------------------------
_SEVEN_MODEL_CSV = """checkpoint,model,accuracy,precision,recall,f1,roc_auc,pr_auc
q1,A,0.750,0.359,0.816,0.499,0.852,0.484
q1,B1,0.638,0.224,0.562,0.321,0.646,0.260
q1,B2,0.233,0.156,0.916,0.267,0.513,0.156
q1,B1+B2,0.639,0.224,0.559,0.320,0.647,0.258
q1,A+B1,0.750,0.359,0.818,0.499,0.854,0.488
q1,A+B2,0.747,0.356,0.819,0.496,0.852,0.484
q1,A+B1+B2,0.753,0.362,0.817,0.501,0.854,0.487
q2,A,0.784,0.414,0.849,0.557,0.882,0.578
q2,B1,0.628,0.237,0.597,0.339,0.655,0.271
q2,B2,0.262,0.167,0.905,0.282,0.524,0.168
q2,B1+B2,0.626,0.235,0.596,0.338,0.657,0.276
q2,A+B1,0.786,0.417,0.851,0.560,0.882,0.580
q2,A+B2,0.785,0.415,0.847,0.557,0.882,0.578
q2,A+B1+B2,0.785,0.416,0.852,0.559,0.882,0.579
q3,A,0.796,0.434,0.877,0.581,0.901,0.641
q3,B1,0.624,0.233,0.578,0.332,0.639,0.255
q3,B2,0.272,0.169,0.895,0.284,0.525,0.169
q3,B1+B2,0.622,0.234,0.587,0.334,0.645,0.259
q3,A+B1,0.798,0.437,0.874,0.583,0.901,0.641
q3,A+B2,0.797,0.435,0.874,0.581,0.901,0.640
q3,A+B1+B2,0.796,0.435,0.874,0.581,0.901,0.639
"""

_MULTISEED_CSV = """checkpoint,model,d_roc_mean,d_roc_sd,d_pr_mean,d_pr_sd
q1,A+B1,0.0016,0.0004,0.0041,0.0041
q1,A+B2,-0.0001,0.0002,-0.0001,0.0016
q1,A+B1+B2,0.0016,0.0004,0.0038,0.0038
q2,A+B1,0.0001,0.0003,-0.0001,0.0013
q2,A+B2,-0.0002,0.0001,-0.0005,0.0006
q2,A+B1+B2,-0.0001,0.0003,-0.0006,0.0016
q3,A+B1,-0.0001,0.0001,0.0001,0.0003
q3,A+B2,-0.0001,0.0002,-0.0002,0.0008
q3,A+B1+B2,-0.0002,0.0001,0.0001,0.0010
"""

_SHAP_CSV = """checkpoint,A_share,B1_share,B2_share
q1,0.8467,0.1470,0.0063
q2,0.9074,0.0904,0.0021
q3,0.9431,0.0543,0.0026
"""


@st.cache_data
def load_results():
    """Load pipeline CSVs from ./data when present, else the embedded canon."""
    def read(path, fallback):
        if os.path.exists(path):
            return pd.read_csv(path), True
        return pd.read_csv(io.StringIO(fallback)), False

    seven, s1 = read("data/FINAL_sevenmodel_xgb_results.csv", _SEVEN_MODEL_CSV)
    shap_b, s3 = read("data/FINAL_threeblock_shap_blocks.csv", _SHAP_CSV)

    # The multiseed summary CSV, when exported by pandas, has a two-row header;
    # the embedded fallback uses flat column names.
    if os.path.exists("data/FINAL_multiseed_gain_summary.csv"):
        raw = pd.read_csv("data/FINAL_multiseed_gain_summary.csv", header=[0, 1], index_col=[0, 1])
        seeds = raw.reset_index()
        seeds.columns = ["checkpoint", "model", "d_roc_mean", "d_roc_sd", "d_pr_mean", "d_pr_sd"]
        s2 = True
    else:
        seeds = pd.read_csv(io.StringIO(_MULTISEED_CSV))
        s2 = False

    return seven, seeds, shap_b, (s1 or s2 or s3)


def gains_vs_baseline(df: pd.DataFrame, baseline: str = "A") -> pd.DataFrame:
    """Compute incremental ROC/PR gains of every model over the baseline,
    per checkpoint. This is the core of the ablation framework."""
    rows = []
    for cp, cdf in df.groupby("checkpoint"):
        base = cdf[cdf["model"] == baseline]
        if base.empty:
            continue
        b_roc = float(base["roc_auc"].iloc[0])
        b_pr = float(base["pr_auc"].iloc[0])
        for _, r in cdf[cdf["model"] != baseline].iterrows():
            d_roc = round(float(r["roc_auc"]) - b_roc, 4)
            d_pr = round(float(r["pr_auc"]) - b_pr, 4)
            rows.append({
                "checkpoint": cp, "model": r["model"],
                "d_roc_auc": d_roc, "d_pr_auc": d_pr,
                "verdict_roc": classify_gain(d_roc),
                "verdict_pr": classify_gain(d_pr),
                "redundant_at_checkpoint": (d_roc < 0.02) and (d_pr < 0.02),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# In-app model training (checkpoint-based prediction + ablation)
# ----------------------------------------------------------------------------
CANONICAL_XGB = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     eval_metric="logloss", random_state=42)

def is_excluded(col):
    """Columns that must never be model features: identifiers, personal data,
    timestamps, and q4_* columns (which would leak the outcome)."""
    c = col.lower()
    return (c.endswith("_id") or c == "id" or "_time" in c
            or c.startswith("q4_") or "date" in c
            or "name" in c or "email" in c)


YES_NO = {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}


def encode_features(df, feats):
    """Encode a feature selection for modelling: numeric columns pass through,
    Yes/No columns become 1/0, other categorical columns are one-hot encoded.
    Returns (X, mapping) where mapping[feature] lists its encoded columns."""
    parts, mapping = [], {}
    for f in feats:
        s = df[f]
        if pd.api.types.is_numeric_dtype(s):
            parts.append(pd.to_numeric(s, errors="coerce").rename(f))
            mapping[f] = [f]
        else:
            sl = s.astype(str).str.strip().str.lower()
            if set(sl.dropna().unique()) <= set(YES_NO):
                parts.append(sl.map(YES_NO).rename(f))
                mapping[f] = [f]
            else:
                dummies = pd.get_dummies(sl, prefix=f, dtype=int)
                parts.append(dummies)
                mapping[f] = list(dummies.columns)
    X = pd.concat(parts, axis=1).fillna(0)
    return X, mapping


def auto_assign_blocks(columns):
    """Suggest a block assignment from column names: assessment_/assess_ -> A,
    video_ -> B1, disc_/discussion_/behaviour_/behavior_/forum_ -> B2."""
    usable = [c for c in columns if not is_excluded(c)]
    return {
        "A": [c for c in usable if c.startswith(("assessment_", "assess_"))],
        "B1": [c for c in usable if c.startswith("video_")],
        "B2": [c for c in usable if c.startswith(
            ("disc_", "discussion_", "behaviour_", "behavior_", "forum_"))],
    }


def maybe_bin_dates(series, n_periods):
    """If the checkpoint column holds dates (or unix timestamps), bin it into
    n time-ordered periods P1..Pn of roughly equal size. Returns the binned
    labels, or None when the column is already categorical (e.g. q1/q2/q3)."""
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s) and s.dropna().abs().min() > 1e9:
        parsed = pd.to_datetime(s, unit="s", errors="coerce")   # unix seconds
    else:
        parsed = pd.to_datetime(s.astype(str), errors="coerce", format="mixed")
    if parsed.notna().mean() < 0.9 or parsed.nunique() <= 8:
        return None  # looks like labels, not dates
    ranked = parsed.rank(method="first")
    return pd.qcut(ranked, n_periods,
                   labels=[f"P{i+1}" for i in range(n_periods)]).astype(str)


def redundancy_summary(gains_df, checkpoints_order, block_labels):
    """Turn the gains table into plain-language redundancy statements: for each
    behavioural configuration, from which checkpoint onward it stays below the
    0.02 threshold on both metrics."""
    lines = []
    for cfg, label in block_labels.items():
        sub = gains_df[gains_df["model"] == cfg].set_index("checkpoint")
        if sub.empty:
            continue
        flags = [bool(sub.loc[cp, "redundant_at_checkpoint"])
                 for cp in checkpoints_order if cp in sub.index]
        cps = [cp for cp in checkpoints_order if cp in sub.index]
        if all(flags):
            lines.append(f"**{label}** adds no meaningful predictive value at "
                         f"any checkpoint measured - for this prediction task, "
                         f"collecting it is not justified by prediction gains "
                         f"at any point.")
        elif not any(flags):
            lines.append(f"**{label}** still adds meaningful value at every "
                         f"checkpoint - not redundant in this data.")
        else:
            # first checkpoint from which it stays redundant to the end
            onset = None
            for i in range(len(flags)):
                if all(flags[i:]):
                    onset = cps[i]
                    break
            if onset is not None:
                lines.append(f"**{label}** becomes redundant from "
                             f"**{str(onset).upper()}** onward - early "
                             f"collection may be justified, continued "
                             f"collection after that point is not.")
            else:
                lines.append(f"**{label}** shows a mixed pattern across "
                             f"checkpoints ({', '.join(str(c) for c in cps)}) "
                             f"- inspect the gains table before deciding.")
    return lines


def score_learners(df, target_col, cp_col, block_map, id_col):
    """Train the full model (all assigned blocks) per checkpoint and score
    every learner's risk. Metrics integrity note: models are trained on the
    80% train split; scores for training-split learners are in-sample and
    slightly optimistic, so the table marks which rows were held out."""
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    feats = [f for b in ["A", "B1", "B2"] for f in block_map.get(b, [])]
    checkpoints = (sorted(df[cp_col].dropna().unique()) if cp_col else ["all"])
    out = []
    for cp in checkpoints:
        cdf = df if cp == "all" else df[df[cp_col] == cp]
        y = cdf[target_col]
        X, _ = encode_features(cdf, feats)
        train_idx, test_idx = train_test_split(
            cdf.index, test_size=0.2, random_state=42, stratify=y)
        spw = (y.loc[train_idx] == 0).sum() / max((y.loc[train_idx] == 1).sum(), 1)
        m = XGBClassifier(scale_pos_weight=spw, **CANONICAL_XGB)
        m.fit(X.loc[train_idx], y.loc[train_idx])
        p = m.predict_proba(X)[:, 1]
        res = pd.DataFrame({
            "learner": (cdf[id_col].values if id_col else cdf.index),
            "checkpoint": str(cp),
            "risk_score": p.round(3),
            "held_out": cdf.index.isin(test_idx),
        })
        res["risk_band"] = pd.cut(res["risk_score"], [-0.01, 0.3, 0.5, 0.7, 1.01],
                                  labels=["low", "watch", "elevated", "high"])
        out.append(res)
    return pd.concat(out, ignore_index=True)


def build_configs(block_map):
    """All non-empty combinations of the provided blocks, in canonical order
    (A, B1, B2, B1+B2, A+B1, A+B2, A+B1+B2 when all three are present)."""
    from itertools import chain, combinations
    present = [b for b in ["A", "B1", "B2"] if block_map.get(b)]
    combos = chain.from_iterable(combinations(present, k)
                                 for k in range(1, len(present) + 1))
    configs = {}
    for combo in combos:
        name = "+".join(combo)
        feats = []
        for b in combo:
            feats += block_map[b]
        configs[name] = feats
    return configs


def run_ablation(df, target_col, cp_col, block_map, progress=None):
    """Train the canonical XGBoost configuration for every block combination
    at every checkpoint, on a stratified 80/20 split (seed 42), and return the
    performance table. This reproduces the dissertation's experimental design
    on any uploaded dataset."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, average_precision_score)
    from xgboost import XGBClassifier

    configs = build_configs(block_map)
    checkpoints = (sorted(df[cp_col].dropna().unique()) if cp_col else ["all"])
    total = len(checkpoints) * len(configs)
    done = 0
    rows = []
    for cp in checkpoints:
        cdf = df if cp == "all" else df[df[cp_col] == cp]
        y = cdf[target_col]
        if y.nunique() != 2:
            raise ValueError(
                f"Target '{target_col}' is not binary at checkpoint {cp} "
                f"(found {y.nunique()} distinct values).")
        all_feats = sorted({f for feats in configs.values() for f in feats})
        X, enc_map = encode_features(cdf, all_feats)
        train_idx, test_idx = train_test_split(
            cdf.index, test_size=0.2, random_state=42, stratify=y)
        ytr, yte = y.loc[train_idx], y.loc[test_idx]
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        for name, feats in configs.items():
            enc_cols = [c for f in feats for c in enc_map[f]]
            m = XGBClassifier(scale_pos_weight=spw, **CANONICAL_XGB)
            m.fit(X.loc[train_idx, enc_cols], ytr)
            p = m.predict_proba(X.loc[test_idx, enc_cols])[:, 1]
            pred = (p >= 0.5).astype(int)
            rows.append(dict(
                checkpoint=str(cp), model=name,
                accuracy=round(accuracy_score(yte, pred), 3),
                precision=round(precision_score(yte, pred, zero_division=0), 3),
                recall=round(recall_score(yte, pred), 3),
                f1=round(f1_score(yte, pred), 3),
                roc_auc=round(roc_auc_score(yte, p), 3),
                pr_auc=round(average_precision_score(yte, p), 3)))
            done += 1
            if progress is not None:
                progress.progress(done / total,
                                  text=f"Training {name} at {cp} ({done}/{total})")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Charts (Altair). One axis per chart; tooltips on every mark; text in ink.
# ----------------------------------------------------------------------------
def block_colour_scale(models):
    """Map each configuration to the hue of its dominant block."""
    def hue(m):
        if m == "A":
            return COLOURS["A"]
        behavioural = m[2:] if m.startswith("A+") else m
        if "B1" in behavioural and "B2" in behavioural:
            return "#6A5FA8"          # combined behavioural streams (blend slot)
        if "B2" in behavioural:
            return COLOURS["B2"]
        return COLOURS["B1"]
    return alt.Scale(domain=list(models), range=[hue(m) for m in models])


def standalone_chart(seven: pd.DataFrame, metric: str):
    d = seven[seven["model"].isin(["A", "B1", "B2"])].copy()
    d["Checkpoint"] = d["checkpoint"].str.upper()
    order = ["A", "B1", "B2"]
    return (
        alt.Chart(d)
        .mark_bar(size=26, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Checkpoint:N", title=None),
            xOffset=alt.XOffset("model:N", sort=order),
            y=alt.Y(f"{metric}:Q", title=metric.replace("_", "-").upper(),
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model:N", sort=order, title="Block",
                            scale=alt.Scale(domain=order,
                                            range=[COLOURS["A"], COLOURS["B1"], COLOURS["B2"]])),
            tooltip=["checkpoint", "model", metric],
        )
        .properties(height=300)
    )


def gains_chart(seeds: pd.DataFrame, metric_prefix: str):
    d = seeds.copy()
    d["Checkpoint"] = d["checkpoint"].str.upper()
    mean_col, sd_col = f"{metric_prefix}_mean", f"{metric_prefix}_sd"
    d["lo"] = d[mean_col] - d[sd_col]
    d["hi"] = d[mean_col] + d[sd_col]
    order = ["A+B1", "A+B2", "A+B1+B2"]
    base = alt.Chart(d).encode(
        x=alt.X("Checkpoint:N", title=None),
        xOffset=alt.XOffset("model:N", sort=order),
        color=alt.Color("model:N", sort=order, title="Configuration",
                        scale=block_colour_scale(order)),
    )
    bars = base.mark_bar(size=22, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        y=alt.Y(f"{mean_col}:Q", title=f"Mean Δ{metric_prefix[2:].upper()}-AUC vs Model A",
                scale=alt.Scale(domain=[-0.005, 0.025])),
        tooltip=["checkpoint", "model", mean_col, sd_col],
    )
    err = base.mark_rule(strokeWidth=2, color=INK).encode(y="lo:Q", y2="hi:Q")
    marginal = alt.Chart(pd.DataFrame({"y": [0.01]})).mark_rule(
        strokeDash=[5, 4], color=MUTED).encode(y="y:Q")
    meaningful = alt.Chart(pd.DataFrame({"y": [0.02]})).mark_rule(
        strokeDash=[2, 3], color=INK).encode(y="y:Q")
    labels = alt.Chart(pd.DataFrame({
        "y": [0.0105, 0.0205], "t": ["marginal (0.01)", "meaningful (0.02)"]})
    ).mark_text(align="left", dx=-160, dy=-6, color=MUTED, fontSize=11).encode(
        y="y:Q", text="t:N")
    return (bars + err + marginal + meaningful + labels).properties(height=320)


def shap_chart(shap_b: pd.DataFrame):
    d = shap_b.melt(id_vars="checkpoint", var_name="block", value_name="share")
    d["block"] = d["block"].str.replace("_share", "")
    d["Checkpoint"] = d["checkpoint"].str.upper()
    d["pct"] = (d["share"] * 100).round(2)
    order = ["A", "B1", "B2"]
    lines = (
        alt.Chart(d)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=80, filled=True))
        .encode(
            x=alt.X("Checkpoint:N", title=None),
            y=alt.Y("pct:Q", title="Share of mean |SHAP| (%)",
                    scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("block:N", sort=order, title="Block",
                            scale=alt.Scale(domain=order,
                                            range=[COLOURS["A"], COLOURS["B1"], COLOURS["B2"]])),
            tooltip=["checkpoint", "block", "pct"],
        )
    )
    labels = (
        alt.Chart(d[d["Checkpoint"] == "Q3"])
        .mark_text(align="left", dx=10, fontSize=12, color=INK)
        .encode(x=alt.X("Checkpoint:N"), y="pct:Q", text=alt.Text("pct:Q", format=".1f"))
    )
    return (lines + labels).properties(height=320)


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Temporal Feature Ablation Platform",
                   page_icon="📉", layout="wide")

seven, seeds, shap_b, from_files = load_results()
gains = gains_vs_baseline(seven)

st.title("Temporal Feature Ablation Platform")
st.caption(
    "Block-wise evaluation of behavioural telemetry beyond assessment performance "
    "for identifying at-risk MOOC learners · CIS4517 · Beryl Odingo (26673207)"
)

with st.sidebar:
    st.header("Study configuration")
    st.markdown(
        "- **Dataset:** MOOCCubeX matched subset — 438,675 learner-checkpoint rows\n"
        "- **Blocks:** A = assessment (6 features) · B1 = video (5) · B2 = discussion (5)\n"
        "- **Outcome:** low Q4 correctness (< 0.60)\n"
        "- **Checkpoints:** Q1–Q3, learner-specific temporal cut-offs\n"
        "- **Models:** XGBoost (primary), Logistic Regression, LightGBM\n"
        "- **Canonical config:** 200 trees · depth 3 · lr 0.05 · subsample 0.8 · xgboost 3.4.1"
    )
    st.header("Redundancy framework")
    st.markdown(BAND_HELP)
    source = "pipeline CSVs in ./data" if from_files else "embedded canonical FINAL run"
    st.caption(f"Data source: {source}")

tab_overview, tab_perf, tab_ablation, tab_shap, tab_run, tab_apply = st.tabs(
    ["Overview", "Checkpoint performance", "Incremental value",
     "SHAP attribution", "Run ablation", "Apply the framework"])

with tab_overview:
    st.subheader("What this platform shows")
    st.markdown(
        "This platform documents the dissertation's block-wise temporal ablation "
        "framework and lets the redundancy analysis be inspected checkpoint by "
        "checkpoint. Two structurally different behavioural streams were tested: "
        "**B1 (video)** — passive content consumption, and **B2 (discussion)** — "
        "active participation via comments and replies. Each stream is evaluated "
        "alone, combined, and added to the assessment baseline (Model A) at "
        "Q1, Q2 and Q3."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Learner-checkpoint rows", "438,675")
    c2.metric("Discussion events (matched)", "144,387")
    c3.metric("Best behavioural gain (10-seed mean ΔROC)", "+0.0016")
    c4.metric("Meaningfulness threshold", "0.02")
    st.markdown(
        "**Headline finding:** across three checkpoints, three algorithms and ten "
        "random splits, no behavioural configuration approached the 0.02 threshold. "
        "The only repeatable effect — video at Q1 — was statistically detectable "
        "but practically negligible under the study's pre-specified bands."
    )

with tab_perf:
    st.subheader("Standalone predictive value per block")
    metric = st.radio("Metric", ["roc_auc", "pr_auc"], horizontal=True,
                      format_func=lambda m: m.replace("_", "-").upper())
    st.altair_chart(standalone_chart(seven, metric), width='stretch')
    st.markdown(
        "Assessment (A) dominates and strengthens as evidence accumulates; video "
        "(B1) carries a modest real signal; discussion (B2) is near chance, partly "
        "reflecting sparsity (~85–88% of learner-checkpoint rows have no "
        "temporally eligible discussion activity)."
    )
    st.subheader("Full seven-configuration results")
    st.dataframe(seven, width='stretch', hide_index=True)

with tab_ablation:
    st.subheader("Incremental gains over the assessment-only baseline")
    st.caption("Bars: mean over ten stratified splits (seeds 0–9). "
               "Rules: ± one standard deviation. Reference lines: framework bands.")
    gmetric = st.radio("Gain metric", ["d_roc", "d_pr"], horizontal=True,
                       format_func=lambda m: "ΔROC-AUC" if m == "d_roc" else "ΔPR-AUC")
    st.altair_chart(gains_chart(seeds, gmetric), width='stretch')
    st.subheader("Framework verdicts (canonical split, seed 42)")
    verd = gains[gains["model"].isin(["A+B1", "A+B2", "A+B1+B2"])].copy()
    st.dataframe(verd, width='stretch', hide_index=True)
    st.markdown(
        "Every behavioural configuration is classified **negligible** at every "
        "checkpoint: the streams tested are practically redundant for this "
        "prediction task once assessment evidence is available."
    )

with tab_shap:
    st.subheader("Block-level SHAP attribution in the combined model")
    st.altair_chart(shap_chart(shap_b), width='stretch')
    st.markdown(
        "Attribution is **corroborative, not criterial**: model reliance shifts "
        "from behavioural evidence toward assessment performance as the course "
        "progresses (video declines to ~5.4% by Q3; discussion never exceeds 1%). "
        "The redundancy claim itself rests on the held-out ablation gains."
    )
    st.dataframe(
        shap_b.assign(**{c: (shap_b[c] * 100).round(2) for c in
                         ["A_share", "B1_share", "B2_share"]}),
        width='stretch', hide_index=True)

with tab_run:
    st.subheader("Run the full ablation pipeline on a dataset")
    st.markdown(
        "Upload a **modelling dataset** — one row per learner (or "
        "learner-checkpoint) with numeric feature columns and a binary outcome "
        "column — and the platform trains the study's canonical XGBoost "
        "configuration (200 trees, depth 3, learning rate 0.05, subsampling "
        "0.8, class weighting, stratified 80/20 split, seed 42) for every "
        "block combination, at every checkpoint, and evaluates the results "
        "with the redundancy framework. Uploading the pipeline's "
        "`full_checkpoint_model_data_with_b2_FINAL.csv` reproduces the "
        "dissertation's seven-model matrix."
    )
    ds = st.file_uploader("Modelling dataset CSV", type="csv", key="run_upload")
    if ds is not None:
        try:
            raw = pd.read_csv(ds)
        except Exception as exc:
            st.error(f"Could not read that file: {exc}")
            raw = None
        if raw is not None:
            st.caption(f"Loaded {len(raw):,} rows × {len(raw.columns)} columns.")
            cols = list(raw.columns)

            # ---- Outcome builder: binary flag, letter grade, pass/fail
            #      label, or numeric score with a threshold ----
            st.markdown("**Define the outcome (who counts as at risk)**")
            out_default = next(
                (c for c in cols if c.lower() in
                 ("low_q4_correctness", "target_low_q4")),
                next((c for c in cols if "grade" in c.lower()
                      or "outcome" in c.lower() or "result" in c.lower()),
                     cols[-1]))
            outcome_src = st.selectbox(
                "Outcome source column - a 0/1 flag, a grade or "
                "pass/fail label, or a numeric score",
                cols, index=cols.index(out_default))
            src = raw[outcome_src]
            target_col = "_target"
            derived = None
            if pd.api.types.is_numeric_dtype(src) and src.dropna().nunique() > 10:
                lo, hi = float(src.min()), float(src.max())
                default_thr = 60.0 if (lo >= 0 and hi <= 100) else float(src.median())
                direction = st.radio(
                    "A learner is at risk when the score is…",
                    ["below the threshold", "at or above the threshold"],
                    horizontal=True)
                thr = st.number_input("Threshold", value=default_thr)
                derived = ((src < thr) if direction.startswith("below")
                           else (src >= thr)).astype(int)
            else:
                values = sorted(src.dropna().astype(str).unique().tolist())
                risky = [v for v in values if v.strip().lower() in
                         ("1", "d", "e", "f", "fail", "failed", "refer",
                          "at risk", "at_risk", "low", "true", "yes")]
                risk_values = st.multiselect(
                    "Values that count as at risk "
                    "(e.g. D and F for letter grades; Fail for pass/fail)",
                    values, default=risky)
                if risk_values:
                    derived = src.astype(str).isin(risk_values).astype(int)
            derived_ok = derived is not None and derived.nunique() == 2
            if derived_ok:
                st.caption(f"Derived outcome: {int(derived.sum()):,} of "
                           f"{len(derived):,} learners at risk "
                           f"({derived.mean():.1%}).")
            else:
                st.warning("Choose a threshold or values that mark some - "
                           "but not all - learners as at risk.")
            cp_options = ["(none - treat as one checkpoint)"] + cols
            cp_default = ("checkpoint" if "checkpoint" in cols
                          else "(none - treat as one checkpoint)")
            cp_choice = st.selectbox(
                "Checkpoint column (labels like q1/q2/q3, or dates)",
                cp_options, index=cp_options.index(cp_default))
            cp_col = None if cp_choice.startswith("(none") else cp_choice

            # If the checkpoint column holds dates, bin it into time-ordered
            # periods so the temporal design still applies.
            work = raw.copy()
            if derived_ok:
                work[target_col] = derived.values
            if cp_col is not None:
                n_periods = st.slider(
                    "Periods to bin into, if the checkpoint column is dates",
                    2, 6, 3,
                    help="Ignored when the column already holds labels like "
                         "q1/q2/q3. Dates are split into this many "
                         "equal-sized, time-ordered periods (P1, P2, ...).")
                binned = maybe_bin_dates(raw[cp_col], n_periods)
                if binned is not None:
                    work["_period"] = binned
                    counts = work["_period"].value_counts().sort_index()
                    st.info(f"'{cp_col}' looks like dates - binned into "
                            f"{n_periods} time-ordered periods: "
                            + ", ".join(f"{k} ({v:,} rows)"
                                        for k, v in counts.items()))
                    cp_col = "_period"

            suggested = auto_assign_blocks(cols)
            id_candidates = [c for c in cols if c.lower().endswith("_id")
                             or c.lower() == "id"]
            id_col = id_candidates[0] if id_candidates else None
            feature_pool = [c for c in cols
                            if c not in {outcome_src, cp_choice}
                            and not is_excluded(c)
                            and (pd.api.types.is_numeric_dtype(raw[c])
                                 or raw[c].dropna().nunique() <= 12)]
            st.caption("Leakage note: if the outcome comes from a total or "
                       "final score, do not assign that score - or columns "
                       "computed from it (e.g. Total_Score, Grade) - as "
                       "features. Non-numeric features (Yes/No, categories) "
                       "are encoded automatically.")
            c1, c2, c3 = st.columns(3)
            block_map = {
                "A": c1.multiselect("Block A - assessment", feature_pool,
                                    default=[f for f in suggested["A"]
                                             if f in feature_pool]),
                "B1": c2.multiselect("Block B1 - video", feature_pool,
                                     default=[f for f in suggested["B1"]
                                              if f in feature_pool]),
                "B2": c3.multiselect("Block B2 - discussion", feature_pool,
                                     default=[f for f in suggested["B2"]
                                              if f in feature_pool]),
            }
            n_blocks = sum(1 for b in block_map.values() if b)
            overlap = (set(block_map["A"]) & set(block_map["B1"])) | \
                      (set(block_map["A"]) & set(block_map["B2"])) | \
                      (set(block_map["B1"]) & set(block_map["B2"]))
            if overlap:
                st.error("A feature can belong to one block only: "
                         + ", ".join(sorted(overlap)))
            elif n_blocks == 0:
                st.info("Assign features to at least one block to continue.")
            elif not derived_ok:
                st.error("Finish defining the outcome above before running.")
            else:
                n_models = len(build_configs(block_map))
                n_cps = work[cp_col].nunique() if cp_col else 1
                st.caption(f"This will train {n_models * n_cps} models "
                           f"({n_models} configurations × {n_cps} checkpoint(s)). "
                           "Large datasets can take several minutes.")
                if st.button("Run ablation", type="primary"):
                    prog = st.progress(0.0, text="Starting…")
                    try:
                        res_run = run_ablation(work, target_col, cp_col,
                                               block_map, progress=prog)
                        prog.empty()
                        cps_order = sorted(res_run["checkpoint"].unique())

                        if "A" in set(res_run["model"]):
                            out_run = gains_vs_baseline(res_run)

                            # ---- Plain-language verdict ----
                            st.subheader("What this means")
                            labels = {}
                            if block_map["B1"]:
                                labels["A+B1"] = "The video stream (B1)"
                            if block_map["B2"]:
                                labels["A+B2"] = "The second behavioural stream (B2)"
                            if block_map["B1"] and block_map["B2"]:
                                labels["A+B1+B2"] = "All behavioural data combined"
                            for line in redundancy_summary(out_run, cps_order,
                                                           labels):
                                if "no meaningful predictive value at any" in line \
                                        or "becomes redundant" in line:
                                    st.success(line, icon="✅")
                                elif "still adds meaningful value" in line:
                                    st.warning(line, icon="📈")
                                else:
                                    st.info(line)
                            st.caption(
                                "Redundant = adding the stream to the "
                                "assessment-only model improves neither ROC-AUC "
                                "nor PR-AUC by 0.02 or more.")

                            st.subheader("Incremental gains and framework verdicts")
                            st.dataframe(out_run, width='stretch',
                                         hide_index=True)
                        else:
                            st.info("No Block A features were assigned, so no "
                                    "assessment baseline exists to compute "
                                    "incremental gains against.")

                        st.subheader("Performance by configuration")
                        st.dataframe(res_run, width='stretch', hide_index=True)
                        st.download_button(
                            "Download performance table (CSV)",
                            res_run.to_csv(index=False).encode(),
                            file_name="ablation_performance.csv",
                            mime="text/csv")

                        # ---- Learners likely to struggle ----
                        st.subheader("Learners likely to struggle")
                        scores = score_learners(work, target_col, cp_col,
                                                block_map, id_col)
                        latest = cps_order[-1]
                        flagged = (scores[scores["checkpoint"] == str(latest)]
                                   .sort_values("risk_score", ascending=False))
                        n_high = int((flagged["risk_band"] == "high").sum())
                        n_elev = int((flagged["risk_band"] == "elevated").sum())
                        st.markdown(
                            f"At the latest checkpoint (**{str(latest).upper()}**), "
                            f"**{n_high}** learners score in the *high* risk band "
                            f"and **{n_elev}** in *elevated*. The 20 highest-risk "
                            "learners:")
                        st.dataframe(flagged.head(20), width='stretch',
                                     hide_index=True)
                        st.download_button(
                            "Download full risk list, all checkpoints (CSV)",
                            scores.to_csv(index=False).encode(),
                            file_name="at_risk_learners.csv",
                            mime="text/csv")
                        st.caption(
                            "Risk scores are decision-support estimates, not "
                            "judgements about learners: use them to prioritise "
                            "outreach, alongside human context. Rows marked "
                            "held_out = False were in the model's training "
                            "split, so their scores are slightly optimistic.")
                    except Exception as exc:
                        prog.empty()
                        st.error(f"Training failed: {exc}")
    else:
        st.caption("Tip: your own pipeline export "
                   "`full_checkpoint_model_data_with_b2_FINAL.csv` works here "
                   "as-is - columns are recognised automatically.")

with tab_apply:
    st.subheader("Apply the redundancy framework to your own results")
    st.markdown(
        "Institutions can evaluate their own telemetry streams with this framework. "
        "Upload a CSV with columns `checkpoint`, `model`, `roc_auc`, `pr_auc`, "
        "including one baseline row per checkpoint (`model = A`). The platform "
        "computes each configuration's incremental gains over the baseline and "
        "classifies them against the pre-specified bands."
    )
    up = st.file_uploader("Results CSV", type="csv")
    if up is not None:
        try:
            user_df = pd.read_csv(up)
            required = {"checkpoint", "model", "roc_auc", "pr_auc"}
            missing = required - set(user_df.columns)
            if missing:
                st.error(f"Missing columns: {', '.join(sorted(missing))}")
            elif "A" not in set(user_df["model"]):
                st.error("No baseline rows found: include one row with model = A "
                         "per checkpoint.")
            else:
                out = gains_vs_baseline(user_df)
                st.dataframe(out, width='stretch', hide_index=True)
                n_red = int(out["redundant_at_checkpoint"].sum())
                st.markdown(
                    f"**{n_red} of {len(out)}** configuration-checkpoint results "
                    "fall below the 0.02 threshold on both metrics and are "
                    "classified as practically redundant."
                )
                st.download_button(
                    "Download classified results (CSV)",
                    out.to_csv(index=False).encode(),
                    file_name="redundancy_classification.csv",
                    mime="text/csv")
        except Exception as exc:  # surface parse errors to the user plainly
            st.error(f"Could not read that file: {exc}")
    else:
        st.caption("No file yet? The dissertation's own results are pre-loaded in "
                   "the other tabs as a worked example.")

st.divider()
st.caption(
    "Temporal Feature Ablation Platform · built with Streamlit, pandas and Altair · "
    "results from the canonical FINAL pipeline run (xgboost 3.4.1)."
)
