"""
Shopper Purchase Predictor - Streamlit front end for ML Assignment 2.

Loads the five pre-trained pipelines from model/ and evaluates them on a
test CSV uploaded by the user. Nothing is retrained here.
"""
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

warnings.filterwarnings("ignore", category=FutureWarning)

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "model"
TARGET = "Revenue"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
REQUIRED_COLS = ["Administrative", "Administrative_Duration", "Informational",
                 "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
                 "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month",
                 "OperatingSystems", "Browser", "Region", "TrafficType",
                 "VisitorType", "Weekend"]

st.set_page_config(page_title="Shopper Purchase Predictor", page_icon="🛒", layout="wide")


@st.cache_resource
def load_pipeline(file_name):
    return joblib.load(MODEL_DIR / file_name)


def clean_upload(raw_df):
    """Bring an uploaded CSV into the same shape the pipelines were trained on."""
    shop_df = raw_df.copy()
    # Weekend / Revenue may arrive as TRUE/FALSE, True/False or 1/0
    bool_map = {"true": 1, "false": 0, "1": 1, "0": 0}
    shop_df["Weekend"] = (shop_df["Weekend"].astype(str).str.strip().str.lower()
                          .map(bool_map).map({1: "True", 0: "False"}))
    shop_df[TARGET] = (shop_df[TARGET].astype(str).str.strip().str.lower()
                       .map(bool_map))
    return shop_df.dropna(subset=[TARGET])


def evaluate(pipeline, X, y):
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]
    scores = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob) if y.nunique() > 1 else float("nan"),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return scores, y_pred


def draw_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax,
                xticklabels=["No purchase", "Purchase"],
                yticklabels=["No purchase", "Purchase"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv",
                                help="Must contain the 17 feature columns and the "
                                     "'Revenue' target column.")
    chosen_model = st.selectbox("Choose a classifier", list(MODEL_FILES))
    st.divider()
    sample_path = APP_DIR / "test_data.csv"
    if sample_path.exists():
        st.download_button("⬇️ Download sample test_data.csv",
                           sample_path.read_bytes(), "test_data.csv", "text/csv")
    st.caption("Models were trained offline on 80% of the UCI Online Shoppers "
               "dataset. This app only runs inference.")

# ---------------- header ----------------
st.title("🛒 Will this visitor buy?")
st.markdown("Evaluate five pre-trained classifiers on e-commerce browsing sessions "
            "from the **UCI Online Shoppers Purchasing Intention** dataset. "
            "Upload a test CSV in the sidebar to begin.")

if uploaded is None:
    st.info("👈 Upload a CSV to see results. No file? Use the sample download "
            "button in the sidebar.")
    st.stop()

raw_df = pd.read_csv(uploaded)
missing = [c for c in REQUIRED_COLS + [TARGET] if c not in raw_df.columns]
if missing:
    st.error(f"The uploaded file is missing these columns: {', '.join(missing)}")
    st.stop()

test_df = clean_upload(raw_df)
X_test, y_test = test_df[REQUIRED_COLS], test_df[TARGET].astype(int)

c1, c2, c3 = st.columns(3)
c1.metric("Sessions in file", f"{len(test_df):,}")
c2.metric("Actual purchases", f"{int(y_test.sum()):,}")
c3.metric("Purchase rate", f"{y_test.mean():.1%}")

tab_single, tab_compare, tab_data = st.tabs(
    ["📊 Selected model", "🏁 Compare all models", "🔍 Data preview"])

# ---------------- single model ----------------
with tab_single:
    pipeline = load_pipeline(MODEL_FILES[chosen_model])
    scores, y_pred = evaluate(pipeline, X_test, y_test)

    st.subheader(f"{chosen_model}: evaluation metrics")
    cols = st.columns(6)
    for col, (name, value) in zip(cols, scores.items()):
        col.metric(name, f"{value:.4f}")

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("**Confusion matrix**")
        st.pyplot(draw_confusion(y_test, y_pred, chosen_model))
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y_test, y_pred, labels=[0, 1],
                                       target_names=["No purchase", "Purchase"],
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

# ---------------- comparison ----------------
with tab_compare:
    st.subheader("All five models on the uploaded data")
    all_scores = {}
    for name, file_name in MODEL_FILES.items():
        all_scores[name], _ = evaluate(load_pipeline(file_name), X_test, y_test)
    compare_df = pd.DataFrame(all_scores).T.round(4)
    compare_df.index.name = "ML Model Name"
    st.dataframe(compare_df.style.highlight_max(axis=0, color="#c6efce"),
                 use_container_width=True)

    metric_pick = st.radio("Plot metric", list(compare_df.columns), horizontal=True,
                           index=list(compare_df.columns).index("MCC"))
    fig, ax = plt.subplots(figsize=(8, 3))
    ordered = compare_df[metric_pick].sort_values()
    ax.barh(ordered.index, ordered.values, color="#2e7d32")
    for i, v in enumerate(ordered.values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(metric_pick)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

    best = compare_df["MCC"].idxmax()
    st.success(f"Best model on this file by MCC: **{best}** "
               f"(MCC = {compare_df.loc[best, 'MCC']:.4f})")

# ---------------- data preview ----------------
with tab_data:
    st.dataframe(raw_df.head(50), use_container_width=True)
    st.caption(f"Showing first 50 of {len(raw_df):,} rows.")
