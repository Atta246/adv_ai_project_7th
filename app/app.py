"""
Breast Cancer Diagnosis — interactive demo (Streamlit)

Run from the project root:
    streamlit run app/app.py

The app loads the five tuned models saved by the notebook (models/models.joblib).
If that file is missing or was saved with a different scikit-learn version, the app
re-trains the models automatically from the dataset using the tuned hyperparameters
stored in results/summary.json.

Educational project only — not a medical device.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "models.joblib"
SUMMARY_PATH = ROOT / "results" / "summary.json"
COMPARISON_PATH = ROOT / "results" / "model_comparison.csv"
DATA_PATH = ROOT / "data" / "breast-cancer-wisconsin.data"

FEATURES = ["Clump_Thickness", "Uniformity_Cell_Size", "Uniformity_Cell_Shape",
            "Marginal_Adhesion", "Single_Epithelial_Cell_Size", "Bare_Nuclei",
            "Bland_Chromatin", "Normal_Nucleoli", "Mitoses"]
HELP = {
    "Clump_Thickness": "Malignant cells tend to grow in multi-layers",
    "Uniformity_Cell_Size": "Cancer cells vary more in size",
    "Uniformity_Cell_Shape": "Cancer cells vary more in shape",
    "Marginal_Adhesion": "Cancer cells tend to lose adhesion",
    "Single_Epithelial_Cell_Size": "Enlarged epithelial cells may indicate malignancy",
    "Bare_Nuclei": "Nuclei not surrounded by cytoplasm",
    "Bland_Chromatin": "Coarser chromatin texture in cancer cells",
    "Normal_Nucleoli": "More prominent nucleoli in cancer cells",
    "Mitoses": "Rate of cell division",
}
PRESETS = {
    "Typical benign": [3, 1, 1, 1, 2, 1, 2, 1, 1],
    "Typical malignant": [8, 7, 7, 6, 5, 10, 7, 7, 2],
    "Borderline": [5, 3, 3, 3, 3, 4, 4, 3, 1],
}


def _retrain_models():
    """Rebuild the tuned models from the data (used when the saved file cannot be loaded)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    summary = json.loads(SUMMARY_PATH.read_text())
    cols = ["id"] + FEATURES + ["Class"]
    df = pd.read_csv(DATA_PATH, header=None, names=cols, na_values="?").drop_duplicates()
    X, y = df[FEATURES], (df["Class"] == 4).astype(int)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    builders = {
        "KNN": KNeighborsClassifier,
        "Decision Tree": lambda **p: DecisionTreeClassifier(random_state=42, **p),
        "Random Forest": lambda **p: RandomForestClassifier(n_estimators=300, random_state=42, **p),
        "ANN (MLP)": lambda **p: MLPClassifier(max_iter=2000, random_state=42, **p),
        "SVM": lambda **p: SVC(random_state=42, **p),
    }
    models = {}
    for name, make in builders.items():
        params = dict(summary["best_params"][name])
        if "hidden_layer_sizes" in params:
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()), ("model", make(**params))])
        if name == "SVM":
            pipe = CalibratedClassifierCV(pipe, method="sigmoid", cv=5)
        models[name] = pipe.fit(X_train, y_train)
    return {"models": models, "best_name": summary["best_model"], "features": FEATURES}


@st.cache_resource(show_spinner="Loading models…")
def load_bundle():
    try:
        import joblib
        bundle = joblib.load(MODEL_PATH)
        return bundle, "loaded"
    except Exception:
        return _retrain_models(), "retrained"


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺", layout="wide")
st.title("🩺 Breast Cancer Diagnosis — ML Demo")
st.caption("Breast Cancer Wisconsin (Original) · five tuned supervised models · "
           "educational project only, **not** a medical device.")

bundle, status = load_bundle()
models, best_name = bundle["models"], bundle["best_name"]
if status == "retrained":
    st.info("Saved models were not compatible with this environment, so they were re-trained "
            "from the data with the tuned hyperparameters.")

for f, v in zip(FEATURES, PRESETS["Typical benign"]):
    st.session_state.setdefault(f, v)

with st.sidebar:
    st.header("Model")
    names = list(models.keys())
    model_name = st.selectbox("Classifier", names, index=names.index(best_name),
                              format_func=lambda n: f"★ {n} (best)" if n == best_name else n)
    st.header("Quick examples")
    cols = st.columns(len(PRESETS))
    for col, (label, vals) in zip(cols, PRESETS.items()):
        if col.button(label.split()[-1].capitalize(), help=label):
            for f, v in zip(FEATURES, vals):
                st.session_state[f] = v
    st.header("Cytology scores (1 = normal, 10 = most abnormal)")
    inputs = {f: st.slider(f.replace("_", " "), 1, 10, key=f, help=HELP[f]) for f in FEATURES}

X_new = pd.DataFrame([inputs], columns=FEATURES)
proba = float(models[model_name].predict_proba(X_new)[0, 1])
is_malignant = proba >= 0.5

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Prediction")
    if is_malignant:
        st.error(f"### Malignant  —  {proba:.1%} probability")
    else:
        st.success(f"### Benign  —  {1 - proba:.1%} probability")
    st.progress(proba, text=f"Estimated probability of malignancy: {proba:.1%}")
    st.caption(f"Model: **{model_name}** · decision threshold 0.5")

with right:
    st.subheader("What do all five models say?")
    all_probs = pd.DataFrame({"Malignancy probability": [float(m.predict_proba(X_new)[0, 1]) for m in models.values()]},
                             index=list(models.keys()))
    st.bar_chart(all_probs, horizontal=True, height=260)
    agree = int((all_probs["Malignancy probability"] >= 0.5).sum())
    st.caption(f"{agree} of {len(models)} models predict **malignant** for this input.")

st.divider()
with st.expander("📊 Test-set comparison of the five models", expanded=False):
    if COMPARISON_PATH.exists():
        table = pd.read_csv(COMPARISON_PATH, index_col=0)
        st.dataframe(table.round(4))
    else:
        st.write("Run the notebook to generate results/model_comparison.csv.")
with st.expander("ℹ️ About this project"):
    st.markdown(
        "- **Data:** 699 fine-needle aspirate samples (UCI), 9 cytology scores, 2 classes.\n"
        "- **Pipeline:** median imputation → standardization → classifier, fitted on the training set only.\n"
        "- **Tuning:** grid search with 10-fold stratified cross-validation, optimising F1 for the malignant class.\n"
        "- **Selection:** best model chosen by 10×5 repeated CV on training data, confirmed on a held-out 20 % test set."
    )
