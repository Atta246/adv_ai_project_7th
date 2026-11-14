# Breast Cancer Diagnosis with Supervised Machine Learning

Classifying breast masses as **benign** or **malignant** from nine cytology scores
(Breast Cancer Wisconsin – Original, UCI) with five tuned supervised models:
**KNN, Decision Tree, Random Forest, Artificial Neural Network (MLP) and SVM**.

> Educational project — not a medical device.

---

## Results at a glance (held-out test set, n = 139, malignant = positive)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Missed cancers |
|---|---|---|---|---|---|---|
| KNN | 0.964 | 0.957 | 0.938 | 0.947 | 0.987 | 3 |
| Decision Tree | 0.928 | 0.932 | 0.854 | 0.891 | 0.966 | 7 |
| Random Forest | 0.964 | 0.957 | 0.938 | 0.947 | 0.994 | 3 |
| **★ ANN (MLP) — selected** | **0.971** | **0.978** | 0.938 | 0.957 | **0.996** | 3 |
| SVM | 0.971 | 0.958 | **0.958** | **0.958** | 0.996 | 2 |
| ANN + screening threshold (0.18) | 0.978 | 0.959 | **0.979** | **0.969** | 0.996 | **1** |

The **ANN (MLP)** was selected **on training data only** (pre-declared rule: highest mean F1 in
10-fold × 5-repeat stratified CV, ties broken by recall) and then confirmed on the untouched
test set. ANN, SVM, Random Forest and KNN are statistically tied; the Decision Tree is clearly weakest.

## What makes this project stand out

- **Leakage-free by design** – imputation and scaling live inside a scikit-learn `Pipeline`,
  re-fitted in every CV fold; the test set is used exactly once.
- **Hidden data issues found** – missing values disguised as `?`, 8 exact duplicates, an ID column.
- **350 hyperparameter configurations** tuned with 10-fold stratified grid search (F1 of the malignant class).
- **Model selection before testing** with a transparent, pre-declared rule.
- **Statistics, not just scores** – corrected resampled t-test, McNemar test, bootstrap 95 % CIs.
- **Nested cross-validation** for an unbiased performance estimate.
- **Clinical screening threshold** chosen on training data (recall 93.8 % → 97.9 %).
- **Explainability** – readable decision-tree rules (on the original 1–10 scale) and permutation importance.
- **Robustness checks** – duplicate-profile analysis, learning curve, error analysis.
- **Interactive demo app** (Streamlit) to try all five models with sliders.

## Project structure

```
breast-cancer-classification/
├── Breast_Cancer_Classification.ipynb   ← main deliverable (code, outputs, charts, explanations)
├── slides/
│   ├── Breast_Cancer_Classification.pptx ← presentation (with speaker notes)
│   └── Breast_Cancer_Classification.pdf
├── app/app.py                            ← Streamlit demo
├── data/breast-cancer-wisconsin.data     ← UCI dataset (downloaded automatically if missing)
├── figures/                              ← all 19 charts (PNG, used in the slides)
├── models/models.joblib                  ← the five tuned models
├── results/                              ← model_comparison.csv, summary.json
├── requirements.txt
└── .vscode/                              ← recommended extensions, settings, launch config
```

## How to run in VS Code

1. Install **Python 3.10+** and open this folder in VS Code (`File → Open Folder…`).
   Accept the recommended extensions (Python, Jupyter).
2. Open a terminal (`` Ctrl+` ``) and create a virtual environment:

   **Windows**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   **macOS / Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Open `Breast_Cancer_Classification.ipynb`, choose the **.venv** kernel (top-right),
   and click **Run All**. A full run takes about 5–10 minutes (nested CV is the slowest part;
   set `RUN_NESTED_CV = False` in the setup cell to skip it).
4. Optional demo app:
   ```bash
   streamlit run app/app.py
   ```
   or press **F5** and pick *Run demo app (Streamlit)*.

Numbers may differ very slightly with other library versions; everything is seeded (`random_state = 42`).


