"""Supervised fraud classification and evaluation."""
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE
from .features import build_features


def make_models() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=15, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE),
    }


def train_and_evaluate(df: pd.DataFrame) -> dict:
    X, y = build_features(df), df["is_fraud"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    rows, details = [], {}
    for name, model in make_models().items():
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred, zero_division=0),
            "Recall": recall_score(y_te, pred, zero_division=0),
            "F1-Score": f1_score(y_te, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_te, proba),
            "PR-AUC": average_precision_score(y_te, proba),
        })
        fpr, tpr, _ = roc_curve(y_te, proba)
        details[name] = {"fpr": fpr, "tpr": tpr, "cm": confusion_matrix(y_te, pred, labels=[0, 1]),
                         "model": model}

    rf = details["Random Forest"]["model"]
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    final_model = clone(make_models()["Random Forest"]).fit(X, y)   # production model: all labelled data
    return {"metrics": pd.DataFrame(rows), "details": details, "importance": importance,
            "final_model": final_model, "n_train": len(X_tr), "n_test": len(X_te)}


def oof_probabilities(df: pd.DataFrame):
    """Out-of-fold fraud probabilities (5-fold) so no claim is scored by a model that saw its label."""
    X, y = build_features(df), df["is_fraud"].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(make_models()["Random Forest"], X, y, cv=cv, method="predict_proba")[:, 1]
