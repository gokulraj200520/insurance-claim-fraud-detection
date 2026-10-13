"""Data mining lab: PCA, classification models with evaluation metrics, anomaly detection."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import database as db
from src.features import FEATURE_LABELS

from . import data_access as da
from . import ui


def _capture(flag: np.ndarray, y: np.ndarray) -> dict:
    tp = int((flag & (y == 1)).sum())
    return {"Flagged claims": int(flag.sum()), "Frauds caught": tp,
            "Precision": round(tp / max(int(flag.sum()), 1), 3), "Recall": round(tp / max(int((y == 1).sum()), 1), 3)}


def render() -> None:
    ui.page_header("Data Mining Lab", "Data cleaning, PCA, classification models and anomaly detection")
    v = db.db_version()
    df = da.claims(v)
    tab1, tab2, tab3 = st.tabs(["Data Cleaning and PCA", "Classification Models", "Anomaly Detection"])

    # ------------------------------------------------------------------ PCA
    with tab1:
        ui.section("Data Cleaning Summary")
        q = da.quality(v)
        stats = db.query("SELECT metric, value FROM etl_log").set_index("metric")["value"]
        ui.kpi_row([
            ("Rows Extracted", f"{int(stats['rows_extracted']):,}", "Raw operational records"),
            ("Duplicates Removed", f"{int(stats['duplicates_removed']):,}", "Exact duplicate claims"),
            ("Invalid Values Fixed", f"{int(stats['invalid_values_fixed']):,}", "Impossible ages / scores"),
            ("Values Imputed", f"{int(stats['missing_values_imputed']):,}", "Median / mode / customer-level"),
            ("Clean Rows Loaded", f"{int(stats['rows_loaded']):,}", "Warehouse fact table", "green"),
        ])
        ui.html_table(q, max_height=330)

        ui.section("Principal Component Analysis (PCA)")
        p = da.pca_result(v)
        evr = p["evr"]
        k = min(len(evr), 12)
        pcs = [f"PC{i + 1}" for i in range(k)]
        c1, c2 = st.columns(2)
        scree = go.Figure()
        scree.add_bar(x=pcs, y=evr[:k] * 100, name="Explained variance (%)", marker_color=ui.LIGHT_BLUE)
        scree.add_scatter(x=pcs, y=np.cumsum(evr)[:k] * 100, name="Cumulative (%)", mode="lines+markers",
                          line=dict(color=ui.NAVY, width=2.5))
        scree.update_layout(yaxis_title="Variance (%)")
        with c1:
            ui.show(ui.style_fig(scree, 380, "Scree plot"), key="pca_scree")
        sample = p["plot"].sample(min(len(p["plot"]), 3000), random_state=1)
        sc = px.scatter(sample, x="PC1", y="PC2", color="Label", opacity=0.6,
                        color_discrete_map={"Genuine": "#9DB7D5", "Fraud": ui.RISK_COLORS["High"]})
        with c2:
            ui.show(ui.style_fig(sc, 380, "Claims projected on the first two components"), key="pca_scatter")
        n80 = int(np.argmax(np.cumsum(evr) >= 0.80)) + 1
        ui.note(f"<b>{n80}</b> principal components explain at least 80% of the variance of the {len(evr)} engineered features. "
                f"PC1 and PC2 together explain <b>{(evr[0] + evr[1]) * 100:.1f}%</b>.")
        ld = p["loadings"].copy()
        ld["Strength"] = ld.abs().max(axis=1)
        ld = ld.sort_values("Strength", ascending=False).head(8).drop(columns="Strength").round(3).reset_index()
        ld = ld.rename(columns={"index": "Feature"})
        st.markdown("**Strongest feature loadings (PC1 and PC2)**")
        ui.html_table(ld, max_height=320)

    # ------------------------------------------------------------------ classification
    with tab2:
        res = da.trained(v)
        m = res["metrics"]
        ui.section("Model Comparison (hold-out test set)")
        st.caption(f"Stratified split: {res['n_train']:,} training and {res['n_test']:,} test claims. Class imbalance handled with balanced class weights.")
        shown = m.copy()
        for c in shown.columns[1:]:
            shown[c] = shown[c].round(3)
        ui.html_table(shown, max_height=200)
        long = m.melt(id_vars="Model", var_name="Metric", value_name="Score")
        bar = px.bar(long, x="Metric", y="Score", color="Model", barmode="group")
        bar.update_yaxes(range=[0, 1.05])
        c1, c2 = st.columns(2)
        with c1:
            ui.show(ui.style_fig(bar, 400, "Evaluation metrics by model"), key="clf_bar")
        roc = go.Figure()
        for name, d in res["details"].items():
            auc = m.loc[m["Model"] == name, "ROC-AUC"].iloc[0]
            roc.add_scatter(x=d["fpr"], y=d["tpr"], mode="lines", name=f"{name} (AUC {auc:.3f})")
        roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#999"), name="Random guess")
        roc.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
        with c2:
            ui.show(ui.style_fig(roc, 400, "ROC curves"), key="clf_roc")

        c3, c4 = st.columns(2)
        with c3:
            name = st.selectbox("Confusion matrix for", list(res["details"]))
            cm = res["details"][name]["cm"]
            heat = px.imshow(cm, text_auto=True, x=["Predicted genuine", "Predicted fraud"],
                             y=["Actual genuine", "Actual fraud"], color_continuous_scale=["#FFFFFF", "#9DB7D5", "#1F3A5F"])
            ui.show(ui.style_fig(heat, 360, f"Confusion matrix - {name}"), key="clf_cm")
        with c4:
            imp = res["importance"].head(12).iloc[::-1]
            ib = px.bar(x=imp.values, y=[FEATURE_LABELS[i] for i in imp.index], orientation="h",
                        labels={"x": "Importance", "y": ""}, color_discrete_sequence=[ui.NAVY])
            ui.show(ui.style_fig(ib, 400, "Random Forest feature importance"), key="clf_imp")
        with st.expander("Evaluation metric definitions"):
            st.markdown(
                "- **Accuracy** - share of all claims classified correctly (misleading when fraud is rare).\n"
                "- **Precision** - of the claims flagged as fraud, the share that really are fraud.\n"
                "- **Recall** - of all real frauds, the share that were detected.\n"
                "- **F1-Score** - harmonic mean of precision and recall.\n"
                "- **ROC-AUC** - ability to rank fraud above genuine claims across all thresholds.\n"
                "- **PR-AUC** - area under the precision-recall curve; more informative for imbalanced data.")

    # ------------------------------------------------------------------ anomaly detection
    with tab3:
        y = df["is_fraud"].to_numpy()
        ui.section("Isolation Forest")
        cont = st.slider("Contamination (expected share of anomalies)", 0.02, 0.20, 0.09, 0.01)
        score, flag = da.isolation(v, float(cont))
        d = pd.DataFrame({"Anomaly score": score, "Label": np.where(y == 1, "Fraud", "Genuine")})
        c1, c2 = st.columns([3, 2])
        hist = px.histogram(d, x="Anomaly score", color="Label", nbins=50, barmode="overlay", opacity=0.7,
                            color_discrete_map={"Genuine": "#9DB7D5", "Fraud": ui.RISK_COLORS["High"]})
        with c1:
            ui.show(ui.style_fig(hist, 360, "Isolation Forest anomaly score distribution"), key="ano_hist")
        iso_stats = _capture(flag, y)
        with c2:
            ui.kpi_row([("Flagged", f"{iso_stats['Flagged claims']:,}", "Anomalous claims")])
            ui.kpi_row([("Precision", f"{iso_stats['Precision']:.2f}", f"{iso_stats['Frauds caught']} frauds caught", "red")])
            ui.kpi_row([("Recall", f"{iso_stats['Recall']:.2f}", "Share of all frauds detected", "green")])

        ui.section("Statistical Outlier Detection (Z-score and IQR)")
        out = da.outliers(v)
        box = px.box(df, x="policy_type", y="claim_amount", color="policy_type", points="outliers",
                     labels={"policy_type": "Policy type", "claim_amount": "Claim amount (INR)"},
                     color_discrete_sequence=[ui.NAVY, "#4C8BC2", "#9DB7D5", "#C98A00"])
        box.update_layout(showlegend=False)
        c3, c4 = st.columns([3, 2])
        with c3:
            ui.show(ui.style_fig(box, 380, "Claim amount distribution and outliers by policy type"), key="ano_box")
        comp = pd.DataFrame([
            {"Method": "Z-score (|z| > 3)", **_capture(out["z_flag"].to_numpy(), y)},
            {"Method": "IQR fences (1.5 x IQR)", **_capture(out["iqr_flag"].to_numpy(), y)},
            {"Method": f"Isolation Forest ({cont:.0%})", **iso_stats},
        ])
        with c4:
            st.markdown("**Detection method comparison**")
            ui.html_table(comp, max_height=240)
            st.caption("Unsupervised methods need no fraud labels, so they can surface previously unseen fraud patterns.")
