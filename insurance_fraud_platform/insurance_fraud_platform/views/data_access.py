"""Cached data access for the Streamlit pages (cache key = database modification time)."""
import pandas as pd
import streamlit as st

from src import anomaly, features, models, preprocessing


@st.cache_data(show_spinner=False)
def claims(version: float) -> pd.DataFrame:
    return features.load_claims_frame()


@st.cache_resource(show_spinner="Training classification models ...")
def trained(version: float) -> dict:
    return models.train_and_evaluate(claims(version))


@st.cache_data(show_spinner="Running PCA ...")
def pca_result(version: float) -> dict:
    df = claims(version)
    X = features.build_features(df)
    pca, comps = preprocessing.run_pca(X)
    loadings = pd.DataFrame(pca.components_[:2].T, index=[features.FEATURE_LABELS[c] for c in X.columns],
                            columns=["PC1", "PC2"])
    plot = pd.DataFrame({"PC1": comps[:, 0], "PC2": comps[:, 1],
                         "Label": df["is_fraud"].map({0: "Genuine", 1: "Fraud"}).values})
    return {"evr": pca.explained_variance_ratio_, "plot": plot, "loadings": loadings}


@st.cache_data(show_spinner=False)
def quality(version: float) -> pd.DataFrame:
    return preprocessing.data_quality_report()


@st.cache_data(show_spinner="Running Isolation Forest ...")
def isolation(version: float, contamination: float):
    return anomaly.isolation_forest_scores(claims(version), contamination)


@st.cache_data(show_spinner=False)
def outliers(version: float) -> pd.DataFrame:
    return anomaly.statistical_outliers(claims(version))
