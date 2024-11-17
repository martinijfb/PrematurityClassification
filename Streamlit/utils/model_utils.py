import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import streamlit as st

@st.cache_resource
def load_model():
    # Load model
    with open("DATA/model_2_lr_with_pca.p", "rb") as file:
        model = pickle.load(file)
    return model

@st.cache_data
def evaluation(_model, X, y, mode="Test"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if mode == "Train":
        y_pred = _model.predict(X_train.to_numpy())
        ground_truth = y_train
    else:
        y_pred = _model.predict(X_test.to_numpy())
        ground_truth = y_test

    metrics = {
        f"{mode} Accuracy": round(accuracy_score(ground_truth, y_pred), 2),
        f"{mode} Precision": round(precision_score(ground_truth, y_pred), 2),
        f"{mode} Recall/Sensitivity": round(recall_score(ground_truth, y_pred), 2),
        f"{mode} Specificity": round(
            recall_score(ground_truth, y_pred, pos_label=0), 2
        ),
        f"{mode} F1 Score": round(f1_score(ground_truth, y_pred), 2),
    }
    return metrics

@st.cache_data
def predict_prematurity(_model, X, y, i):
    prediction = _model.predict(X.iloc[i].values.reshape(1, -1))
    return prediction[0], y.iloc[i]
