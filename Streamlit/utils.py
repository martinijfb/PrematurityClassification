from io import BytesIO
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
from cryptography.fernet import Fernet
import seaborn as sns
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import streamlit as st


@st.cache_data
def plot_confusion_matrices(_model, X, y):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Predictions
    y_train_pred = _model.predict(X_train.to_numpy())
    y_test_pred = _model.predict(X_test.to_numpy())
    y_all_pred = _model.predict(X.to_numpy())

    # Confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    cm_all = confusion_matrix(y, y_all_pred)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot train confusion matrix
    disp_train = ConfusionMatrixDisplay(
        confusion_matrix=cm_train, display_labels=["Term", "Preterm"]
    )
    disp_train.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Train Confusion Matrix")

    # Plot test confusion matrix
    disp_test = ConfusionMatrixDisplay(
        confusion_matrix=cm_test, display_labels=["Term", "Preterm"]
    )
    disp_test.plot(ax=axes[1], cmap="Blues", colorbar=False)
    axes[1].set_title("Test Confusion Matrix")

    # Plot all data confusion matrix
    disp_all = ConfusionMatrixDisplay(
        confusion_matrix=cm_all, display_labels=["Term", "Preterm"]
    )
    disp_all.plot(ax=axes[2], cmap="Blues", colorbar=False)
    axes[2].set_title("All Data Confusion Matrix")

    # Adjust layout and remove grid lines
    for ax in axes:
        ax.grid(False)

    plt.tight_layout()
    return fig


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


# Function to load and decrypt data
@st.cache_data
def load_data():
    # Load encryption key
    key = os.getenv("ENCRYPTION_KEY")
    fernet = Fernet(key.encode())

    # Load encrypted matrices and decrypt
    with open("DATA/encrypted_matrices.p", "rb") as file:
        encrypted_matrices = pickle.load(file)
    matrices = pickle.loads(fernet.decrypt(encrypted_matrices))

    # Load encrypted subject info and decrypt
    with open("DATA/encrypted_subject_info.csv", "rb") as file:
        encrypted_subject_info = file.read()

    decrypted_data = fernet.decrypt(encrypted_subject_info)
    subject_info = pd.read_csv(BytesIO(decrypted_data))

    return matrices, subject_info


@st.cache_data
def generate_matrix_figure(matrices, i):
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(matrices[i - 1], cmap="bwr")
    plt.axis("off")
    plt.title(f"Matrix {i}")
    return fig


@st.cache_resource
def load_model():
    # Load model
    with open("DATA/model_2_lr_with_pca.p", "rb") as file:
        model = pickle.load(file)
    return model


@st.cache_data
def preprocess_data(matrices, subject_info):
    n = matrices.shape[0]  # number of matrices
    m = matrices.shape[1]  # number of rows/columns in each matrix
    D = round(m * (m - 1) / 2)  # length of feature vector

    # feature matrix from upper triangular part
    X = np.zeros([n, D])
    for i in range(n):
        index = 0
        for j in range(m):
            for k in range(j):
                X[i, index] = matrices[i, j, k]
                index += 1

    # add subject info
    df = pd.concat([subject_info, pd.DataFrame(X)], axis=1)
    X = df.drop(columns=["prematurity"])
    y = df["prematurity"]

    return X, y


@st.cache_data
def predict_prematurity(_model, X, y, i):
    prediction = _model.predict(X.iloc[i].values.reshape(1, -1))
    return prediction[0], y.iloc[i]


@st.cache_data
def visualise_pca(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.to_numpy())

    fig = plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="bwr")
    plt.title("PCA of the data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Prematurity")

    return fig


@st.cache_data
def visualise_tsne(X, y):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X.to_numpy())

    fig = plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="bwr")
    plt.title("t-SNE of the data")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Prematurity")

    return fig


@st.cache_data
def visualise_umap(X, y):
    umap_model = umap.UMAP()
    X_umap = umap_model.fit_transform(X.to_numpy())

    fig = plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, palette="bwr")
    plt.title("UMAP of the data")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.legend(title="Prematurity")

    return fig


@st.cache_data
def visualise_data_counts(y):
    fig = plt.figure(figsize=(8, 5))
    sns.countplot(x=y, palette="bwr", hue=y)
    plt.title("Distribution of Prematurity")
    plt.xlabel("Prematurity")
    plt.ylabel("Count")

    return fig
