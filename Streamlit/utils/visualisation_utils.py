import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import umap.umap_ as umap
import streamlit as st


@st.cache_data
def generate_matrix_figure(matrices, i):
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(matrices[i - 1], cmap="bwr")
    plt.axis("off")
    plt.title(f"Matrix {i}")
    return fig


@st.cache_data
def plot_confusion_matrices(_model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_train_pred = _model.predict(X_train.to_numpy())
    y_test_pred = _model.predict(X_test.to_numpy())
    y_all_pred = _model.predict(X.to_numpy())

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    cm_all = confusion_matrix(y, y_all_pred)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    disp_train = ConfusionMatrixDisplay(
        confusion_matrix=cm_train, display_labels=["Term", "Preterm"]
    )
    disp_train.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Train Confusion Matrix")

    disp_test = ConfusionMatrixDisplay(
        confusion_matrix=cm_test, display_labels=["Term", "Preterm"]
    )
    disp_test.plot(ax=axes[1], cmap="Blues", colorbar=False)
    axes[1].set_title("Test Confusion Matrix")

    disp_all = ConfusionMatrixDisplay(
        confusion_matrix=cm_all, display_labels=["Term", "Preterm"]
    )
    disp_all.plot(ax=axes[2], cmap="Blues", colorbar=False)
    axes[2].set_title("All Data Confusion Matrix")

    for ax in axes:
        ax.grid(False)

    plt.tight_layout()
    return fig


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
