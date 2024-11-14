import streamlit as st
from utils import (
    evaluation,
    load_data,
    generate_matrix_figure,
    load_model,
    preprocess_data,
    predict_prematurity,
    visualise_pca,
    visualise_tsne,
    visualise_umap,
    visualise_data_counts,
    plot_confusion_matrices,
)


def main():
    # Load data
    matrices, subject_info = load_data()

    # Prepare data
    X, y = preprocess_data(matrices, subject_info)

    # Load models
    model = load_model()

    st.title("Prematurity Prediction Based on Brain Connectivity Matrices")
    st.divider()
    project_description()
    st.divider()
    understand_prematurity()

    tab1, tab2, tab3 = st.tabs(
        ["Predict Prematurity", "Visualise Data", "Model Evaluation"]
    )
    with tab1:
        plot_and_predict(matrices, model, X, y)
    with tab2:
        visualisaton_section(X, y)
    with tab3:
        metrics_section(model, X, y)


def model_explanation():
    st.subheader("Model Explanation")
    st.write(
        """
    To counter overfitting, this model integrates **Principal Component Analysis (PCA)** for dimensionality reduction before applying **Logistic Regression**. This approach introduces several advantages:
    """
    )

    st.markdown("1. **Dimensionality Reduction with PCA**")
    st.write(
        """
    This model uses PCA to reduce the number of features from 4000+ to 50. This reduction helps prevent overfitting and improves the model’s ability to generalize to new data.
    """
    )

    st.markdown("2. **Logistic Regression with Balanced Class Weights**")
    st.write(
        """
    Similar to prior attempts, this model applies `class_weight='balanced'` in Logistic Regression. This adjustment improves the model’s capability to handle imbalanced data effectively.
    """
    )

    st.markdown("3. **Hyperparameter Tuning**")
    st.write(
        """
    The GridSearchCV method is used to evaluate different values of the regularization parameter `C` for logistic regression, along with various `n_components` values for PCA. Tuning these parameters helps optimize the model, finding the best balance between underfitting and overfitting.
    """
    )

    st.markdown("4. **Logistic Regression with 'saga' Solver**")
    st.write(
        """
    This model uses the 'saga' solver for logistic regression, which is well-suited for large datasets and improves computational efficiency.
    """
    )


def metrics_section(model, X, y):
    st.subheader("Model Performance on Train and Test Data")
    col1, col2 = st.columns(2)
    with col1:
        display_metrics(evaluation(model, X, y, mode="Train"))
    with col2:
        display_metrics(evaluation(model, X, y, mode="Test"))
    st.divider()
    st.subheader("Confusion Matrices")
    st.pyplot(plot_confusion_matrices(model, X, y))
    st.divider()
    model_explanation()


# Displaying the cached metrics in Streamlit
def display_metrics(metrics):
    for metric, value in metrics.items():
        st.write(f"**{metric}:** {value}")


def visualisation_info():
    st.write(
        """
    - **PCA (Principal Component Analysis)**: A linear dimensionality reduction technique that projects data onto a lower-dimensional space to maximize variance. Useful for gaining insights while preserving global structure.
    - **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: A non-linear technique that emphasizes local structure, effectively clustering similar points together but may distort global structure. Good for visualizing clusters.
    - **UMAP (Uniform Manifold Approximation and Projection)**: Another non-linear method that preserves both local and some global structure, providing a balance between clustering and data spread. Useful for capturing more complex relationships in high-dimensional data.
    """
    )


def choose_visualisation(X, y):
    visualisation = st.selectbox(
        "Select a visualisation", ["PCA", "t-SNE", "UMAP", "Data Counts"]
    )
    if visualisation == "PCA":
        return visualise_pca(X, y)
    elif visualisation == "t-SNE":
        return visualise_tsne(X, y)
    elif visualisation == "UMAP":
        return visualise_umap(X, y)
    elif visualisation == "Data Counts":
        return visualise_data_counts(y)


def visualisaton_section(X, y):
    st.subheader("Visualisation of the data")
    st.pyplot(choose_visualisation(X, y))
    visualisation_info()


def project_description():
    st.subheader("Project Overview")
    st.write(
        """
    This application predicts whether a baby is preterm or full-term based on brain connectivity matrices. 
    Brain connectivity matrices represent patterns of neural connections in the brain, which can provide insights into neurological development.
    In this project, machine learning models analyze these matrices to classify the prematurity status of a baby.
    """
    )


def understand_prematurity():
    # Explanation of preterm vs full-term
    st.markdown("### **Understanding Prematurity**")
    st.write(
        "The primary distinction between preterm and full-term birth is the gestational age, or how long the baby was in the womb at birth."
    )
    st.markdown(
        """
        - **Premature:** born before 37 weeks
        - **Full term:** born between 37 and 42 weeks
        """
    )


def plot_and_predict(matrices, model, X, y):
    # Plot matrices
    if matrices is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Connectivity Matrices")
            matrix_index = st.number_input(
                "Select a connectivity matrix",
                min_value=0,
                max_value=matrices.shape[0] - 1,
                value=1,
            )
            st.pyplot(generate_matrix_figure(matrices, matrix_index))

        with col2:
            st.subheader("Classification Results")

            # Prediction outcome in a highlighted box
            y_pred, y_true = predict_prematurity(model, X, y, matrix_index)
            outcome = "Premature" if y_pred == 1 else "Full term"
            true_status = "Premature" if y_true == 1 else "Full term"

            st.markdown(f"### **Prediction Output:**")
            st.info(f"**{outcome}**", icon="🪄")

            st.markdown(f"### **True Prematurity Status:**")
            st.success(f"**{true_status}**", icon="📅")

    else:
        st.error("Failed to load matrices.")


if __name__ == "__main__":
    main()
