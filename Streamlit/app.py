import streamlit as st
from utils import (
    load_data,
    generate_matrix_figure,
    load_model,
    preprocess_data,
    predict_prematurity,
)


def main():
    # Load data
    matrices, subject_info = load_data()

    # Load model
    model = load_model()

    # Prepare data
    X, y = preprocess_data(matrices, subject_info)

    st.title("Prematurity Prediction Based on Brain Connectivity Matrices")
    st.divider()
    project_description()
    st.divider()
    understand_prematurity()
    st.divider()
    plot_and_predict(matrices, model, X, y)


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
                min_value=1,
                max_value=matrices.shape[0],
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
