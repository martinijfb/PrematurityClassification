from io import BytesIO
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from cryptography.fernet import Fernet

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
    plt.imshow(matrices[i], cmap="bwr")
    plt.axis("off")
    plt.title(f"Matrix {i + 1}")
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
    m = matrices.shape[1] # number of rows/columns in each matrix
    D = round(m*(m-1)/2) # length of feature vector

    # feature matrix from upper triangular part
    X=np.zeros([n,D])
    for i in range(n):
        index=0
        for j in range(m):
            for k in range(j):
                X[i,index] = matrices[i,j,k]
                index += 1
    
    # add subject info
    df = pd.concat([subject_info, pd.DataFrame(X)], axis=1)
    X = df.drop(columns=["prematurity"])
    y = df['prematurity']

    return X, y

@st.cache_data
def predict_prematurity(_model, X, y, i):
    prediction = _model.predict(X.iloc[i].values.reshape(1, -1))
    return prediction[0], y.iloc[i]