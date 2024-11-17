import os
import pickle
from io import BytesIO
import pandas as pd
import numpy as np
from cryptography.fernet import Fernet
import streamlit as st

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
