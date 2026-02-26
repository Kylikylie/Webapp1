
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Product Image kNN Predictor", layout="wide")
st.title("Product Image Classification Web App")
st.write("Upload labeled and unlabeled datasets to generate predictions.")

# =========================
# Parameters (UI adjustable)
# =========================
st.sidebar.header("Model Parameters")

CELL = st.sidebar.slider("HOG Cell Size", 2, 7, 4)
BINS = st.sidebar.slider("HOG Bins", 6, 12, 9)
PCA_COMPONENTS = st.sidebar.slider("PCA Components", 50, 200, 140)
K = st.sidebar.slider("k (Neighbours)", 5, 41, 21, step=2)
BATCH_SIZE = 200
EPS = 1e-9

# =========================
# Upload Files
# =========================
lab_file = st.file_uploader("Upload Labeled CSV", type=["csv"])
unlab_file = st.file_uploader("Upload Unlabeled CSV", type=["csv"])

# =========================
# Utility Functions
# =========================
def standardize_per_image(X):
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True)
    return (X - m) / (s + 1e-6)

def l2_normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + EPS)

def pca_fit_transform(X_train, X_test, n_components):
    mean = X_train.mean(axis=0, keepdims=True)
    Xc = (X_train - mean).astype(np.float64)
    Xt = (X_test - mean).astype(np.float64)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:n_components].T.astype(np.float32)
    return (Xc @ W).astype(np.float32), (Xt @ W).astype(np.float32)

def knn_predict_cosine_weighted(X_train, y_train, X_test, k=21, batch=200):
    X_train = l2_normalize(X_train)
    X_test = l2_normalize(X_test)
    X_train_T = X_train.T

    preds = []
    for start in range(0, X_test.shape[0], batch):
        end = min(start + batch, X_test.shape[0])
        sims = X_test[start:end] @ X_train_T
        idx = np.argpartition(sims, kth=sims.shape[1]-k, axis=1)[:, -k:]
        top_sims = np.take_along_axis(sims, idx, axis=1)
        top_labels = y_train[idx]
        w = np.maximum(top_sims, 0.0) + 1e-6

        votes = np.zeros((end-start, 10))
        rows = np.arange(end-start)
        for j in range(k):
            votes[rows, top_labels[:, j]] += w[:, j]

        preds.append(np.argmax(votes, axis=1))

    return np.concatenate(preds)

# =========================
# Run Model
# =========================
if lab_file and unlab_file:

    if st.button("Run Model"):

        with st.spinner("Processing... This may take a few minutes."):

            df_lab = pd.read_csv(lab_file)
            df_unlab = pd.read_csv(unlab_file)

            X_train = df_lab.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
            y_train = df_lab["label"].to_numpy(dtype=np.int32)
            X_test = df_unlab.to_numpy(dtype=np.float32) / 255.0

            # Standardize
            X_train = standardize_per_image(X_train)
            X_test = standardize_per_image(X_test)

            # PCA
            X_train_p, X_test_p = pca_fit_transform(X_train, X_test, PCA_COMPONENTS)

            # kNN
            preds = knn_predict_cosine_weighted(X_train_p, y_train, X_test_p, k=K)

            df_out = pd.DataFrame({"Predicted_Label": preds.astype(int)})

        st.success("Prediction Complete")
        st.dataframe(df_out.head())

        st.download_button(
            label="Download Submission CSV",
            data=df_out.to_csv(index=False),
            file_name="submission.csv",
            mime="text/csv"
        )
