
import streamlit as st
import numpy as np
import pandas as pd

# =========================
# 1. 页面配置与 UI 设计
# =========================
st.set_page_config(page_title="高级产品图像分类器", layout="wide", page_icon="🛍️")

st.title("🛍️ 图像分类 Web 应用 (HOG + PCA + kNN)")
st.markdown("""
本工具通过 **HOG 特征** 捕捉轮廓，结合 **强度池化** 保留形状信息，最后利用 **PCA 降维** 与 **加权 kNN** 进行分类。
""")

# --- 侧边栏参数调节 ---
st.sidebar.header("⚙️ 特征提取设置")
CELL = st.sidebar.slider("HOG 单元大小 (Cell Size)", 2, 7, 4)
BINS = st.sidebar.slider("HOG 方向梯度柱数 (Bins)", 6, 12, 9)
USE_INTENSITY_POOL = st.sidebar.checkbox("启用强度池化 (Intensity Pooling)", value=True)
POOL = st.sidebar.slider("池化大小 (Pool Size)", 2, 7, 4) if USE_INTENSITY_POOL else 4

st.sidebar.header("🧠 模型参数")
PCA_COMPONENTS = st.sidebar.slider("PCA 主成分数", 50, 250, 140)
K = st.sidebar.slider("k (邻居数量)", 5, 51, 21, step=2)
BATCH_SIZE = 200
EPS = 1e-9

# =========================
# 2. 核心算法逻辑
# =========================

def standardize_per_image(X):
    X = X.astype(np.float32)
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True)
    return (X - m) / (s + 1e-6)

def sobel_gradients(imgs):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    pad = np.pad(imgs, ((0,0),(1,1),(1,1)), mode="edge")
    gx = np.zeros_like(imgs)
    gy = np.zeros_like(imgs)
    for i in range(28):
        for j in range(28):
            patch = pad[:, i:i+3, j:j+3]
            gx[:, i, j] = (patch * kx).sum(axis=(1,2))
            gy[:, i, j] = (patch * ky).sum(axis=(1,2))
    return gx, gy

@st.cache_data(show_spinner="正在提取 HOG 特征 (此过程较慢，完成后将缓存)...")
def get_features(X_raw, cell, bins, use_pool, pool_size):
    X_std = standardize_per_image(X_raw)
    n = X_std.shape[0]
    imgs = X_std.reshape(n, 28, 28)
    
    gx, gy = sobel_gradients(imgs)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    
    ncy, ncx = 28 // cell, 28 // cell
    hog_feat = np.zeros((n, ncy * ncx * bins), dtype=np.float32)
    bin_width = np.pi / bins
    
    idx = 0
    for cy in range(ncy):
        for cx in range(ncx):
            m_cell = mag[:, cy*cell:(cy+1)*cell, cx*cell:(cx+1)*cell]
            a_cell = ang[:, cy*cell:(cy+1)*cell, cx*cell:(cx+1)*cell]
            hist = np.zeros((n, bins), dtype=np.float32)
            for bi in range(bins):
                mask = (a_cell >= bi * bin_width) & (a_cell < (bi+1) * bin_width)
                hist[:, bi] = (m_cell * mask).reshape(n, -1).sum(axis=1)
            hist /= (np.linalg.norm(hist, axis=1, keepdims=True) + 1e-6)
            hog_feat[:, idx:idx+bins] = hist
            idx += bins
            
    if use_pool:
        h = 28 // pool_size
        p_feat = imgs.reshape(n, h, pool_size, h, pool_size).mean(axis=(2,4)).reshape(n, -1)
        return np.concatenate([hog_feat, p_feat], axis=1)
    
    return hog_feat

def pca_fit_transform(X_train, X_test, n_components):
    mean = X_train.mean(axis=0, keepdims=True)
    Xc = (X_train - mean).astype(np.float64)
    Xt = (X_test - mean).astype(np.float64)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:n_components].T.astype(np.float32)
    return (Xc @ W).astype(np.float32), (Xt @ W).astype(np.float32)

def knn_predict(X_train, y_train, X_test, k, batch):
    X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + EPS)
    X_test = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + EPS)
    
    n_test = X_test.shape[0]
    preds = np.empty(n_test, dtype=np.int32)
    
    for start in range(0, n_test, batch):
        end = min(start + batch, n_test)
        sims = X_test[start:end] @ X_train.T
        idx = np.argpartition(sims, kth=sims.shape[1]-k, axis=1)[:, -k:]
        top_sims = np.take_along_axis(sims, idx, axis=1)
        top_labels = y_train[idx]
        
        w = np.maximum(top_sims, 0.0) + 1e-6
        votes = np.zeros((end-start, 10), dtype=np.float32)
        for j in range(k):
            rows = np.arange(end-start)
            votes[rows, top_labels[:, j]] += w[:, j]
        preds[start:end] = np.argmax(votes, axis=1)
    return preds

# =========================
# 3. 文件上传与主程序
# =========================
col1, col2 = st.columns(2)
with col1:
    lab_file = st.file_uploader("1. 上传已标记 CSV (训练集)", type=["csv"])
with col2:
    unlab_file = st.file_uploader("2. 上传待预测 CSV (测试集)", type=["csv"])

if lab_file and unlab_file:
    if st.button("🚀 开始提取特征并运行模型"):
        df_lab = pd.read_csv(lab_file)
        df_unlab = pd.read_csv(unlab_file)
        
        X_train_raw = df_lab.drop(columns=["label"]).to_numpy(dtype=np.float32) / 255.0
        y_train = df_lab["label"].to_numpy(dtype=np.int32)
        X_test_raw = df_unlab.to_numpy(dtype=np.float32) / 255.0
        
        F_train = get_features(X_train_raw, CELL, BINS, USE_INTENSITY_POOL, POOL)
        F_test = get_features(X_test_raw, CELL, BINS, USE_INTENSITY_POOL, POOL)
        
        with st.spinner("执行 PCA 降维与 kNN 搜索..."):
            F_train_p, F_test_p = pca_fit_transform(F_train, F_test, PCA_COMPONENTS)
            preds = knn_predict(F_train_p, y_train, F_test_p, K, BATCH_SIZE)
        
        st.success(f"✅ 处理完成！已生成 {len(preds)} 条预测结果。")
        
        df_out = pd.DataFrame({"Predicted_Label": preds})
        
        c1, c2 = st.columns([1, 2])
        c1.write("结果预览:")
        c1.dataframe(df_out.head(10))
        
        csv_download = df_out.to_csv(index=False).encode('utf-8')
        c2.download_button(
            label="📥 下载预测结果 (submission.csv)",
            data=csv_download,
            file_name="submission.csv",
            mime="text/csv"
        )
else:
    st.info("👋 请在上方上传两个 CSV 文件以开始。")
