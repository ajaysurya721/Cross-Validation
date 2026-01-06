import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# -------------------------------------------------
# Page settings
# -------------------------------------------------
st.set_page_config(page_title="Hierarchical Clustering App", layout="centered")

st.title("Hierarchical Clustering Using Streamlit")
st.write("Upload a CSV file to perform Hierarchical Clustering")

# -------------------------------------------------
# Upload CSV
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------------------------
    # Select numeric columns
    # -------------------------------------------------
    X = df.select_dtypes(include=["int64", "float64"])

    if X.shape[1] < 2:
        st.error("Dataset must contain at least two numeric columns.")
    else:
        # -------------------------------------------------
        # Scaling
        # -------------------------------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # -------------------------------------------------
        # Dendrogram
        # -------------------------------------------------
        st.subheader("Dendrogram")
        linked = linkage(X_scaled, method="ward")

        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(linked, ax=ax)
        ax.set_xlabel("Data Points")
        ax.set_ylabel("Distance")
        st.pyplot(fig)

        # -------------------------------------------------
        # Cluster selection
        # -------------------------------------------------
        k = st.slider("Select number of clusters", 2, 6, 4)

        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward"
        )

        clusters = model.fit_predict(X_scaled)
        df["Cluster"] = clusters

        # -------------------------------------------------
        # Clustered data
        # -------------------------------------------------
        st.subheader("Clustered Dataset")
        st.dataframe(df)

        # -------------------------------------------------
        # Visualization
        # -------------------------------------------------
        st.subheader("Cluster Visualization")

        col1, col2 = X.columns[0], X.columns[1]

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.scatter(
            df[col1],
            df[col2],
            c=df["Cluster"]
        )
        ax2.set_xlabel(col1)
        ax2.set_ylabel(col2)
        ax2.set_title("Hierarchical Clustering Result")
        st.pyplot(fig2)

        # -------------------------------------------------
        # Cluster summary
        # -------------------------------------------------
        st.subheader("Cluster-wise Mean Analysis")
        st.dataframe(df.groupby("Cluster").mean())
