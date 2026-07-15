import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# App header
st.title("Business Location Explorer")
st.subheader("Lab Task 05")
st.write("Submission for Atif M. Mahmud")
st.write("SFU ID: atifm@sfu.ca")


# TODO: cache a function and data
# Not now, maybe later

## Create a dataframe from data source
def load_data(path="business_locations.geojson"):
    with open(path) as f:
        geojson = json.load(f)
    rows = [] # create an array
    for feat in geojson["features"]:
        props = feat["properties"]
        lon, lat = feat["geometry"]["coordinates"]
        rows.append({**props, "lon":lon, "lat":lat})
    return pd.DataFrame(rows)

df = load_data()

st.header("The data source")
st.dataframe(df)

# Click to expand box thingy
with st.expander("Descriptive statistics on our data:"):
    st.dataframe(df.describe())
    st.write(f"{len(df)} locations")

st.subheader("These are the columns in our source")
st.write(df.columns)

st.sidebar.header("1. Select Features")

# Do K-means clustering on these numeric variables
NUMERIC_COLS = ["Floor_Area_sqm", "Daily_Foot_Traffic", "Community_Impact_Score", "Annual_Revenue_k"]
selected_features = st.sidebar.multiselect("Features to use in Map", options=NUMERIC_COLS)

if (len(selected_features) < 2):
    st.warning("Select at least 2 features")
    st.stop()

st.sidebar.header("2. Cluster")
algo = st.sidebar.selectbox("Algorithm", ["KMeans", "DBScan"])

X = df[selected_features].to_numpy()
X_scaled = StandardScaler().fit_transform(X)

labels = []
if algo == "KMeans":
    k = st.sidebar.slider("Number of clusters", 2, 15)
    model = KMeans(n_clusters=k)
    labels = model.fit_predict(X_scaled)

### =========================================================================
## THIS IS LAB TASK SUBMISSION
### =========================================================================

elif algo == "DBScan":
    epsilon = st.sidebar.slider("Radius of cluster", 0.01, 1.0)
    n_points = st.sidebar.slider("Number of points in cluster", 1, 20)
    model = DBSCAN(eps=epsilon, min_samples=n_points)
    labels = model.fit_predict(X_scaled)

### =========================================================================
###
###==========================================================================

df["cluster"] = pd.Categorical(labels.astype(str))
n_clusters_found = df["cluster"].nunique()
st.metric("Number of clusters:", n_clusters_found)

map_tab, dr_tab = st.tabs(["Map", "Dimensionality Reduction"])

with map_tab:
    st.write("I'm a map I'm a map....")
    fig = px.scatter_map(df, lat="lat", lon="lon", zoom=10, height=550, map_style="carto-darkmatter", color="cluster")
    st.plotly_chart(fig, width="stretch")

# Dimensionality Reduction
with dr_tab:
    reducer = PCA(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    df["dim_1"] = embedding[:,0]
    df["dim_2"] = embedding[:,1]
    
    fig_dr = px.scatter(
        df,
        x="dim_1",
        y="dim_2",
        color="cluster"
    )
st.plotly_chart(fig_dr, width="stretch")
