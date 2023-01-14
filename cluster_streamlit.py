import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster  import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go


df = pd.read_csv('mainAndWeather.csv')
feats = ["Laundry_count", "Hour", "TimeSpent_minutes", "Age_Range", "Humidity_percent", "Wind_kmph", "Temp_celsius",
"Basket_colour", "Age_Range", "Laundry_count", "Basket_colour", "Shirt_Colour", "Pants_Colour"]

feat_options = st.multiselect('Top features according to RFE and BORUTA', feats, default=["Shirt_Colour", "Pants_Colour"])

copy_df = df[feat_options].copy()

# Normalization
scaler = MinMaxScaler()
scaler.fit(df[feat_options])
copy_df[feat_options] = scaler.transform(copy_df[feat_options])

cluster_amount = st.radio("Select the cluster number for the K-Means model:", range(2,10), horizontal=True)

# Determining the K number for K-Means
sqr_dist = []
silhoettte_avg = []

for i in range(2,10):
    temp_kmeans = KMeans(n_clusters=i)
    temp_kmeans.fit(df[feat_options])
    cluster_labels = temp_kmeans.fit_predict(df[feat_options])

    sqr_dist.append(temp_kmeans.inertia_)
    silhoettte_avg.append(silhouette_score(df[feat_options], cluster_labels))

cluster_fig, (elbow, silh) = plt.subplots(2, sharex=True)
elbow.plot(range(2,10), sqr_dist)
elbow.set(ylabel='Sum of Squared Distance')

silh.plot(range(2,10), silhoettte_avg)
silh.set(xlabel='Cluster Number', ylabel='Silhouette Score')

st.pyplot(cluster_fig)

# Develop the cluster model
cluster_model = KMeans(n_clusters=cluster_amount)
y_predict = cluster_model.fit_predict(df[feat_options])

df['clusters'] = y_predict
feat_options.append('clusters')

# st.dataframe(df[feat_options].head())


# Displaying Clustering onto a map
cluster_num = st.radio("Select the cluster you want to focus:", range(0, cluster_amount), horizontal=True)

heat = px.density_mapbox(
    data_frame=df[df['clusters'] == cluster_num],
    lat="latitude",
    lon="longitude",
    mapbox_style="stamen-terrain",
    color_continuous_scale= [
                [0.0, "green"],
                [0.5, "green"],
                [0.51111111, "yellow"],
                [0.71111111, "yellow"],
                [0.71111112, "red"],
                [1, "red"]],
    opacity = 0.5,
    zoom=9.2
)


st.plotly_chart(heat)

# Explain what the cluster is
cluster_type = []

for col in feat_options[:-1]:
    if (df[col].dtype == np.int64):
        cluster_type.append(df.loc[df['clusters'] == cluster_num, col].mode()[0])
    else:
        cluster_type.append(df.loc[df['clusters'] == cluster_num, col].mean())

st.write("The cluster represents: ")

for i in range(len(cluster_type)):
    st.write(feat_options[i], ": ", cluster_type[i])
        