import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster  import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import plotly.express as px

df = pd.read_csv('mainAndWeather.csv')
cluster_choice = st.radio("Select a cluster method: ", ["By location", "By features"], horizontal=True)
# List of features selected by Boruta and RFE
feats = ["Laundry_count", "Hour", "TimeSpent_minutes", "Age_Range", "Humidity_percent", "Wind_kmph", "Temp_celsius",
 "Basket_colour", "Age_Range", "Basket_colour", "Shirt_Colour", "Pants_Colour"]


def normalization(df, cols):
    copy = df.copy()
    scaler = MinMaxScaler()
    scaler.fit(copy[cols])
    copy[cols] = scaler.transform(copy[cols])

    return copy[cols]



# Explain what the cluster represents, the mean/mode of its features
def cluster_explain(df, feat_options, cluster_num):
    cluster_type = []

    for col in feat_options[:-1]:
        if (df[col].dtype == np.int64):
            cluster_type.append(df.loc[df['cluster'] == cluster_num, col].mode()[0])
        else:
            cluster_type.append(df.loc[df['cluster'] == cluster_num, col].mean())

    st.write("The cluster represents: ")

    for i in range(len(cluster_type)):
        st.write(feat_options[i], ": ", cluster_type[i])
    
    center_point = (df.loc[df['cluster'] == cluster_num, "latitude"].mean(), df.loc[df['cluster'] == cluster_num, "longitude"].mean())
    st.write("The center point of the cluster is at: ", center_point)



# Determining the K number for K-Means
def cluster_number_graph(feat_options):
    # Determining the K number for K-Means
    sqr_dist = []
    silhoettte_avg = []

    for i in range(2,10):
        temp_kmeans = KMeans(n_clusters=i)
        temp_kmeans.fit(df[feat_options])
        cluster_labels = temp_kmeans.fit_predict(df[feat_options])

        sqr_dist.append(temp_kmeans.inertia_)
        silhoettte_avg.append(silhouette_score(df[feat_options], cluster_labels))


    elbow_graph = px.line(x=range(2,10), y=sqr_dist,
                            labels=dict(y="Sum of Squared Distance", x="Number of Clusters"))
    st.plotly_chart(elbow_graph)

    silh_graph = px.line(x=range(2,10), y=silhoettte_avg,
                            labels=dict(y="Silhouette Score", x="Number of Clusters"))
    st.plotly_chart(silh_graph)



# Clusters the dataset based on its features
def cluster_features():
    feat_options = st.multiselect('Top features according to RFE and BORUTA', feats,
                                    default=["Shirt_Colour", "Pants_Colour"], max_selections=3)

    train_df = normalization(df, feat_options)
    
    cluster_amount = st.radio("Select the cluster number for the K-Means model:", range(2,10), horizontal=True)

    # Determining the K number for K-Means
    cluster_number_graph(feat_options)

    # Develop the cluster model
    cluster_model = KMeans(n_clusters=cluster_amount, random_state=1)
    y_predict = cluster_model.fit_predict(train_df[feat_options])

    df['cluster'] = y_predict
    feat_options.append('cluster')


    # Display the feature clusters
    cluster_num = st.radio("Select the cluster to focus: ", range(0, cluster_amount), horizontal=True)

    if len(feat_options) == 3:
        feature_cluster = px.scatter(data_frame=df[df["cluster"] == cluster_num], x=feat_options[0], y=feat_options[1])
        st.plotly_chart(feature_cluster)

    
    elif len(feat_options) == 4:
        feature_cluster = px.scatter(data_frame=df[df["cluster"] == cluster_num], x=feat_options[0], y=feat_options[1], color=feat_options[2])
        st.plotly_chart(feature_cluster)

    cluster_explain(df, feat_options, cluster_num)



# Clusters the dataset based on lat/long
def cluster_local():
    # List of features selected by Boruta and RFE
    local = ["latitude", "longitude"]

    train_df = normalization(df, local)
    
    # Determining the K number for K-Means
    cluster_number_graph(local)

    
    st.write("Based on the graphs above, we can determine that 3 is the optimal cluster amount.")

    # Develop the cluster model
    cluster_model = KMeans(n_clusters=3, random_state=1)
    y_predict = cluster_model.fit_predict(train_df[local])

    df['cluster'] = y_predict
    feats.append('cluster')

    # Display the feature clusters
    cluster_num = st.radio("Select the cluster to focus: ", range(0, 3), horizontal=True)

    heat = px.density_mapbox(
        data_frame=df[df["cluster"] == cluster_num],
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

    cluster_explain(df, feats, cluster_num)


# Main function
if __name__ == "__main__":
    if cluster_choice == "By location":
        cluster_local()

    else:
        cluster_features()
