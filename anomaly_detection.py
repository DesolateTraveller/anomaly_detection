#---------------------------------------------------------------------------------------------------------------------------------
### Authenticator
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
#---------------------------------------------------------------------------------------------------------------------------------
### Import Libraries
#---------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#----------------------------------------
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
#----------------------------------------
from scipy.stats import zscore
#----------------------------------------
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.gmm import GMM
from pyod.models.cblof import CBLOF
from pyod.models.xgbod import XGBOD
#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
#import custom_style()
st.set_page_config(page_title="Anomaly Detection",
                   layout="wide",
                   #page_icon=               
                   initial_sidebar_state="collapsed")
#----------------------------------------
st.title(f""":rainbow[Anomaly Detection | v0.1]""")
st.markdown('Created by | <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a>', 
            unsafe_allow_html=True)
#t.info('**Disclaimer : :blue[Thank you for visiting the app] | Unauthorized uses or copying of the app is strictly prohibited | Click the :blue[sidebar] to follow the instructions to start the applications.**', icon="ℹ️")
#----------------------------------------
# Set the background image
st.divider()

#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------

@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

@st.cache_data(ttl="2h")
def detect_anomalies_zscore(df, feature_column, threshold=3):
    z_scores = np.abs(zscore(df[feature_column].dropna()))
    anomaly_mask = z_scores > threshold
    anomalies = df[anomaly_mask]
    return anomalies

@st.cache_data(ttl="2h")
def detect_anomalies_isolation_forest(df, feature_column, n_estimators, contamination):
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination)
    model.fit(df[[feature_column]].dropna())
    anomalies = df[model.predict(df[[feature_column]].dropna()) == -1]
    return anomalies

@st.cache_data(ttl="2h")
def detect_anomalies_dbscan(df, feature_column, eps, min_samples):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[[feature_column]].dropna())
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['anomaly_db'] = dbscan.fit_predict(df_scaled)
    anomalies = df[df['anomaly_db'] == -1]
    return anomalies

@st.cache_data(ttl="2h")
def detect_anomalies_lof(df, feature_column, n_neighbors):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05)
    df['anomaly_lof'] = lof.fit_predict(df[[feature_column]].dropna())
    anomalies = df[df['anomaly_lof'] == -1]
    return anomalies

@st.cache_data(ttl="2h")
def detect_anomalies_ecod(df, feature_column):
    model = ECOD()
    model.fit(df[[feature_column]].dropna())
    anomalies = df[model.labels_ == 1]
    return anomalies

@st.cache_data(ttl="2h")
def detect_anomalies_hbos(df, feature_column):
    model = HBOS()
    model.fit(df[[feature_column]].dropna())
    anomalies = df[model.labels_ == 1]
    return anomalies

@st.cache_data(ttl="2h")
def detect_anomalies_gmm(df, feature_column):
    model = GMM()
    model.fit(df[[feature_column]].dropna())
    anomalies = df[model.labels_ == 1]
    return anomalies

@st.cache_data(ttl="2h")
def detect_anomalies_ocsvm(df, feature_column):
    model = OneClassSVM(gamma='auto')
    model.fit(df[[feature_column]].dropna())
    anomalies = df[model.predict(df[[feature_column]].dropna()) == -1]
    return anomalies

@st.cache_data(ttl="2h")
def detect_anomalies_cblof(df, feature_column):
    model = CBLOF()
    model.fit(df[[feature_column]].dropna())
    anomalies = df[model.labels_ == 1]
    return anomalies

@st.cache_data(ttl="2h")
def detect_anomalies_xgbod(df, feature_column):
    model = XGBOD()
    model.fit(df[[feature_column]].dropna())
    anomalies = df[model.labels_ == 1]
    return anomalies

@st.cache_data(ttl="2h")
def get_numerical_columns(df):
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    return numerical_cols
#---------------------------------------------------------------------------------------------------------------------------------
### Main App
#---------------------------------------------------------------------------------------------------------------------------------

uploaded_file = st.file_uploader("**:blue[Choose a file]**",type=["csv", "xls", "xlsx"], accept_multiple_files=False, key="file_upload")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    stats_expander = st.expander("**Preview of Information**", expanded=True)
    with stats_expander:  
        st.table(df.head(2))
    st.divider()

    numerical_columns = get_numerical_columns(df)
    if not numerical_columns:
        st.warning("No numerical columns found in the uploaded file.")

    else:
        col1, col2 = st.columns((0.15,0.85))

        with col1:

            st.subheader("Methods", divider='blue')    

            target_variable = st.selectbox("Target variable for anomaly detection", numerical_columns)
            #target_variable = st.selectbox("target variable for anomaly detection", df.columns)

            ad_det_type = st.selectbox("Anomaly Detection Method", [
                                    "Isolation Forest",
                                    "Z-score",
                                    "DBSCAN",
                                    "Local Outlier factor (LOF)",
                                    "ECOD",
                                    "HBOS",
                                    "GMM",
                                    "OCSVM",
                                    "CBLOF",
                                    "XGBOD"
                                    ])
            st.divider()

            if ad_det_type == "Z-score":
                st.subheader("Parameters", divider='blue')    
                zscore_threshold = st.slider("Z-score Threshold", min_value=1, max_value=10, value=3)
                anomalies = detect_anomalies_zscore(df, target_variable, threshold=zscore_threshold)
    
            elif ad_det_type == "Isolation Forest":
                st.subheader("Parameters", divider='blue')  
                n_estimators = st.number_input("Number of trees in the forest", 100, 5000, step=10, key='n_estimators_ad')
                contamination = st.number_input("Proportion of outliers in the data set", 0.0, 0.1, 0.05, step=0.01, key='contamination_ad')
                anomalies = detect_anomalies_isolation_forest(df, target_variable, n_estimators, contamination)

            elif ad_det_type == "DBSCAN":
                st.subheader("Parameters", divider='blue')  
                eps = st.slider("DBSCAN eps", 0.1, 10.0, 0.5)
                min_samples = st.slider("DBSCAN min_samples", 1, 50, 5)
                anomalies = detect_anomalies_dbscan(df, target_variable, eps, min_samples)

            elif ad_det_type == "Local Outlier factor (LOF)":
                st.subheader("Parameters", divider='blue')  
                n_neighbors = st.slider("LOF n_neighbors", 1, 50, 20)
                anomalies = detect_anomalies_lof(df, target_variable, n_neighbors)

            elif ad_det_type == "ECOD":
                anomalies = detect_anomalies_ecod(df, target_variable)

            elif ad_det_type == "HBOS":
                anomalies = detect_anomalies_hbos(df, target_variable)

            elif ad_det_type == "GMM":
                anomalies = detect_anomalies_gmm(df, target_variable)

            elif ad_det_type == "OCSVM":
                anomalies = detect_anomalies_ocsvm(df, target_variable)

            elif ad_det_type == "CBLOF":
                anomalies = detect_anomalies_cblof(df, target_variable)

            elif ad_det_type == "XGBOD":
                anomalies = detect_anomalies_xgbod(df, target_variable)

            with col2:

                st.subheader("Output", divider='blue')    

                st.warning("#### Anomalies Detected:")
                st.table(anomalies.head())

                st.subheader("Visualizations", divider='blue') 
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.lineplot(data=df, x=df.index, y=target_variable, label='Original Data', color='blue', ax=ax)
                sns.scatterplot(data=anomalies, x=anomalies.index, y=target_variable, color='red', label='Anomalies', ax=ax)
                ax.set_title('Anomaly Detection')
                ax.set_xlabel('Index')
                ax.set_ylabel(target_variable)
                ax.legend()
                plt.xticks(rotation=45)
                sns.despine()
                st.pyplot(fig, use_container_width=True)



