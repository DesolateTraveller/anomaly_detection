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
from xgboost.sklearn import XGBClassifier
#----------------------------------------
from io import BytesIO
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
st.set_page_config(page_title="Anomaly Detection | v0.3",
                   layout="wide",
                   page_icon= "📈",             
                   initial_sidebar_state="collapsed")
#----------------------------------------
st.markdown(
    """
    <style>
    .title-large {
        text-align: center;
        font-size: 35px;
        font-weight: bold;
        background: linear-gradient(to left, red, orange, blue, indigo, violet);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .title-small {
        text-align: center;
        font-size: 20px;
        background: linear-gradient(to left, red, orange, blue, indigo, violet);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    <div class="title-large">Anomaly Detection</div>
    <div class="title-small">Play with Abnormalities | v0.3</div>
    """,
    unsafe_allow_html=True
)
#----------------------------------------
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #F0F2F6;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #333;
        z-index: 100;
    }
    .footer p {
        margin: 0;
    }
    .footer .highlight {
        font-weight: bold;
        color: blue;
    }
    </style>

    <div class="footer">
        <p>© 2025 | Created by : <span class="highlight">Avijit Chakraborty</span> | <a href="mailto:avijit.mba18@gmail.com"> 📩 </a></p>  <span class="highlight">Thank you for visiting the app | Unauthorized uses or copying is strictly prohibited | For best view of the app, please zoom out the browser to 75%.</span>
    </div>
    """,
    unsafe_allow_html=True)
#----------------------------------------
st.markdown(
            """
            <style>
                .centered-info {
                display: flex;
                justify-content: center;
                align-items: center;
                font-weight: bold;
                font-size: 15px;
                color: #007BFF; 
                padding: 5px;
                background-color: #E8F4FF; 
                border-radius: 5px;
                border: 1px solid #007BFF;
                margin-top: 5px;
                margin-bottom: 5px;
                }
            </style>
            """,unsafe_allow_html=True,)
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

@st.cache_data(ttl="2h")
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')
#---------------------------------------------------------------------------------------------------------------------------------
### Main App
#---------------------------------------------------------------------------------------------------------------------------------

# Knowledge Database
with st.popover("**:red[📚 Knowledge Database: Anomaly Detection Methods]**", disabled=False, use_container_width=True): 
#with st.expander("**:blue[📚 Knowledge Database: Anomaly Detection Methods]**", expanded=False):
    st.markdown("""
    <style>
    .info-container {
        padding: 20px;
        background-color: #f9f9f9;
        border-left: 6px solid #3498db;
        border-radius: 5px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .info-container h3 {
        color: #3498db;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .info-container p {
        color: #333;
        margin: 5px 0;
    }
    .info-container ul {
        list-style-type: none;
        padding: 0;
    }
    .info-container li {
        margin: 10px 0;
        display: flex;
        align-items: center;
    }
    .info-container li:before {
        content: "⭐";
        margin-right: 10px;
        color: #3498db;
        font-size: 1.2em;
    }
    </style>
    <div class="info-container">
        <h4>🛠️ Anomaly Detection Methods</h4>
        <ul>
            <li><strong>Isolation Forest:</strong> A tree-based model that isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.</li>
            <li><strong>Z-score:</strong> Detects anomalies by calculating the Z-score of observations, identifying those that deviate significantly from the mean.</li>
            <li><strong>DBSCAN:</strong> A clustering-based method that groups together closely packed points and marks points in low-density regions as anomalies.</li>
            <li><strong>Local Outlier Factor (LOF):</strong> Detects anomalies by measuring the local density deviation of a given data point with respect to its neighbors.</li>
            <li><strong>Empirical Cumulative Outlier Detection (ECOD):</strong> Uses empirical distribution functions to detect anomalies.</li>
            <li><strong>Histogram-Based Outlier Score (HBOS):</strong> Detects anomalies by creating histograms for the data and identifying points that fall into bins with low densities.</li>
            <li><strong>Gaussian Mixture Models (GMM):</strong> Fits multiple Gaussian distributions to the data and identifies points with low probability densities as anomalies.</li>
            <li><strong>One-Class Support Vector Machine (OCSVM):</strong> A SVM algorithm for anomaly detection that separates the normal data points from outliers by learning a decision function.</li>
            <li><strong>Clustering-Based Local Outlier Factor (CBLOF):</strong> Combines clustering and LOF to detect anomalies by examining the local deviation of data points within clusters.</li>
            <li><strong>Extreme Boosting Based Outlier Detection (XGBOD):</strong> Utilizes extreme gradient boosting techniques to detect anomalies.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
#st.divider()
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------

col1, col2 = st.columns((0.2,0.8))
with col1: 
    with st.container(border=True):

        uploaded_file = st.file_uploader("**:blue[Choose a file]**",type=["csv", "xls", "xlsx"], accept_multiple_files=False, key="file_upload")
        if uploaded_file is not None:
            df = load_data(uploaded_file)

            numerical_columns = get_numerical_columns(df)
            if not numerical_columns:
                st.warning("No numerical columns found in the uploaded file.")
                
            else:    
                
                #st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Methods</span></div>',unsafe_allow_html=True,)
                st.divider()
                target_variable = st.selectbox("**:blue[Target variable (for anomaly detection)]**", numerical_columns)
                st.divider()
                
                st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Methods & Parameters</span></div>',unsafe_allow_html=True,)
                ad_det_type = st.selectbox("", [
                                    "Isolation Forest",
                                    "Z-score",
                                    "DBSCAN",
                                    "Local Outlier factor (LOF)",
                                    "Empirical Cumulative Outlier Detection (ECOD)",
                                    "Histogram Based Outlier Score (HBOS)",
                                    "Gaussian Mixture Models (GMM)",
                                    "One Class Support Vector Machine (OCSVM)",
                                    "Clustering based Local Outlier Factor (CBLOF)",
                                    "Extreme Boosting Based Outlier Detection (XGBOD)"
                                    ])
                                
                if ad_det_type == "Z-score":  
                    zscore_threshold = st.slider("Z-score Threshold", min_value=1, max_value=10, value=3)
                    anomalies = detect_anomalies_zscore(df, target_variable, threshold=zscore_threshold)
    
                elif ad_det_type == "Isolation Forest":
                    n_estimators = st.number_input("Number of trees in the forest", 100, 5000, step=10, key='n_estimators_ad')
                    contamination = st.number_input("Proportion of outliers in the data set", 0.0, 0.1, 0.05, step=0.01, key='contamination_ad')
                    anomalies = detect_anomalies_isolation_forest(df, target_variable, n_estimators, contamination)

                elif ad_det_type == "DBSCAN": 
                    eps = st.slider("DBSCAN eps", 0.1, 10.0, 0.5)
                    min_samples = st.slider("DBSCAN min_samples", 1, 50, 5)
                    anomalies = detect_anomalies_dbscan(df, target_variable, eps, min_samples)

                elif ad_det_type == "Local Outlier factor (LOF)": 
                    n_neighbors = st.slider("LOF n_neighbors", 1, 50, 20)
                    anomalies = detect_anomalies_lof(df, target_variable, n_neighbors)

                elif ad_det_type == "Empirical Cumulative Outlier Detection (ECOD)":
                    anomalies = detect_anomalies_ecod(df, target_variable)

                elif ad_det_type == "Histogram Based Outlier Score (HBOS)":
                    anomalies = detect_anomalies_hbos(df, target_variable)

                elif ad_det_type == "Gaussian Mixture Models (GMM)":
                    anomalies = detect_anomalies_gmm(df, target_variable)

                elif ad_det_type == "One Class Support Vector Machine (OCSVM)":
                    anomalies = detect_anomalies_ocsvm(df, target_variable)

                elif ad_det_type == "Clustering based Local Outlier Factor (CBLOF)":
                    anomalies = detect_anomalies_cblof(df, target_variable)

                elif ad_det_type == "Extreme Boosting Based Outlier Detection (XGBOD)":
                    anomalies = detect_anomalies_xgbod(df, target_variable)
            
                with col2:
                    with st.container(border=True):
                        
                        st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Previews of Basic Informations</span></div>',unsafe_allow_html=True,)
                        st.table(df.head(2))
                        st.write("No of rows before anomaly detection :",df.shape[0], use_container_width=True)
       
                        st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Anomalies</span></div>',unsafe_allow_html=True,)
                        st.warning("#### Anomalies Detected:")
                        st.write("No of rows having anomaly : ",anomalies.shape[0], use_container_width=True)
                        st.table(anomalies.head(3))
                        csv = convert_df_to_csv(anomalies)
                        st.download_button(label="📥 Download Anomalies CSV",data=csv,file_name='anomalies.csv',mime='text/csv')


                        st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Visualizations</span></div>',unsafe_allow_html=True,)
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

