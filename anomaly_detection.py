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
from sklearn.cluster import DBSCAN
from scipy.stats import zscore
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
def load_file(file):
    file_extension = file.name.split('.')[-1]
    if file_extension == 'csv':
        df = pd.read_csv(file, sep=None, engine='python', encoding='utf-8', parse_dates=True, infer_datetime_format=True)
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format")
        df = pd.DataFrame()
    return df

@st.cache_data(ttl="2h")
def anomalies_if(df, n_estimators, contamination):
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination)
    model.fit(df)
    anomalies = df[model.predict(df) == -1]
    return anomalies
#---------------------------------------------------------------------------------------------------------------------------------
### Main App
#---------------------------------------------------------------------------------------------------------------------------------

file = st.file_uploader("**:blue[Choose a file]**",type=["csv", "xls", "xlsx"], accept_multiple_files=False, key="file_upload")
if file:
    df = load_file(file)
    st.divider()

    col1, col2 = st.columns((0.3,0.7))
    
    with col1:

        target_variable = st.selectbox("**Target (Dependent) Variable**", df.columns)
        if target_variable:
            
            st.subheader("Method & Parameters", divider='blue')
            ad_type = st.selectbox("**Select an Anomaly Detection Method**", [
                                    "Isolation Forest",
                                    "Z-score",
                                    "DBSCAN",
                                    "Local Outlier Factor (LOF)"])
        
            st.divider()

            if ad_type == "Isolation Forest":

                n_estimators = st.number_input("**The number of trees in the forest**", 100, 5000, step=10, key='n_estimators_ad')
                contamination = st.number_input("**The proportion of outliers in the data set**", 0.0, 0.1, 0.05, step=0.01, key='contamination_ad')

                with col2:

                    st.subheader("Result & Visualizations", divider='blue')

                    anomalies = anomalies_if(df[target_variable], n_estimators, contamination)
                    st.warning("#### Anomalies Detected:")
                    st.table(anomalies.head())

                    st.divider()


