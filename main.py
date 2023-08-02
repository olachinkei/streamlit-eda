#モジュールのインポート
!pip install pandas==1.5.0
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import codecs
import pandas as pd
from matplotlib import pyplot as plt
import sweetviz as sv
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, mean_squared_error, make_scorer, r2_score , mean_absolute_error,mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
from base64 import b64encode


# タイトルを表示

st.set_page_config(layout="centered",  page_title="線形モデル作成ツール")
st.title("GUI De EDA")
st.markdown("created by Keisuke Kamata")
st.markdown("")

st.sidebar.markdown("### 1. データの読み込み")
uploaded_file = st.sidebar.file_uploader("CSVファイルをドラッグ&ドロップ、またはブラウザから選択してください", type='csv', key='train')

if uploaded_file is not None:
    #データの読込み
    df = pd.read_csv(uploaded_file)
    df_0 = df

    #object型をcategory型に変更
    df.loc[:, df.dtypes == 'object'] = df.select_dtypes(['object']).apply(lambda x: x.astype('category'))
    

    #ID・目的変数の選択
    st.sidebar.markdown("### 2. 目的変数の選択")
    target = st.sidebar.selectbox(
        '目的変数を選択してください',
        df.columns
    )

    st.dataframe(df)
    st.markdown(df.shape)

    # EDAの実行
    st.sidebar.markdown("### 3. データの確認")
    if st.sidebar.button('実行する'):

        # check process    
        if df[target].isnull().any() == "True":
            st.write("errorです: 目的変数に欠損が含まれております。欠損が内容にデータを準備してください")
            st.stop()


        if df[target].dtypes != "float64" and df[target].dtypes != "int64":
            try:
                df[target] = df[target].astype("float64")
            except:
                st.write("errorです: 目的変数にカテゴリが含まれている可能性があります。データを確認し、再度データを取り込んでください")
                st.stop()
            df[target] = df[target].astype("float64")

        with st.spinner('実行中...'):            
            feature_config = sv.FeatureConfig(skip=[], force_num = [],force_cat=[],force_text=[])
            my_report = sv.analyze(df, target_feat= target,feat_cfg=feature_config, pairwise_analysis="on")
            my_report.show_html("EDA.html")
            report_file = codecs.open("EDA.html",'r')
            page = report_file.read()
            components.html(page, width=1000,height=1000, scrolling=True)

    
