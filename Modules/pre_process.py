# %matplotlib ipympl   
# %matplotlib widget     
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR, SVC
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier,BalancedBaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.datasets import make_regression,make_classification
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,IterativeImputer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.linear_model import LinearRegression,LogisticRegression,SGDRegressor,Ridge,Lasso,ElasticNet
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,median_absolute_error,accuracy_score, precision_score, recall_score,f1_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,GradientBoostingClassifier,HistGradientBoostingRegressor,BaggingRegressor, VotingRegressor, BaggingClassifier, RandomForestClassifier,AdaBoostClassifier,VotingClassifier,AdaBoostRegressor
from sklearn.cluster import KMeans,HDBSCAN,AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, OneHotEncoder,PolynomialFeatures, FunctionTransformer,PowerTransformer,LabelEncoder,MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor # type: ignore
import scipy.cluster.hierarchy as shc
class pre_process:
    def __init__(self,data,target,cap=False,trim=False,format='csv'):
        self.data=data
        self.cap=cap
        self.trim=trim
        self.format=format
        if 'Unnamed: 0' in self.data.columns:
            self.data.drop(columns='Unnamed: 0', inplace=True)
        self.target=target
        if isinstance(self.data[self.target].iloc[0], str):
            enc = LabelEncoder()
            self.data[self.target] = enc.fit_transform(self.data[self.target])
        self.num_col = [col for col in self.data.select_dtypes(include=[np.number]).columns if col != target]
        self.cat_col = [col for col in self.data.select_dtypes(exclude=[np.number]).columns if col != target]
        self.num_missing_mean=[]
        self.num_missing_median=[]
        self.cat_missing=[]
    def outliers(self):
        for i in self.num_col:
            val=self.data[i].skew()
            lower_range,upper_range=int(self.data[i].mean()-(2*self.data[i].std())),int(self.data[i].mean()+(2*self.data[i].std()))
            Q1,Q3 = self.data[i].quantile(0.25),self.data[i].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            if val>=-0.2 and val<=0.2:
                if self.cap:
                    self.data.loc[self.data[i]<lower_range,i]=lower_range
                    self.data.loc[self.data[i]>upper_range,i]=upper_range
                elif self.trim:
                    self.data.drop(self.data[self.data[i]<lower_range].index,inplace=True,axis=0)
                    self.data.drop(self.data[self.data[i]>upper_range].index,inplace=True,axis=0)
            else:
                if self.cap:
                    self.data.loc[self.data[i]<lower_bound,i]=lower_bound
                    self.data.loc[self.data[i]>upper_bound,i]=upper_bound
                elif self.trim:
                    self.data.drop(self.data[self.data[i]<lower_bound].index,inplace=True,axis=0)
                    self.data.drop(self.data[self.data[i]>upper_bound].index,inplace=True,axis=0)
    def missingvalues(self):
        mean_filled=SimpleImputer(strategy='mean')
        median_filled=SimpleImputer(strategy='median')
        for i in self.num_col:
            val=self.data[i].skew()
            if val>=-0.2 and val<=0.2:
                self.num_missing_mean.append(i)
                self.data[i]=mean_filled.fit_transform(self.data[i].values.reshape(-1,1))
            else:
                self.num_missing_median.append(i)
                self.data[i]=median_filled.fit_transform(self.data[i].values.reshape(-1,1))
        for i in self.cat_col:
            if self.data[i].isna().sum()>=0:
                self.cat_missing.append(i)
                mode_filled=SimpleImputer(strategy='most_frequent')
                self.data[i]=mode_filled.fit_transform(self.data[i].values.reshape(-1,1)).ravel()
    def transform(self):
        trans=PowerTransformer(method='yeo-johnson')
        for i in self.num_col:
            val=self.data[i].skew()
            if val>=0:
                self.data[i]=trans.fit_transform(self.data[i].values.reshape(-1,1))
    def scaling_encoding(self):
        scaler=StandardScaler()
        encoder=OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False)
        for i in self.num_col:
            self.data[i]=scaler.fit_transform(self.data[i].values.reshape(-1,1))
        for i in self.cat_col:
            encoded=encoder.fit_transform(self.data[[i]])
            encoded_cols = [f'{i}_{j}' for j in range(encoded.shape[1])]
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=self.data.index)
            self.data = self.data.drop(i, axis=1)
            self.data = pd.concat([self.data, encoded_df], axis=1)
    def save(self,path):
        if self.format=="json":
            self.data.to_json(path)
        elif self.format=="csv":
            self.data.to_csv(path)
        elif self.format=="xml":
            self.data.to_xml(path)
    