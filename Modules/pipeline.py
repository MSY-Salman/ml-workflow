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
class pipeline:
    def __init__(self, data, target, num_missing_mean, num_missing_median, cat_missing, best_model, target_type):
        self.data = data.copy()
        self.target = target
        if 'Unnamed: 0' in self.data.columns:
            self.data.drop(columns='Unnamed: 0', inplace=True)
        self.data.dropna(subset=[self.target],inplace=True)
        self.X = self.data.drop(columns=[target])
        self.y = self.data[target]
        self.num_col = self.X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_col = self.X.select_dtypes(exclude=[np.number]).columns.tolist()
        self.num_missing_mean = num_missing_mean
        self.num_missing_median = num_missing_median
        self.cat_missing = cat_missing
        self.best_model = best_model
        self.target_type = target_type
        self.pipelinee = None
        self.encoder=None
    def train_test(self):
        if isinstance(self.y.iloc[0], str):
            self.encoder = LabelEncoder()
            self.y = self.encoder.fit_transform(self.y)
    def preprocessing(self):
        transformers = []

        if self.num_missing_mean:
            numeric_mean_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num_mean', numeric_mean_pipe, self.num_missing_mean))

        if self.num_missing_median:
            numeric_median_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num_median', numeric_median_pipe, self.num_missing_median))

        if self.cat_col:
            categorical_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))  # Note sparse_output=False
            ])
            transformers.append(('cat', categorical_pipe, self.cat_col))

        if not transformers:
            raise ValueError("No valid columns provided for preprocessing.")

        preprocessor = ColumnTransformer(transformers=transformers)
        return preprocessor

    def save_model(self):
        self.pipelinee = Pipeline(steps=[
            ('preprocessor', self.preprocessing()),
            ('model', self.best_model) 
        ])
        self.pipelinee.fit(self.X,self.y)
