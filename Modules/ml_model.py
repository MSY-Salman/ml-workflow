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
class ml_model:
    def __init__(self, data, target_type, target):
        self.data = data
        self.target = target
        self.target_type = target_type
        self.col = self.data.columns.values.tolist()
        self.X = self.data.drop(columns=[target], axis=1)
        self.y = self.data[target]
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.best_model=None
        self.best_score=None
        self.scores=[]
    def train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Ensure arrays are contiguous and properly typed
        if isinstance(self.y.iloc[0], str):
            self.encoder = LabelEncoder()
            self.y_train = self.encoder.fit_transform(self.y_train)
            self.y_test = self.encoder.transform(self.y_test)
        
        # Convert to numpy arrays and ensure contiguity
        self.X_train = np.ascontiguousarray(self.X_train)
        self.X_test = np.ascontiguousarray(self.X_test)
        self.y_train = np.ascontiguousarray(self.y_train)
        self.y_test = np.ascontiguousarray(self.y_test)
    def pca(self):
        if len(self.col) > 100:
            pca = PCA(n_components=int(np.sqrt(len(self.col))))
            self.X_train = pca.fit_transform(self.X_train)
            self.X_test = pca.transform(self.X_test)

    def linear_regression(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return r2_score(self.y_test, y_pred), model

    def sgd_regression(self):
        params = {
            "loss": ["squared_error", "huber"],
            "penalty": ["l2", "elasticnet"],
            "alpha": [0.0001, 0.01],
            "learning_rate": ["constant", "adaptive"],
            "eta0": [0.01],
            "max_iter": [500],
            "tol": [1e-3]
        }
        model = GridSearchCV(SGDRegressor(random_state=42), param_grid=params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.best_estimator_.predict(self.X_test)
        return r2_score(self.y_test, y_pred), model.best_estimator_

    def logistic_regression(self):
        params = {
            "penalty": ["l1", "l2", "elasticnet"],
            "C": [0.1, 1],
            "solver": ["saga"],
            "max_iter": [200],
            "tol": [1e-4],
            "l1_ratio": [0.5]
        }

        model = GridSearchCV(LogisticRegression(random_state=42), param_grid=params, cv=3, scoring='accuracy', n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.best_estimator_.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred), model.best_estimator_

    def decision_tree_regression(self):
        params = {
            "criterion": ["squared_error", "absolute_error"],
            "splitter": ["best"],
            "max_depth": [5, 10],
            "min_samples_split": [2],
            "min_samples_leaf": [1, 2],
            "max_features": [None, "sqrt"]
        }
        model = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid=params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.best_estimator_.predict(self.X_test)
        return r2_score(self.y_test, y_pred), model.best_estimator_

    def decision_tree_classification(self):
        params = {
            "criterion": ["gini", "entropy"],
            "splitter": ["best"],
            "max_depth": [5, 10],
            "min_samples_split": [2],
            "min_samples_leaf": [1, 2],
            "max_features": [None, "sqrt"]
        }
        model = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid=params, cv=3, scoring='accuracy', n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.best_estimator_.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred), model.best_estimator_

    def random_forest_regression(self):
        params = {
            "criterion": ["squared_error"],
            "max_depth": [5, 10],
            "min_samples_split": [2],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt"],
            "n_estimators": [100]
        }
        model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid=params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.best_estimator_.predict(self.X_test)
        return r2_score(self.y_test, y_pred), model.best_estimator_

    def random_forest_classification(self):
        params = {
            "criterion": ["gini"],
            "max_depth": [5, 10],
            "min_samples_split": [2],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt"],
            "n_estimators": [100]
        }
        model = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=params, cv=3, scoring='accuracy', n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.best_estimator_.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred), model.best_estimator_

    def ada_regression(self):
        params = {
            "n_estimators": [50, 100],
            "learning_rate": [0.1],
            "estimator__max_depth": [3],
            "estimator__min_samples_split": [2],
            "estimator__min_samples_leaf": [1]
        }
        model = GridSearchCV(AdaBoostRegressor(estimator=DecisionTreeRegressor(), random_state=42), param_grid=params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.best_estimator_.predict(self.X_test)
        return r2_score(self.y_test, y_pred), model.best_estimator_

    def ada_classification(self):
        params = {
            "n_estimators": [50, 100],
            "learning_rate": [0.1],
            "estimator__max_depth": [3],
            "estimator__min_samples_split": [2],
            "estimator__min_samples_leaf": [1]
        }
        model = GridSearchCV(AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=42), param_grid=params, cv=3, scoring='accuracy', n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.best_estimator_.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred), model.best_estimator_

    def gradient_regression(self):
        model = GridSearchCV(GradientBoostingRegressor(), param_grid={"n_estimators": [100], "learning_rate": [0.1], "max_depth": [3]}, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.best_estimator_.predict(self.X_test)
        return r2_score(self.y_test, y_pred), model.best_estimator_

    def gradient_classification(self):
        model = GridSearchCV(GradientBoostingClassifier(), param_grid={"n_estimators": [100], "learning_rate": [0.1], "max_depth": [3]}, cv=3, scoring='accuracy', n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.best_estimator_.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred), model.best_estimator_

    def xgradient_regression(self):
        model = GridSearchCV(XGBRegressor(), param_grid={"n_estimators": [100], "learning_rate": [0.1], "max_depth": [3]}, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.best_estimator_.predict(self.X_test)
        return r2_score(self.y_test, y_pred), model.best_estimator_

    def xgradient_classification(self):
        model = GridSearchCV(XGBClassifier(), param_grid={"n_estimators": [100], "learning_rate": [0.1], "max_depth": [3]}, cv=3, scoring='accuracy', n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.best_estimator_.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred), model.best_estimator_

    def naive_byes(self):
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred), model

    def knn_regression(self):
        model = GridSearchCV(KNeighborsRegressor(), param_grid={"n_neighbors": [3, 5], "weights": ["uniform"]}, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.best_estimator_.predict(self.X_test)
        return r2_score(self.y_test, y_pred), model.best_estimator_

    def knn_classification(self):
        model = GridSearchCV(KNeighborsClassifier(), param_grid={"n_neighbors": [3, 5], "weights": ["uniform"]}, cv=3, scoring='accuracy', n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.best_estimator_.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred), model.best_estimator_

    def svm_regression(self):
        model = GridSearchCV(SVR(), param_grid={"C": [1], "kernel": ["rbf"]}, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.best_estimator_.predict(self.X_test)
        return r2_score(self.y_test, y_pred), model.best_estimator_

    def svm_classification(self):
        model = GridSearchCV(SVC(), param_grid={"C": [1], "kernel": ["rbf"]}, cv=3, scoring='accuracy', n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.best_estimator_.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred), model.best_estimator_

    def get_best_regression(self):
        l = [self.linear_regression(), self.sgd_regression(), self.decision_tree_regression(),
             self.random_forest_regression(), self.ada_regression(), self.gradient_regression(),
             self.xgradient_regression(), self.svm_regression(), self.knn_regression()]
        best_model,best_score = None,-float('inf')
        for score, model in l:
            # Store both model name and score in the required format
            model_name = type(model).__name__
            self.scores.append({
                'Model': model_name,
                'Score': score
            })
            
            if score > best_score:
                best_model, best_score = model, score
        return best_model, best_score

    def get_best_classification(self):
        l = [self.logistic_regression(), self.decision_tree_classification(), self.random_forest_classification(),
             self.ada_classification(), self.gradient_classification(), self.xgradient_classification(),
             self.svm_classification(), self.naive_byes(), self.knn_classification()]
        best_model,best_score = None,-float('inf')
        for score, model in l:
            # Store both model name and score in the required format
            model_name = type(model).__name__
            self.scores.append({
                'Model': model_name,
                'Score': score
            })
            if score > best_score:
                best_model, best_score = model, score
        return best_model, best_score
        

    def model_selection(self):
        # Ensure data is contiguous before model training
        self.X_train = np.ascontiguousarray(self.X_train)
        self.X_test = np.ascontiguousarray(self.X_test)
        self.y_train = np.ascontiguousarray(self.y_train)
        self.y_test = np.ascontiguousarray(self.y_test)
        
        if self.target_type == 'R':
            self.best_model, self.best_score = self.get_best_regression()
        else:
            self.best_model, self.best_score = self.get_best_classification()
