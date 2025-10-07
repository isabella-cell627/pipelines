"""
Advanced Mixed Data Preprocessing Framework

Provides enterprise-grade pipelines for preprocessing mixed data types
with advanced feature engineering, automatic type detection, and robust handling.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, RobustScaler, 
    PowerTransformer, QuantileTransformer, KBinsDiscretizer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import warnings
from typing import List, Union, Optional, Dict, Any, Callable
import copy


class SmartImputer(BaseEstimator, TransformerMixin):
    """
    Intelligent imputation with automatic strategy selection based on data characteristics.
    """
    
    def __init__(self, 
                 numerical_strategy: str = 'auto',
                 categorical_strategy: str = 'auto',
                 max_knn_neighbors: int = 5,
                 missing_threshold: float = 0.3):
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.max_knn_neighbors = max_knn_neighbors
        self.missing_threshold = missing_threshold
        self.num_imputers_ = {}
        self.cat_imputers_ = {}
        self.column_metadata_ = {}
        
    def _detect_best_strategy(self, X: pd.DataFrame, col: str, is_numeric: bool) -> str:
        """Automatically detect the best imputation strategy."""
        missing_ratio = X[col].isna().mean()
        
        if missing_ratio > self.missing_threshold:
            return 'constant' if is_numeric else 'most_frequent'
        
        if is_numeric:
            if self.numerical_strategy != 'auto':
                return self.numerical_strategy
                
            # Analyze data distribution
            if X[col].nunique() < 10:
                return 'median'
            elif X[col].skew() > 2:
                return 'median'
            else:
                return 'mean'
        else:
            if self.categorical_strategy != 'auto':
                return self.categorical_strategy
                
            return 'most_frequent' if X[col].nunique() < 20 else 'constant'
    
    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        
        for col in X.columns:
            is_numeric = pd.api.types.is_numeric_dtype(X[col])
            strategy = self._detect_best_strategy(X, col, is_numeric)
            
            if is_numeric:
                if strategy == 'knn':
                    imp = KNNImputer(n_neighbors=min(self.max_knn_neighbors, 
                                                   X[col].notna().sum() - 1))
                else:
                    fill_value = 0 if strategy == 'constant' else None
                    imp = SimpleImputer(strategy=strategy, fill_value=fill_value)
                
                imp.fit(X[[col]])
                self.num_imputers_[col] = imp
            else:
                fill_value = 'missing' if strategy == 'constant' else None
                imp = SimpleImputer(strategy=strategy, fill_value=fill_value)
                imp.fit(X[[col]])
                self.cat_imputers_[col] = imp
            
            self.column_metadata_[col] = {
                'is_numeric': is_numeric,
                'strategy': strategy,
                'missing_ratio': X[col].isna().mean()
            }
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        X = X.copy()
        
        for col in X.columns:
            if col in self.num_imputers_:
                imputer = self.num_imputers_[col]
                X[col] = imputer.transform(X[[col]]).ravel()
            elif col in self.cat_imputers_:
                imputer = self.cat_imputers_[col]
                X[col] = imputer.transform(X[[col]]).ravel()
        
        return X


class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering with automatic feature creation and transformation.
    """
    
    def __init__(self, 
                 create_interactions: bool = True,
                 polynomial_degree: int = 2,
                 create_ratios: bool = True,
                 datetime_features: bool = True):
        self.create_interactions = create_interactions
        self.polynomial_degree = polynomial_degree
        self.create_ratios = create_ratios
        self.datetime_features = datetime_features
        self.numeric_columns_ = []
        self.feature_mapping_ = {}
        
    def _extract_datetime_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract features from datetime columns."""
        datetime_columns = X.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_columns:
            X[f'{col}_year'] = X[col].dt.year
            X[f'{col}_month'] = X[col].dt.month
            X[f'{col}_day'] = X[col].dt.day
            X[f'{col}_dayofweek'] = X[col].dt.dayofweek
            X[f'{col}_quarter'] = X[col].dt.quarter
            X[f'{col}_is_weekend'] = (X[col].dt.dayofweek >= 5).astype(int)
        
        return X.drop(columns=datetime_columns)
    
    def _create_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between numeric columns."""
        from itertools import combinations
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return X
            
        for col1, col2 in combinations(numeric_cols, 2):
            # Only create interactions for top correlated pairs
            if abs(X[col1].corr(X[col2])) > 0.3:
                X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                if self.create_ratios and X[col2].min() > 0:
                    X[f'{col1}_div_{col2}'] = X[col1] / X[col2]
        
        return X
    
    def _create_polynomials(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features for highly predictive numeric columns."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # Limit to top 5 columns
            for degree in range(2, self.polynomial_degree + 1):
                X[f'{col}_pow_{degree}'] = X[col] ** degree
        
        return X
    
    def fit(self, X: pd.DataFrame, y=None):
        self.numeric_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        X = X.copy()
        
        # Extract datetime features
        if self.datetime_features:
            X = self._extract_datetime_features(X)
        
        # Create interaction features
        if self.create_interactions:
            X = self._create_interactions(X)
        
        # Create polynomial features
        if self.polynomial_degree > 1:
            X = self._create_polynomials(X)
        
        return X


class DynamicEncoder(BaseEstimator, TransformerMixin):
    """
    Dynamic encoding for categorical variables with multiple strategies.
    """
    
    def __init__(self, 
                 max_categories: int = 50,
                 encoding_strategy: str = 'auto',
                 target_encoding: bool = False,
                 rare_threshold: float = 0.01):
        self.max_categories = max_categories
        self.encoding_strategy = encoding_strategy
        self.target_encoding = target_encoding
        self.rare_threshold = rare_threshold
        self.encoders_ = {}
        self.category_mappings_ = {}
        
    def _auto_select_encoding(self, X: pd.Series, y=None) -> str:
        """Automatically select the best encoding strategy."""
        n_unique = X.nunique()
        cardinality = n_unique / len(X)
        
        if n_unique <= 10:
            return 'onehot'
        elif n_unique <= 50:
            return 'target' if self.target_encoding and y is not None else 'onehot'
        else:
            return 'frequency'
    
    def _handle_rare_categories(self, X: pd.Series) -> pd.Series:
        """Group rare categories into 'other'."""
        value_counts = X.value_counts(normalize=True)
        rare_categories = value_counts[value_counts < self.rare_threshold].index
        return X.replace(dict.fromkeys(rare_categories, 'rare'))
    
    def fit(self, X: pd.DataFrame, y=None):
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            X_clean = self._handle_rare_categories(X[col].copy())
            
            strategy = self.encoding_strategy
            if strategy == 'auto':
                strategy = self._auto_select_encoding(X_clean, y)
            
            if strategy == 'onehot':
                encoder = OneHotEncoder(
                    handle_unknown='ignore', 
                    sparse_output=False,
                    drop='first'
                )
                encoder.fit(X_clean.values.reshape(-1, 1))
                
            elif strategy == 'frequency':
                freq_encoding = X_clean.value_counts().to_dict()
                encoder = freq_encoding
                
            elif strategy == 'target' and y is not None:
                target_means = y.groupby(X_clean).mean().to_dict()
                encoder = target_means
            
            self.encoders_[col] = (strategy, encoder)
            self.category_mappings_[col] = X_clean.unique().tolist()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        X = X.copy()
        
        for col, (strategy, encoder) in self.encoders_.items():
            X_clean = self._handle_rare_categories(X[col].copy())
            
            if strategy == 'onehot':
                encoded = encoder.transform(X_clean.values.reshape(-1, 1))
                encoded_df = pd.DataFrame(
                    encoded, 
                    columns=[f'{col}_{cat}' for cat in encoder.categories_[0][1:]],
                    index=X.index
                )
                X = pd.concat([X.drop(columns=[col]), encoded_df], axis=1)
                
            elif strategy == 'frequency':
                X[col] = X_clean.map(encoder).fillna(0)
                
            elif strategy == 'target':
                X[col] = X_clean.map(encoder).fillna(0)
        
        return X


def create_mixed_data_pipeline(numerical_features, categorical_features, 
                              numerical_strategy='auto', categorical_strategy='auto',
                              feature_engineering=True, advanced_encoding=True):
    """
    Creates an intelligent pipeline for mixed numerical and categorical data.
    """
    # Smart preprocessing pipelines
    numerical_transformer = Pipeline([
        ('smart_imputer', SmartImputer(
            numerical_strategy=numerical_strategy,
            categorical_strategy=categorical_strategy
        )),
        ('scaler', StandardScaler()),
        ('variance_threshold', VarianceThreshold(threshold=0.01))
    ])
    
    categorical_transformer = Pipeline([
        ('smart_imputer', SmartImputer(
            numerical_strategy=numerical_strategy,
            categorical_strategy=categorical_strategy
        )),
        ('dynamic_encoder', DynamicEncoder(
            encoding_strategy='auto',
            target_encoding=False
        ))
    ])
    
    transformers = [
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        n_jobs=-1
    )
    
    # Create final pipeline with optional feature engineering
    pipeline_steps = [('preprocessor', preprocessor)]
    
    if feature_engineering:
        pipeline_steps.insert(0, ('feature_engineer', AdvancedFeatureEngineer()))
    
    return Pipeline(pipeline_steps)


def create_robust_mixed_pipeline(numerical_features, categorical_features,
                                outlier_handling=True, power_transform=True):
    """
    Creates a robust pipeline for mixed data with advanced outlier handling.
    """
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier_scaler', RobustScaler()),
        ('power_transform', PowerTransformer(method='yeo-johnson') if power_transform else 'passthrough'),
        ('variance_threshold', VarianceThreshold(threshold=0.05))
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            handle_unknown='infrequent_if_exist',
            sparse_output=False,
            max_categories=30
        ))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop',
        n_jobs=-1
    )
    
    return Pipeline([
        ('feature_engineer', AdvancedFeatureEngineer(create_interactions=True)),
        ('preprocessor', preprocessor)
    ])


def create_custom_mixed_pipeline(numerical_features, categorical_features,
                                numerical_pipeline=None, categorical_pipeline=None,
                                feature_engineering_config=None):
    """
    Creates a highly customizable mixed data pipeline.
    """
    # Default pipelines if not provided
    if numerical_pipeline is None:
        numerical_pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', QuantileTransformer(output_distribution='normal')),
            ('variance_threshold', VarianceThreshold(threshold=0.02))
        ])
    
    if categorical_pipeline is None:
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', DynamicEncoder(
                max_categories=100,
                encoding_strategy='auto',
                rare_threshold=0.005
            ))
        ])
    
    # Feature engineering configuration
    fe_config = feature_engineering_config or {
        'create_interactions': True,
        'polynomial_degree': 2,
        'create_ratios': True,
        'datetime_features': True
    }
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough',
        n_jobs=-1
    )
    
    return Pipeline([
        ('feature_engineer', AdvancedFeatureEngineer(**fe_config)),
        ('preprocessor', preprocessor)
    ])


def create_auto_ml_pipeline(X, target_column=None):
    """
    Creates an automatic machine learning pipeline with intelligent feature detection.
    """
    # Auto-detect feature types
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if target_column and target_column in numerical_features:
        numerical_features.remove(target_column)
    if target_column and target_column in categorical_features:
        categorical_features.remove(target_column)
    
    print(f"Detected {len(numerical_features)} numerical features")
    print(f"Detected {len(categorical_features)} categorical features")
    
    # Choose pipeline based on data characteristics
    total_features = len(numerical_features) + len(categorical_features)
    high_dimensional = total_features > 50
    
    if high_dimensional:
        print("Using robust pipeline for high-dimensional data")
        return create_robust_mixed_pipeline(
            numerical_features, 
            categorical_features,
            power_transform=True
        )
    else:
        print("Using intelligent pipeline with feature engineering")
        return create_mixed_data_pipeline(
            numerical_features,
            categorical_features,
            feature_engineering=True,
            advanced_encoding=True
        )


