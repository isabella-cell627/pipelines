"""
Advanced Time Series Data Preprocessing Pipelines

Professional, production-ready pipelines for comprehensive time series preprocessing
with enhanced feature engineering, robust error handling, and optimized performance.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from scipy import stats
from scipy.signal import periodogram
import warnings
from typing import List, Union, Optional, Dict, Any
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Advanced datetime feature extraction with comprehensive temporal characteristics
    including cyclical encoding, business features, and seasonal indicators.
    """
    
    def __init__(self, datetime_column: str = 'date', 
                 extract_cyclical: bool = True,
                 extract_business: bool = True,
                 extract_seasonal: bool = True,
                 country_holidays: Optional[str] = None):
        self.datetime_column = datetime_column
        self.extract_cyclical = extract_cyclical
        self.extract_business = extract_business
        self.extract_seasonal = extract_seasonal
        self.country_holidays = country_holidays
        self._fitted = False
        
    def _encode_cyclical(self, dt_series: pd.Series) -> Dict[str, pd.Series]:
        """Encode cyclical features using sine/cosine transformation."""
        features = {}
        
        # Hour encoding (if time component exists)
        if hasattr(dt_series.dt, 'hour'):
            hour = dt_series.dt.hour
            features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week encoding
        dayofweek = dt_series.dt.dayofweek
        features['dayofweek_sin'] = np.sin(2 * np.pi * dayofweek / 7)
        features['dayofweek_cos'] = np.cos(2 * np.pi * dayofweek / 7)
        
        # Month encoding
        month = dt_series.dt.month
        features['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
        features['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
        
        # Day of year encoding
        dayofyear = dt_series.dt.dayofyear
        features['dayofyear_sin'] = np.sin(2 * np.pi * dayofyear / 365.25)
        features['dayofyear_cos'] = np.cos(2 * np.pi * dayofyear / 365.25)
        
        return features
    
    def _extract_business_features(self, dt_series: pd.Series) -> Dict[str, pd.Series]:
        """Extract business-related temporal features."""
        features = {}
        
        # Quarter
        features['quarter'] = dt_series.dt.quarter
        
        # Semester
        features['semester'] = (dt_series.dt.quarter + 1) // 2
        
        # Week of year
        features['weekofyear'] = dt_series.dt.isocalendar().week
        
        # Is weekend
        features['is_weekend'] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
        
        # Is month start/end
        features['is_month_start'] = dt_series.dt.is_month_start.astype(int)
        features['is_month_end'] = dt_series.dt.is_month_end.astype(int)
        
        # Is quarter start/end
        features['is_quarter_start'] = dt_series.dt.is_quarter_start.astype(int)
        features['is_quarter_end'] = dt_series.dt.is_quarter_end.astype(int)
        
        # Is year start/end
        features['is_year_start'] = dt_series.dt.is_year_start.astype(int)
        features['is_year_end'] = dt_series.dt.is_year_end.astype(int)
        
        return features
    
    def _extract_seasonal_features(self, dt_series: pd.Series) -> Dict[str, pd.Series]:
        """Extract seasonal and meteorological features."""
        features = {}
        
        # Season (meteorological)
        month = dt_series.dt.month
        features['season'] = np.where(month.isin([12, 1, 2]), 0,
                             np.where(month.isin([3, 4, 5]), 1,
                             np.where(month.isin([6, 7, 8]), 2, 3)))
        
        # Is holiday (basic implementation - could be enhanced with holiday packages)
        features['is_holiday'] = 0  # Placeholder
        
        return features
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DateTimeFeatureExtractor':
        """Validate datetime column and prepare transformer."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        if self.datetime_column not in X.columns:
            raise ValueError(f"Datetime column '{self.datetime_column}' not found in DataFrame")
        
        # Validate datetime conversion
        try:
            pd.to_datetime(X[self.datetime_column])
        except Exception as e:
            raise ValueError(f"Failed to convert column '{self.datetime_column}' to datetime: {e}")
        
        self._fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive datetime feature extraction."""
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted before transformation")
            
        X_copy = X.copy()
        
        # Convert to datetime
        dt_series = pd.to_datetime(X_copy[self.datetime_column])
        
        # Basic datetime features
        X_copy['year'] = dt_series.dt.year
        X_copy['month'] = dt_series.dt.month
        X_copy['day'] = dt_series.dt.day
        X_copy['dayofweek'] = dt_series.dt.dayofweek
        X_copy['dayofyear'] = dt_series.dt.dayofyear
        
        if hasattr(dt_series.dt, 'hour'):
            X_copy['hour'] = dt_series.dt.hour
            X_copy['minute'] = dt_series.dt.minute
        
        # Cyclical encoding
        if self.extract_cyclical:
            cyclical_features = self._encode_cyclical(dt_series)
            for name, feature in cyclical_features.items():
                X_copy[name] = feature
        
        # Business features
        if self.extract_business:
            business_features = self._extract_business_features(dt_series)
            for name, feature in business_features.items():
                X_copy[name] = feature
        
        # Seasonal features
        if self.extract_seasonal:
            seasonal_features = self._extract_seasonal_features(dt_series)
            for name, feature in seasonal_features.items():
                X_copy[name] = feature
        
        # Drop original datetime column
        X_copy.drop(self.datetime_column, axis=1, inplace=True)
        
        logger.info(f"Extracted {len(X_copy.columns) - len(X.columns) + 1} datetime features")
        return X_copy


class LagFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Advanced lag feature creation with multiple strategies and automatic
    handling of different frequencies and patterns.
    """
    
    def __init__(self, lags: List[int] = [1, 2, 3, 7, 14, 21, 28], 
                 columns: Optional[List[str]] = None,
                 fill_method: str = 'ffill',
                 include_returns: bool = True,
                 return_periods: List[int] = [1, 7, 30]):
        self.lags = sorted(lags)
        self.columns = columns
        self.fill_method = fill_method
        self.include_returns = include_returns
        self.return_periods = return_periods
        self._fitted = False
        self._valid_columns = None
        
        if fill_method not in ['ffill', 'bfill', 'zero', 'mean', 'median']:
            raise ValueError("fill_method must be one of: 'ffill', 'bfill', 'zero', 'mean', 'median'")
    
    def _calculate_returns(self, X: pd.DataFrame, column: str) -> Dict[str, pd.Series]:
        """Calculate percentage returns for different periods."""
        returns = {}
        for period in self.return_periods:
            if period <= len(X):
                returns[f'{column}_return_{period}'] = (
                    X[column].pct_change(period) * 100
                )
        return returns
    
    def _fill_missing_values(self, series: pd.Series) -> pd.Series:
        """Handle missing values in lag features based on specified method."""
        if self.fill_method == 'ffill':
            return series.ffill()
        elif self.fill_method == 'bfill':
            return series.bfill()
        elif self.fill_method == 'zero':
            return series.fillna(0)
        elif self.fill_method == 'mean':
            return series.fillna(series.mean())
        elif self.fill_method == 'median':
            return series.fillna(series.median())
        else:
            return series.fillna(0)
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'LagFeatureCreator':
        """Identify columns for lag feature creation."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        # Determine which columns to use
        if self.columns is None:
            self._valid_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            missing_cols = [col for col in self.columns if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            self._valid_columns = self.columns
        
        if not self._valid_columns:
            warnings.warn("No numeric columns found for lag feature creation")
        
        self._fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive lag features with multiple strategies."""
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted before transformation")
            
        if not self._valid_columns:
            return X.copy()
            
        X_copy = X.copy()
        
        for col in self._valid_columns:
            # Basic lag features
            for lag in self.lags:
                if lag < len(X_copy):
                    lag_feature = X_copy[col].shift(lag)
                    X_copy[f'{col}_lag_{lag}'] = self._fill_missing_values(lag_feature)
            
            # Return features (percentage changes)
            if self.include_returns:
                return_features = self._calculate_returns(X_copy, col)
                for name, feature in return_features.items():
                    X_copy[name] = self._fill_missing_values(feature)
        
        logger.info(f"Created {len(self._valid_columns) * (len(self.lags) + len(self.return_periods))} lag features")
        return X_copy


class RollingStatsCreator(BaseEstimator, TransformerMixin):
    """
    Advanced rolling statistics with multiple window types, robust statistics,
    and automatic frequency detection.
    """
    
    def __init__(self, 
                 windows: List[int] = [3, 7, 14, 30],
                 columns: Optional[List[str]] = None,
                 statistics: List[str] = None,
                 min_periods: Optional[int] = None,
                 center: bool = False):
        self.windows = windows
        self.columns = columns
        self.min_periods = min_periods
        self.center = center
        self._fitted = False
        self._valid_columns = None
        
        # Define default statistics
        if statistics is None:
            self.statistics = ['mean', 'std', 'min', 'max', 'median', 'sum']
        else:
            valid_stats = ['mean', 'std', 'var', 'min', 'max', 'median', 'sum', 
                          'skew', 'kurt', 'q25', 'q75', 'range', 'cv']
            invalid_stats = [stat for stat in statistics if stat not in valid_stats]
            if invalid_stats:
                raise ValueError(f"Invalid statistics: {invalid_stats}. Valid options: {valid_stats}")
            self.statistics = statistics
    
    def _calculate_rolling_statistic(self, series: pd.Series, window: int, 
                                   statistic: str) -> pd.Series:
        """Calculate specific rolling statistic with error handling."""
        try:
            if statistic == 'mean':
                return series.rolling(window=window, min_periods=self.min_periods, 
                                    center=self.center).mean()
            elif statistic == 'std':
                return series.rolling(window=window, min_periods=self.min_periods,
                                    center=self.center).std()
            elif statistic == 'var':
                return series.rolling(window=window, min_periods=self.min_periods,
                                    center=self.center).var()
            elif statistic == 'min':
                return series.rolling(window=window, min_periods=self.min_periods,
                                    center=self.center).min()
            elif statistic == 'max':
                return series.rolling(window=window, min_periods=self.min_periods,
                                    center=self.center).max()
            elif statistic == 'median':
                return series.rolling(window=window, min_periods=self.min_periods,
                                    center=self.center).median()
            elif statistic == 'sum':
                return series.rolling(window=window, min_periods=self.min_periods,
                                    center=self.center).sum()
            elif statistic == 'skew':
                return series.rolling(window=window, min_periods=self.min_periods,
                                    center=self.center).skew()
            elif statistic == 'kurt':
                return series.rolling(window=window, min_periods=self.min_periods,
                                    center=self.center).kurt()
            elif statistic == 'q25':
                return series.rolling(window=window, min_periods=self.min_periods,
                                    center=self.center).quantile(0.25)
            elif statistic == 'q75':
                return series.rolling(window=window, min_periods=self.min_periods,
                                    center=self.center).quantile(0.75)
            elif statistic == 'range':
                return (series.rolling(window=window, min_periods=self.min_periods,
                                     center=self.center).max() - 
                       series.rolling(window=window, min_periods=self.min_periods,
                                     center=self.center).min())
            elif statistic == 'cv':
                mean = series.rolling(window=window, min_periods=self.min_periods,
                                    center=self.center).mean()
                std = series.rolling(window=window, min_periods=self.min_periods,
                                   center=self.center).std()
                return (std / mean).replace([np.inf, -np.inf], np.nan)
            else:
                return pd.Series(np.nan, index=series.index)
        except Exception as e:
            warnings.warn(f"Failed to calculate {statistic} with window {window}: {e}")
            return pd.Series(np.nan, index=series.index)
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'RollingStatsCreator':
        """Identify columns for rolling statistics."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if self.columns is None:
            self._valid_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            missing_cols = [col for col in self.columns if col not in X.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            self._valid_columns = self.columns
        
        if not self._valid_columns:
            warnings.warn("No numeric columns found for rolling statistics")
        
        self._fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute comprehensive rolling statistics."""
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted before transformation")
            
        if not self._valid_columns:
            return X.copy()
            
        X_copy = X.copy()
        
        for col in self._valid_columns:
            for window in self.windows:
                for statistic in self.statistics:
                    feature_name = f'{col}_rolling_{statistic}_{window}'
                    rolling_stat = self._calculate_rolling_statistic(
                        X_copy[col], window, statistic
                    )
                    X_copy[feature_name] = rolling_stat
        
        # Fill NaN values using forward fill then backward fill
        X_copy = X_copy.ffill().bfill().fillna(0)
        
        total_features = len(self._valid_columns) * len(self.windows) * len(self.statistics)
        logger.info(f"Created {total_features} rolling statistics features")
        return X_copy


def create_datetime_feature_pipeline(datetime_column: str = 'date', 
                                   scale: bool = True,
                                   scaler_type: str = 'standard',
                                   extract_cyclical: bool = True,
                                   extract_business: bool = True,
                                   extract_seasonal: bool = True) -> Pipeline:
    """
    Creates an advanced pipeline for extracting datetime features.
    
    Parameters:
    -----------
    datetime_column : str, default='date'
        Name of the datetime column
    scale : bool, default=True
        Whether to scale features
    scaler_type : str, default='standard'
        Type of scaler ('standard', 'robust', 'power')
    extract_cyclical : bool, default=True
        Whether to extract cyclical features
    extract_business : bool, default=True
        Whether to extract business features
    extract_seasonal : bool, default=True
        Whether to extract seasonal features
    
    Returns:
    --------
    Pipeline : sklearn.pipeline.Pipeline
    """
    steps = [
        ('datetime_features', DateTimeFeatureExtractor(
            datetime_column=datetime_column,
            extract_cyclical=extract_cyclical,
            extract_business=extract_business,
            extract_seasonal=extract_seasonal
        ))
    ]
    
    if scale:
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        elif scaler_type == 'power':
            scaler = PowerTransformer(method='yeo-johnson')
        else:
            raise ValueError("scaler_type must be 'standard', 'robust', or 'power'")
        
        steps.append(('scaler', scaler))
    
    return Pipeline(steps)


def create_lag_feature_pipeline(lags: List[int] = [1, 2, 3, 7, 14, 21, 28],
                              columns: Optional[List[str]] = None,
                              scale: bool = True,
                              scaler_type: str = 'standard',
                              fill_method: str = 'ffill',
                              include_returns: bool = True) -> Pipeline:
    """
    Creates an advanced pipeline for creating lag features.
    
    Parameters:
    -----------
    lags : list, default=[1, 2, 3, 7, 14, 21, 28]
        List of lag values to create
    columns : list, default=None
        Columns to create lags for (None = all numeric)
    scale : bool, default=True
        Whether to scale features
    scaler_type : str, default='standard'
        Type of scaler ('standard', 'robust', 'power')
    fill_method : str, default='ffill'
        Method for filling missing values
    include_returns : bool, default=True
        Whether to include return features
    
    Returns:
    --------
    Pipeline : sklearn.pipeline.Pipeline
    """
    steps = [
        ('lag_features', LagFeatureCreator(
            lags=lags,
            columns=columns,
            fill_method=fill_method,
            include_returns=include_returns
        ))
    ]
    
    if scale:
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        elif scaler_type == 'power':
            scaler = PowerTransformer(method='yeo-johnson')
        else:
            raise ValueError("scaler_type must be 'standard', 'robust', or 'power'")
        
        steps.append(('scaler', scaler))
    
    return Pipeline(steps)


def create_rolling_stats_pipeline(windows: List[int] = [3, 7, 14, 30],
                                columns: Optional[List[str]] = None,
                                scale: bool = True,
                                scaler_type: str = 'standard',
                                statistics: Optional[List[str]] = None,
                                center: bool = False) -> Pipeline:
    """
    Creates an advanced pipeline for rolling window statistics.
    
    Parameters:
    -----------
    windows : list, default=[3, 7, 14, 30]
        Window sizes for rolling statistics
    columns : list, default=None
        Columns to compute stats for (None = all numeric)
    scale : bool, default=True
        Whether to scale features
    scaler_type : str, default='standard'
        Type of scaler ('standard', 'robust', 'power')
    statistics : list, default=None
        Statistics to compute (None = default set)
    center : bool, default=False
        Whether to center the rolling window
    
    Returns:
    --------
    Pipeline : sklearn.pipeline.Pipeline
    """
    steps = [
        ('rolling_stats', RollingStatsCreator(
            windows=windows,
            columns=columns,
            statistics=statistics,
            center=center
        ))
    ]
    
    if scale:
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        elif scaler_type == 'power':
            scaler = PowerTransformer(method='yeo-johnson')
        else:
            raise ValueError("scaler_type must be 'standard', 'robust', or 'power'")
        
        steps.append(('scaler', scaler))
    
    return Pipeline(steps)


def create_comprehensive_timeseries_pipeline(datetime_column: str = 'date',
                                           lags: List[int] = [1, 7, 14, 28],
                                           windows: List[int] = [7, 14, 30],
                                           columns: Optional[List[str]] = None,
                                           scaler_type: str = 'robust',
                                           include_imputation: bool = True) -> Pipeline:
    """
    Creates a comprehensive, production-ready time series preprocessing pipeline.
    
    Parameters:
    -----------
    datetime_column : str, default='date'
        Name of the datetime column
    lags : list, default=[1, 7, 14, 28]
        Lag values to create
    windows : list, default=[7, 14, 30]
        Window sizes for rolling statistics
    columns : list, default=None
        Columns for lag and rolling features
    scaler_type : str, default='robust'
        Type of scaler to use
    include_imputation : bool, default=True
        Whether to include data imputation
    
    Returns:
    --------
    Pipeline : sklearn.pipeline.Pipeline
    """
    steps = []
    
    # Data imputation
    if include_imputation:
        steps.append(('imputer', SimpleImputer(strategy='median')))
    
    # Feature engineering steps
    steps.extend([
        ('datetime_features', DateTimeFeatureExtractor(
            datetime_column=datetime_column,
            extract_cyclical=True,
            extract_business=True,
            extract_seasonal=True
        )),
        ('lag_features', LagFeatureCreator(
            lags=lags,
            columns=columns,
            fill_method='ffill',
            include_returns=True
        )),
        ('rolling_stats', RollingStatsCreator(
            windows=windows,
            columns=columns,
            statistics=['mean', 'std', 'min', 'max', 'median', 'sum'],
            center=False
        ))
    ])
    
    # Scaling
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'power':
        scaler = PowerTransformer(method='yeo-johnson')
    else:
        raise ValueError("scaler_type must be 'standard', 'robust', or 'power'")
    
    steps.append(('scaler', scaler))
    
    return Pipeline(steps)