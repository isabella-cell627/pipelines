"""
Advanced Numerical Data Preprocessing Pipelines

Provides industrial-grade pipelines for preprocessing numerical data with
advanced statistical methods, robust outlier handling, and comprehensive
validation. Features include:
- Advanced scaling and normalization techniques
- Statistical missing value imputation with confidence intervals
- Multivariate outlier detection and handling
- Power transformations with automatic parameter selection
- Comprehensive data validation and quality checks
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import scipy.stats as stats
from scipy import optimize
import warnings
from typing import Union, Optional, Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedImputer(BaseEstimator, TransformerMixin):
    """
    Advanced statistical imputation with multiple strategies and confidence intervals.
    """
    
    def __init__(self, strategy: str = 'adaptive_median', n_neighbors: int = 5, 
                 random_state: int = 42, confidence_level: float = 0.95):
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.confidence_level = confidence_level
        self.imputation_values_ = {}
        self.confidence_intervals_ = {}
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X = self._validate_data(X)
        n_samples, n_features = X.shape
        
        for col_idx in range(n_features):
            col_data = X[:, col_idx]
            mask = ~np.isnan(col_data)
            valid_data = col_data[mask]
            
            if len(valid_data) == 0:
                self.imputation_values_[col_idx] = 0.0
                self.confidence_intervals_[col_idx] = (0.0, 0.0)
                continue
                
            if self.strategy == 'adaptive_median':
                impute_val = self._adaptive_median(valid_data)
            elif self.strategy == 'trimmed_mean':
                impute_val = self._trimmed_mean(valid_data)
            elif self.strategy == 'm_estimator':
                impute_val = self._m_estimator(valid_data)
            else:
                impute_val = np.median(valid_data)
                
            self.imputation_values_[col_idx] = impute_val
            self.confidence_intervals_[col_idx] = self._calculate_confidence_interval(valid_data)
            
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_data(X, reset=False)
        X_transformed = X.copy()
        
        for col_idx in range(X.shape[1]):
            mask = np.isnan(X[:, col_idx])
            if np.any(mask):
                X_transformed[mask, col_idx] = self.imputation_values_[col_idx]
                
        return X_transformed
    
    def _adaptive_median(self, data: np.ndarray) -> float:
        """Adaptive median that adjusts for skewness."""
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        skewness = stats.skew(data)
        
        if abs(skewness) > 1:  # Highly skewed
            return np.median(data)
        else:
            return np.mean(data[(data >= q1 - 1.5*iqr) & (data <= q3 + 1.5*iqr)])
    
    def _trimmed_mean(self, data: np.ndarray, proportiontocut: float = 0.1) -> float:
        """Trimmed mean that removes outliers."""
        return stats.trim_mean(data, proportiontocut)
    
    def _m_estimator(self, data: np.ndarray, c: float = 4.685) -> float:
        """M-estimator using Tukey's biweight function for robust estimation."""
        def tukey_biweight(x, c):
            t = np.abs(x) / c
            return np.where(t < 1, (1 - t**2)**2, 0)
        
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        scale = mad / 0.6745
        
        if scale == 0:
            return median
            
        # Iteratively reweighted least squares
        for _ in range(50):
            residuals = (data - median) / scale
            weights = tukey_biweight(residuals, c)
            new_median = np.average(data, weights=weights)
            if np.abs(new_median - median) < 1e-6:
                break
            median = new_median
            
        return median
    
    def _calculate_confidence_interval(self, data: np.ndarray) -> tuple:
        """Calculate confidence interval for imputation values."""
        if len(data) < 2:
            return (data[0], data[0])
        
        try:
            return stats.t.interval(self.confidence_level, len(data)-1, 
                                  loc=np.mean(data), scale=stats.sem(data))
        except:
            return (np.min(data), np.max(data))
    
    def _validate_data(self, X: np.ndarray, reset: bool = True) -> np.ndarray:
        """Validate and convert input data."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        return X.astype(float)


class OutlierRobustScaler(BaseEstimator, TransformerMixin):
    """
    Advanced robust scaler with multiple outlier detection methods.
    """
    
    def __init__(self, method: str = 'iqr', contamination: float = 0.1, 
                 random_state: int = 42, threshold: float = 3.0):
        self.method = method
        self.contamination = contamination
        self.random_state = random_state
        self.threshold = threshold
        self.center_ = None
        self.scale_ = None
        self.outlier_mask_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X = self._validate_data(X)
        n_samples, n_features = X.shape
        
        self.center_ = np.zeros(n_features)
        self.scale_ = np.zeros(n_features)
        self.outlier_mask_ = np.zeros((n_samples, n_features), dtype=bool)
        
        for col_idx in range(n_features):
            col_data = X[:, col_idx]
            mask = ~np.isnan(col_data)
            valid_data = col_data[mask]
            
            if self.method == 'iqr':
                self.center_[col_idx], self.scale_[col_idx] = self._iqr_scaling(valid_data)
                self.outlier_mask_[mask, col_idx] = self._detect_iqr_outliers(valid_data)
            elif self.method == 'isolation_forest':
                self.center_[col_idx], self.scale_[col_idx] = self._robust_scaling(valid_data)
                self.outlier_mask_[mask, col_idx] = self._detect_isolation_forest_outliers(valid_data.reshape(-1, 1))
            elif self.method == 'elliptic_envelope':
                self.center_[col_idx], self.scale_[col_idx] = self._robust_scaling(valid_data)
                self.outlier_mask_[mask, col_idx] = self._detect_elliptic_envelope_outliers(valid_data.reshape(-1, 1))
            elif self.method == 'zscore':
                self.center_[col_idx], self.scale_[col_idx] = self._zscore_scaling(valid_data)
                self.outlier_mask_[mask, col_idx] = self._detect_zscore_outliers(valid_data)
            else:
                self.center_[col_idx], self.scale_[col_idx] = self._robust_scaling(valid_data)
                
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_data(X, reset=False)
        X_transformed = X.copy()
        
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            mask = ~np.isnan(col_data)
            
            if self.scale_[col_idx] != 0:
                X_transformed[mask, col_idx] = (
                    (col_data[mask] - self.center_[col_idx]) / self.scale_[col_idx]
                )
                
        return X_transformed
    
    def _iqr_scaling(self, data: np.ndarray) -> tuple:
        """IQR-based robust scaling."""
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        center = np.median(data)
        scale = iqr / 1.349  # Convert to normal distribution scale
        
        if scale == 0:
            scale = np.std(data) if len(data) > 1 else 1.0
            
        return center, scale
    
    def _robust_scaling(self, data: np.ndarray) -> tuple:
        """Robust scaling using median and MAD."""
        center = np.median(data)
        mad = np.median(np.abs(data - center))
        scale = mad / 0.6745
        
        if scale == 0:
            scale = np.std(data) if len(data) > 1 else 1.0
            
        return center, scale
    
    def _zscore_scaling(self, data: np.ndarray) -> tuple:
        """Standard z-score scaling."""
        return np.mean(data), np.std(data)
    
    def _detect_iqr_outliers(self, data: np.ndarray) -> np.ndarray:
        """Detect outliers using IQR method."""
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return (data < lower_bound) | (data > upper_bound)
    
    def _detect_zscore_outliers(self, data: np.ndarray) -> np.ndarray:
        """Detect outliers using z-score method."""
        z_scores = np.abs(stats.zscore(data))
        return z_scores > self.threshold
    
    def _detect_isolation_forest_outliers(self, data: np.ndarray) -> np.ndarray:
        """Detect outliers using Isolation Forest."""
        iso_forest = IsolationForest(contamination=self.contamination, 
                                   random_state=self.random_state)
        preds = iso_forest.fit_predict(data)
        return preds == -1
    
    def _detect_elliptic_envelope_outliers(self, data: np.ndarray) -> np.ndarray:
        """Detect outliers using Elliptic Envelope."""
        envelope = EllipticEnvelope(contamination=self.contamination, 
                                  random_state=self.random_state)
        preds = envelope.fit_predict(data)
        return preds == -1
    
    def _validate_data(self, X: np.ndarray, reset: bool = True) -> np.ndarray:
        """Validate and convert input data."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        return X.astype(float)


class AdaptivePowerTransformer(BaseEstimator, TransformerMixin):
    """
    Advanced power transformer with automatic method selection and parameter optimization.
    """
    
    def __init__(self, method: str = 'auto', standardize: bool = True, 
                 random_state: int = 42):
        self.method = method
        self.standardize = standardize
        self.random_state = random_state
        self.lambdas_ = None
        self.optimized_methods_ = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        X = self._validate_data(X)
        n_samples, n_features = X.shape
        
        self.lambdas_ = np.zeros(n_features)
        self.optimized_methods_ = []
        
        for col_idx in range(n_features):
            col_data = X[:, col_idx]
            mask = ~np.isnan(col_data)
            valid_data = col_data[mask]
            
            if self.method == 'auto':
                best_method, best_lambda = self._auto_select_method(valid_data)
                self.optimized_methods_.append(best_method)
                self.lambdas_[col_idx] = best_lambda
            else:
                self.optimized_methods_.append(self.method)
                self.lambdas_[col_idx] = self._optimize_lambda(valid_data, self.method)
                
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_data(X, reset=False)
        X_transformed = X.copy()
        
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            mask = ~np.isnan(col_data)
            valid_data = col_data[mask]
            
            if len(valid_data) > 0:
                method = self.optimized_methods_[col_idx]
                lambda_val = self.lambdas_[col_idx]
                X_transformed[mask, col_idx] = self._apply_transform(valid_data, method, lambda_val)
                
        if self.standardize:
            X_transformed = (X_transformed - np.mean(X_transformed, axis=0)) / np.std(X_transformed, axis=0)
            
        return X_transformed
    
    def _auto_select_method(self, data: np.ndarray) -> tuple:
        """Automatically select the best transformation method."""
        methods = ['yeo-johnson', 'box-cox'] if np.all(data > 0) else ['yeo-johnson']
        best_method = 'yeo-johnson'
        best_lambda = 1.0
        best_normality = float('inf')
        
        for method in methods:
            try:
                lambda_val = self._optimize_lambda(data, method)
                transformed = self._apply_transform(data, method, lambda_val)
                
                # Test for normality using multiple criteria
                normality_score = self._assess_normality(transformed)
                
                if normality_score < best_normality:
                    best_normality = normality_score
                    best_method = method
                    best_lambda = lambda_val
            except:
                continue
                
        return best_method, best_lambda
    
    def _optimize_lambda(self, data: np.ndarray, method: str) -> float:
        """Optimize lambda parameter for power transformation."""
        try:
            if method == 'box-cox' and np.all(data > 0):
                return stats.boxcox_normmax(data, method='mle')
            else:  # yeo-johnson
                return self._optimize_yeo_johnson_lambda(data)
        except:
            return 1.0
    
    def _optimize_yeo_johnson_lambda(self, data: np.ndarray) -> float:
        """Optimize lambda for Yeo-Johnson transformation."""
        def neg_log_likelihood(lmbda, data):
            transformed = self._yeo_johnson_transform(data, lmbda)
            n = len(transformed)
            return -(-n/2 * np.log(np.var(transformed)) + (lmbda - 1) * np.sum(np.sign(data) * np.log1p(np.abs(data))))
        
        try:
            result = optimize.minimize_scalar(neg_log_likelihood, args=(data,), 
                                            bounds=(-2, 2), method='bounded')
            return result.x
        except:
            return 1.0
    
    def _yeo_johnson_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        """Apply Yeo-Johnson transformation."""
        x = np.asarray(x)
        pos = x >= 0
        neg = ~pos
        
        result = np.zeros_like(x)
        
        # For positive values
        if lmbda != 0:
            result[pos] = (np.power(x[pos] + 1, lmbda) - 1) / lmbda
        else:
            result[pos] = np.log(x[pos] + 1)
            
        # For negative values
        if lmbda != 2:
            result[neg] = -(np.power(-x[neg] + 1, 2 - lmbda) - 1) / (2 - lmbda)
        else:
            result[neg] = -np.log(-x[neg] + 1)
            
        return result
    
    def _apply_transform(self, data: np.ndarray, method: str, lmbda: float) -> np.ndarray:
        """Apply the specified transformation."""
        if method == 'box-cox':
            return stats.boxcox(data, lmbda) if lmbda != 0 else np.log(data)
        else:  # yeo-johnson
            return self._yeo_johnson_transform(data, lmbda)
    
    def _assess_normality(self, data: np.ndarray) -> float:
        """Assess normality using multiple statistical tests."""
        if len(data) < 3:
            return float('inf')
            
        try:
            # Combine multiple normality tests
            shapiro_stat, shapiro_p = stats.shapiro(data)
            anderson_stat = stats.anderson(data).statistic
            dagostino_stat, dagostino_p = stats.normaltest(data)
            
            # Composite score (lower is better)
            return (1 - shapiro_p) + anderson_stat/10 + (1 - dagostino_p)
        except:
            return float('inf')
    
    def _validate_data(self, X: np.ndarray, reset: bool = True) -> np.ndarray:
        """Validate and convert input data."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        return X.astype(float)


def create_standard_scaler_pipeline(impute_strategy: str = 'adaptive_median', 
                                  scaler_type: str = 'standard',
                                  handle_outliers: bool = True) -> Pipeline:
    """
    Creates an advanced pipeline with statistical imputation and robust scaling.
    
    Parameters:
    -----------
    impute_strategy : str, default='adaptive_median'
        Advanced imputation strategy ('adaptive_median', 'trimmed_mean', 'm_estimator')
    scaler_type : str, default='standard'
        Type of scaler ('standard', 'minmax', 'robust', 'outlier_robust')
    handle_outliers : bool, default=True
        Whether to use advanced outlier handling
    
    Returns:
    --------
    Pipeline : sklearn.pipeline.Pipeline
    """
    steps = []
    
    # Advanced imputation
    steps.append(('advanced_imputer', AdvancedImputer(strategy=impute_strategy)))
    
    # Outlier handling
    if handle_outliers and scaler_type != 'outlier_robust':
        steps.append(('outlier_scaler', OutlierRobustScaler(method='iqr')))
    elif scaler_type == 'outlier_robust':
        steps.append(('outlier_scaler', OutlierRobustScaler(method='isolation_forest')))
    
    # Final scaling
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    
    if scaler_type in scalers:
        steps.append(('scaler', scalers[scaler_type]))
    
    return Pipeline(steps)


def create_minmax_scaler_pipeline(feature_range: tuple = (0, 1), 
                                impute_strategy: str = 'adaptive_median',
                                handle_outliers: bool = True) -> Pipeline:
    """
    Creates an advanced pipeline with imputation and MinMax scaling.
    
    Parameters:
    -----------
    feature_range : tuple, default=(0, 1)
        Desired range of transformed data
    impute_strategy : str, default='adaptive_median'
        Advanced imputation strategy
    handle_outliers : bool, default=True
        Whether to handle outliers before scaling
    
    Returns:
    --------
    Pipeline : sklearn.pipeline.Pipeline
    """
    steps = []
    
    steps.append(('advanced_imputer', AdvancedImputer(strategy=impute_strategy)))
    
    if handle_outliers:
        steps.append(('outlier_handler', OutlierRobustScaler(method='iqr')))
    
    steps.append(('minmax_scaler', MinMaxScaler(feature_range=feature_range)))
    
    return Pipeline(steps)


def create_robust_scaler_pipeline(impute_strategy: str = 'adaptive_median',
                                outlier_method: str = 'isolation_forest') -> Pipeline:
    """
    Creates an advanced pipeline with imputation and robust scaling.
    
    Parameters:
    -----------
    impute_strategy : str, default='adaptive_median'
        Advanced imputation strategy
    outlier_method : str, default='isolation_forest'
        Method for outlier detection ('iqr', 'isolation_forest', 'elliptic_envelope')
    
    Returns:
    --------
    Pipeline : sklearn.pipeline.Pipeline
    """
    return Pipeline([
        ('advanced_imputer', AdvancedImputer(strategy=impute_strategy)),
        ('outlier_robust_scaler', OutlierRobustScaler(method=outlier_method)),
        ('robust_scaler', RobustScaler())
    ])


def create_knn_imputer_pipeline(n_neighbors: int = 5, 
                              scaler_type: str = 'standard',
                              outlier_handling: bool = True) -> Pipeline:
    """
    Creates an advanced pipeline with KNN imputation and scaling.
    
    Parameters:
    -----------
    n_neighbors : int, default=5
        Number of neighboring samples to use for imputation
    scaler_type : str, default='standard'
        Type of scaler to use ('standard', 'minmax', 'robust', 'outlier_robust')
    outlier_handling : bool, default=True
        Whether to apply outlier handling
    
    Returns:
    --------
    Pipeline : sklearn.pipeline.Pipeline
    """
    steps = []
    
    steps.append(('knn_imputer', KNNImputer(n_neighbors=n_neighbors)))
    
    if outlier_handling:
        steps.append(('outlier_scaler', OutlierRobustScaler(method='isolation_forest')))
    
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
        'outlier_robust': OutlierRobustScaler(method='iqr')
    }
    
    steps.append(('scaler', scalers.get(scaler_type, StandardScaler())))
    
    return Pipeline(steps)


def create_power_transform_pipeline(method: str = 'auto',
                                  impute_strategy: str = 'adaptive_median',
                                  standardize: bool = True) -> Pipeline:
    """
    Creates an advanced pipeline with imputation and power transformation.
    
    Parameters:
    -----------
    method : str, default='auto'
        The power transform method ('auto', 'yeo-johnson', 'box-cox')
    impute_strategy : str, default='adaptive_median'
        Advanced imputation strategy
    standardize : bool, default=True
        Whether to standardize after transformation
    
    Returns:
    --------
    Pipeline : sklearn.pipeline.Pipeline
    """
    return Pipeline([
        ('advanced_imputer', AdvancedImputer(strategy=impute_strategy)),
        ('adaptive_power_transform', AdaptivePowerTransformer(method=method, standardize=standardize))
    ])


def create_comprehensive_numerical_pipeline(
    impute_strategy: str = 'adaptive_median',
    scaler_type: str = 'outlier_robust',
    power_transform: bool = False,
    power_transform_method: str = 'auto',
    handle_outliers: bool = True,
    outlier_method: str = 'isolation_forest'
) -> Pipeline:
    """
    Creates a comprehensive industrial-grade pipeline for numerical data preprocessing.
    
    Parameters:
    -----------
    impute_strategy : str, default='adaptive_median'
        Advanced imputation strategy
    scaler_type : str, default='outlier_robust'
        Type of scaler with built-in outlier handling
    power_transform : bool, default=False
        Whether to apply power transformation for normality
    power_transform_method : str, default='auto'
        Method for power transformation
    handle_outliers : bool, default=True
        Whether to handle outliers
    outlier_method : str, default='isolation_forest'
        Method for outlier detection
    
    Returns:
    --------
    Pipeline : sklearn.pipeline.Pipeline
    """
    steps = []
    
    # Step 1: Advanced statistical imputation
    steps.append(('advanced_imputer', AdvancedImputer(strategy=impute_strategy)))
    
    # Step 2: Outlier detection and handling
    if handle_outliers:
        steps.append(('outlier_detector', OutlierRobustScaler(method=outlier_method)))
    
    # Step 3: Optional power transformation for normality
    if power_transform:
        steps.append(('power_transformer', 
                     AdaptivePowerTransformer(method=power_transform_method, standardize=True)))
    
    # Step 4: Final scaling
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
        'outlier_robust': OutlierRobustScaler(method='iqr')
    }
    
    if scaler_type in scalers:
        steps.append(('final_scaler', scalers[scaler_type]))
    
    logger.info(f"Created comprehensive pipeline with {len(steps)} steps")
    
    return Pipeline(steps)