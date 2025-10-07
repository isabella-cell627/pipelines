"""
Image Data Preprocessing Pipelines

Provides reusable pipelines for preprocessing image data including:
- Resizing
- Normalization
- Flattening
- Color space conversion
"""

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np



from scipy import ndimage
from scipy.ndimage import zoom, affine_transform
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union, Tuple, Optional, Literal


class ImageResizer(BaseEstimator, TransformerMixin):
    """
    Image Resizer with multiple interpolation methods and performance optimizations.
    
    Features:
    - Multiple interpolation methods (nearest, linear, cubic, lanczos)
    - Anti-aliasing support
    - Batch processing optimization
    - Memory-efficient operations
    - Support for various image formats (2D, 3D, RGB, RGBA)
    - Aspect ratio preservation with padding/cropping options
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (64, 64),
                 interpolation: Literal['nearest', 'linear', 'cubic', 'lanczos'] = 'cubic',
                 anti_aliasing: bool = True,
                 preserve_aspect_ratio: bool = False,
                 padding_mode: Literal['constant', 'edge', 'reflect'] = 'constant',
                 padding_value: Union[int, float] = 0,
                 dtype: Optional[np.dtype] = None):
        """
        Initialize the ImageResizer.
        
        Parameters:
        -----------
        target_size : tuple of int
            Target size (height, width)
        interpolation : str
            Interpolation method ('nearest', 'linear', 'cubic', 'lanczos')
        anti_aliasing : bool
            Whether to apply anti-aliasing filter
        preserve_aspect_ratio : bool
            Whether to preserve aspect ratio (uses padding/cropping)
        padding_mode : str
            Padding mode when preserving aspect ratio
        padding_value : int/float
            Value to use for constant padding
        dtype : numpy dtype, optional
            Output data type (None preserves input dtype)
        """
        self.target_size = target_size
        self.interpolation = interpolation
        self.anti_aliasing = anti_aliasing
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        self.dtype = dtype
        
       
        self._interpolation_map = {
            'nearest': 0,
            'linear': 1,
            'cubic': 3,
            'lanczos': 5
        }
        
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate input parameters."""
        if len(self.target_size) != 2:
            raise ValueError("target_size must be a tuple of 2 integers (height, width)")
        
        if self.interpolation not in self._interpolation_map:
            raise ValueError(f"Interpolation must be one of {list(self._interpolation_map.keys())}")
        
        if not all(isinstance(dim, int) and dim > 0 for dim in self.target_size):
            raise ValueError("Target dimensions must be positive integers")
    
    def _get_zoom_factors(self, input_shape: Tuple[int, ...]) -> Tuple[float, ...]:
        """Calculate zoom factors for resizing."""
        if len(input_shape) < 2:
            raise ValueError("Input must have at least 2 dimensions")
        
        h, w = input_shape[:2]
        target_h, target_w = self.target_size
        
        if self.preserve_aspect_ratio:
            scale = min(target_h / h, target_w / w)
            return (scale, scale) + (1,) * (len(input_shape) - 2)
        else:
            return (target_h / h, target_w / w) + (1,) * (len(input_shape) - 2)
    
    def _apply_anti_aliasing(self, image: np.ndarray, zoom_factors: Tuple[float, ...]) -> np.ndarray:
        """Apply anti-aliasing filter if needed."""
        if not self.anti_aliasing:
            return image
        
        
        if all(zf >= 1.0 for zf in zoom_factors[:2]):
            return image
        
        
        sigma = max(0.5 / min(zoom_factors[:2]), 0.5)
        
        
        if len(image.shape) == 2:
            return ndimage.gaussian_filter(image, sigma=sigma)
        else:
            
            result = np.empty_like(image)
            for channel in range(image.shape[2]):
                result[..., channel] = ndimage.gaussian_filter(image[..., channel], sigma=sigma)
            return result
    
    def _pad_to_target(self, image: np.ndarray) -> np.ndarray:
        """Pad image to target size while preserving aspect ratio."""
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)
        
        if pad_h == 0 and pad_w == 0:
            return image
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        padding = [(pad_top, pad_bottom), (pad_left, pad_right)]
        if len(image.shape) == 3:
            padding.append((0, 0))
        
        if self.padding_mode == 'constant':
            return np.pad(image, padding, mode=self.padding_mode, constant_values=self.padding_value)
        else:
            return np.pad(image, padding, mode=self.padding_mode)
    
    def _crop_to_target(self, image: np.ndarray) -> np.ndarray:
        """Crop image to target size while preserving aspect ratio."""
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        crop_h = max(h - target_h, 0)
        crop_w = max(w - target_w, 0)
        
        if crop_h == 0 and crop_w == 0:
            return image
        
        start_h = crop_h // 2
        start_w = crop_w // 2
        
        return image[start_h:start_h + target_h, start_w:start_w + target_w]
    
    def _resize_single_image(self, image: np.ndarray) -> np.ndarray:
        """Resize a single image with advanced options."""
        original_dtype = image.dtype
        input_shape = image.shape
        
        
        if image.dtype.kind in 'iu':
            image = image.astype(np.float64)
        
        if self.preserve_aspect_ratio:
            
            zoom_factors = self._get_zoom_factors(input_shape)
            
            
            if self.anti_aliasing:
                image = self._apply_anti_aliasing(image, zoom_factors)
            
            
            resized = zoom(image, zoom_factors, order=self._interpolation_map[self.interpolation])
            
            
            if resized.shape[0] < self.target_size[0] or resized.shape[1] < self.target_size[1]:
                resized = self._pad_to_target(resized)
            else:
                resized = self._crop_to_target(resized)
        else:
            
            zoom_factors = self._get_zoom_factors(input_shape)
            
            
            if self.anti_aliasing:
                image = self._apply_anti_aliasing(image, zoom_factors)
            
            
            resized = zoom(image, zoom_factors, order=self._interpolation_map[self.interpolation])
        
        
        if self.dtype is not None:
            target_dtype = self.dtype
        else:
            target_dtype = original_dtype
        
        
        if target_dtype.kind in 'iu':
            resized = np.clip(resized, np.iinfo(target_dtype).min, np.iinfo(target_dtype).max)
        
        return resized.astype(target_dtype)
    
    def fit(self, X: np.ndarray, y=None) -> 'ImageResizer':
        """
        Fit the transformer (no operation for this transformer).
        
        Parameters:
        -----------
        X : array-like
            Input data
        y : array-like, optional
            Target values
            
        Returns:
        --------
        self : AdvancedImageResizer
            Returns self
        """
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input images to target size.
        
        Parameters:
        -----------
        X : numpy array
            Input images (2D, 3D, or 4D array)
            
        Returns:
        --------
        resized : numpy array
            Resized images
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        original_shape = X.shape
        original_ndim = X.ndim
        
        
        if original_ndim == 2:
            
            return self._resize_single_image(X)
        
        elif original_ndim == 3:
            
            if X.shape[-1] in [1, 3, 4]:  
                return self._resize_single_image(X)
            else:  
                return np.array([self._resize_single_image(img) for img in X])
        
        elif original_ndim == 4:
            
            return np.array([self._resize_single_image(img) for img in X])
        
        else:
            raise ValueError(f"Unsupported number of dimensions: {original_ndim}")
    
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Fit and transform in one operation.
        
        Parameters:
        -----------
        X : array-like
            Input data
        y : array-like, optional
            Target values
            
        Returns:
        --------
        resized : numpy array
            Resized images
        """
        return self.fit(X, y).transform(X)
    
    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            'target_size': self.target_size,
            'interpolation': self.interpolation,
            'anti_aliasing': self.anti_aliasing,
            'preserve_aspect_ratio': self.preserve_aspect_ratio,
            'padding_mode': self.padding_mode,
            'padding_value': self.padding_value,
            'dtype': self.dtype
        }
    
    def set_params(self, **params) -> 'ImageResizer':
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        self._validate_parameters()
        return self



import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom
from scipy.stats import variation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from typing import Union, Tuple, Optional, Literal, Dict, Any
import warnings
from functools import wraps
import time


def timing_decorator(func):
    """Decorator to measure execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if hasattr(args[0], 'verbose') and getattr(args[0], 'verbose', False):
            print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


class ImageNormalizer(BaseEstimator, TransformerMixin):
    """
    Advanced Image Normalization with multiple methods and batch processing optimization.
    
    Features:
    - Multiple normalization methods (minmax, standard, robust, etc.)
    - Per-channel and global normalization
    - Batch statistics and precomputed statistics
    - Outlier handling
    - Memory-efficient operations
    """
    
    def __init__(self, 
                 method: Literal['minmax', 'standard', 'robust', 'divide255', 
                               'mean', 'l2', 'max', 'adaptive'] = 'minmax',
                 per_channel: bool = True,
                 clip_outliers: bool = False,
                 clip_range: Tuple[float, float] = (0.1, 99.9),
                 epsilon: float = 1e-8,
                 dtype: np.dtype = np.float32,
                 verbose: bool = False):
        """
        Initialize the AdvancedImageNormalizer.
        
        Parameters:
        -----------
        method : str
            Normalization method
        per_channel : bool
            Whether to normalize each channel separately
        clip_outliers : bool
            Whether to clip outliers before normalization
        clip_range : tuple
            Percentile range for outlier clipping
        epsilon : float
            Small value to avoid division by zero
        dtype : numpy dtype
            Output data type
        verbose : bool
            Whether to print progress information
        """
        self.method = method
        self.per_channel = per_channel
        self.clip_outliers = clip_outliers
        self.clip_range = clip_range
        self.epsilon = epsilon
        self.dtype = dtype
        self.verbose = verbose
        
        # Statistics storage
        self.min_ = None
        self.max_ = None
        self.mean_ = None
        self.std_ = None
        self.median_ = None
        self.iqr_ = None
        
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate input parameters."""
        valid_methods = ['minmax', 'standard', 'robust', 'divide255', 'mean', 'l2', 'max', 'adaptive']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        
        if not all(0 <= p <= 100 for p in self.clip_range):
            raise ValueError("Clip range percentiles must be between 0 and 100")
    
    def _compute_statistics(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute comprehensive statistics for normalization."""
        stats = {}
        axis = (0, 1) if self.per_channel and X.ndim >= 3 else None
        
        if self.method in ['minmax', 'adaptive']:
            stats['min'] = X.min(axis=axis, keepdims=True)
            stats['max'] = X.max(axis=axis, keepdims=True)
        
        if self.method in ['standard', 'adaptive', 'mean']:
            stats['mean'] = X.mean(axis=axis, keepdims=True)
        
        if self.method in ['standard', 'adaptive']:
            stats['std'] = X.std(axis=axis, keepdims=True)
        
        if self.method == 'robust':
            stats['median'] = np.median(X, axis=axis, keepdims=True)
            stats['q1'] = np.percentile(X, 25, axis=axis, keepdims=True)
            stats['q3'] = np.percentile(X, 75, axis=axis, keepdims=True)
            stats['iqr'] = stats['q3'] - stats['q1']
        
        return stats
    
    def _clip_outliers(self, X: np.ndarray) -> np.ndarray:
        """Clip outliers based on percentiles."""
        if not self.clip_outliers:
            return X
        
        lower, upper = np.percentile(X, self.clip_range, axis=(0, 1) if self.per_channel else None)
        return np.clip(X, lower, upper)
    
    def _apply_normalization(self, X: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply normalization based on method and statistics."""
        X_normalized = X.astype(self.dtype)
        
        if self.method == 'minmax':
            min_val, max_val = stats['min'], stats['max']
            X_normalized = (X_normalized - min_val) / (max_val - min_val + self.epsilon)
        
        elif self.method == 'standard':
            X_normalized = (X_normalized - stats['mean']) / (stats['std'] + self.epsilon)
        
        elif self.method == 'robust':
            X_normalized = (X_normalized - stats['median']) / (stats['iqr'] + self.epsilon)
        
        elif self.method == 'divide255':
            X_normalized = X_normalized / 255.0
        
        elif self.method == 'mean':
            X_normalized = X_normalized - stats['mean']
        
        elif self.method == 'l2':
            norm = np.linalg.norm(X_normalized, axis=(1, 2), keepdims=True)
            X_normalized = X_normalized / (norm + self.epsilon)
        
        elif self.method == 'max':
            max_val = np.max(np.abs(X_normalized), axis=(1, 2), keepdims=True)
            X_normalized = X_normalized / (max_val + self.epsilon)
        
        elif self.method == 'adaptive':
            # Adaptive normalization combining multiple methods
            # Min-max normalization first
            min_val, max_val = stats['min'], stats['max']
            X_mm = (X_normalized - min_val) / (max_val - min_val + self.epsilon)
            
            # Standard normalization
            X_std = (X_normalized - stats['mean']) / (stats['std'] + self.epsilon)
            
            # Combine based on coefficient of variation
            cv = variation(X_normalized, axis=(0, 1))
            if self.per_channel and X_normalized.ndim >= 3:
                weight = 1 / (1 + np.exp(-cv))[np.newaxis, np.newaxis, :]
            else:
                weight = 1 / (1 + np.exp(-cv))
            
            X_normalized = weight * X_mm + (1 - weight) * X_std
        
        return np.clip(X_normalized, 0, 1) if self.method in ['minmax', 'divide255'] else X_normalized
    
    @timing_decorator
    def fit(self, X: np.ndarray, y=None) -> 'ImageNormalizer':
        """Fit the normalizer to compute statistics."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Compute and store statistics
        stats = self._compute_statistics(X)
        
        # Store statistics as attributes
        for key, value in stats.items():
            setattr(self, f"{key}_", value)
        
        return self
    
    @timing_decorator
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform images using fitted normalization."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        original_dtype = X.dtype
        
        # Clip outliers if enabled
        if self.clip_outliers:
            X = self._clip_outliers(X)
        
        # Apply normalization
        stats = {key: getattr(self, f"{key}_") for key in ['min', 'max', 'mean', 'std', 'median', 'iqr'] 
                if getattr(self, f"{key}_") is not None}
        
        X_normalized = self._apply_normalization(X, stats)
        
        return X_normalized.astype(self.dtype)
    
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform in one operation."""
        return self.fit(X, y).transform(X)


class ImageFlattener(BaseEstimator, TransformerMixin):
    """
    Advanced Image Flattener with multiple reshaping strategies.
    
    Features:
    - Multiple flattening strategies
    - Feature selection based on variance
    - PCA integration capability
    - Memory-efficient flattening
    """
    
    def __init__(self, 
                 strategy: Literal['flatten', 'spatial_pooling', 'patch_extraction'] = 'flatten',
                 pool_size: Optional[Tuple[int, int]] = None,
                 patch_size: Optional[Tuple[int, int]] = None,
                 max_features: Optional[int] = None,
                 variance_threshold: float = 0.0,
                 verbose: bool = False):
        """
        Initialize the AdvancedImageFlattener.
        
        Parameters:
        -----------
        strategy : str
            Flattening strategy
        pool_size : tuple, optional
            Pooling size for spatial pooling
        patch_size : tuple, optional
            Patch size for patch extraction
        max_features : int, optional
            Maximum number of features to keep
        variance_threshold : float
            Variance threshold for feature selection
        verbose : bool
            Whether to print progress information
        """
        self.strategy = strategy
        self.pool_size = pool_size
        self.patch_size = patch_size
        self.max_features = max_features
        self.variance_threshold = variance_threshold
        self.verbose = verbose
        
        # Feature selection attributes
        self.feature_mask_ = None
        self.variance_ = None
        
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate input parameters."""
        valid_strategies = ['flatten', 'spatial_pooling', 'patch_extraction']
        if self.strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
    
    def _spatial_pooling(self, X: np.ndarray) -> np.ndarray:
        """Apply spatial pooling before flattening."""
        if self.pool_size is None:
            pool_h, pool_w = 2, 2  # Default 2x2 pooling
        else:
            pool_h, pool_w = self.pool_size
        
        n_samples, h, w, channels = X.shape
        new_h, new_w = h // pool_h, w // pool_w
        
        # Simple average pooling
        pooled = np.zeros((n_samples, new_h, new_w, channels))
        for i in range(new_h):
            for j in range(new_w):
                pooled[:, i, j, :] = X[:, i*pool_h:(i+1)*pool_h, j*pool_w:(j+1)*pool_w, :].mean(axis=(1, 2))
        
        return pooled
    
    def _extract_patches(self, X: np.ndarray) -> np.ndarray:
        """Extract patches from images."""
        if self.patch_size is None:
            patch_h, patch_w = 8, 8  # Default 8x8 patches
        else:
            patch_h, patch_w = self.patch_size
        
        n_samples, h, w, channels = X.shape
        patches_per_image = (h - patch_h + 1) * (w - patch_w + 1)
        
        patches = np.zeros((n_samples * patches_per_image, patch_h, patch_w, channels))
        
        for sample_idx in range(n_samples):
            patch_idx = 0
            for i in range(h - patch_h + 1):
                for j in range(w - patch_w + 1):
                    patches[sample_idx * patches_per_image + patch_idx] = \
                        X[sample_idx, i:i+patch_h, j:j+patch_w, :]
                    patch_idx += 1
        
        return patches
    
    def _select_features(self, X_flat: np.ndarray) -> np.ndarray:
        """Select features based on variance threshold and maximum features."""
        if self.variance_threshold > 0 or self.max_features is not None:
            self.variance_ = np.var(X_flat, axis=0)
            
            # Create feature mask
            if self.max_features is not None:
                # Select top k features by variance
                top_k_indices = np.argsort(self.variance_)[-self.max_features:]
                self.feature_mask_ = np.zeros(X_flat.shape[1], dtype=bool)
                self.feature_mask_[top_k_indices] = True
            else:
                # Select features above variance threshold
                self.feature_mask_ = self.variance_ > self.variance_threshold
            
            return X_flat[:, self.feature_mask_]
        
        return X_flat
    
    @timing_decorator
    def fit(self, X: np.ndarray, y=None) -> 'ImageFlattener':
        """Fit the flattener (computes feature selection if needed)."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # For feature selection, we need to compute variances
        if self.variance_threshold > 0 or self.max_features is not None:
            X_temp = self.transform(X, fit_mode=True)
            # Variance computation is done in _select_features during transform
        
        return self
    
    @timing_decorator
    def transform(self, X: np.ndarray, fit_mode: bool = False) -> np.ndarray:
        """Transform images to flattened vectors."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        original_shape = X.shape
        original_ndim = X.ndim
        
        # Handle different input dimensions
        if original_ndim == 2:
            # Single grayscale image
            return X.flatten().reshape(1, -1)
        
        elif original_ndim == 3:
            # Single RGB image or batch of grayscale
            if X.shape[-1] in [1, 3, 4]:
                # Single color image
                X = X[np.newaxis, ...]
            else:
                # Batch of grayscale images
                X = X[..., np.newaxis] if X.ndim == 3 else X
        
        # Apply strategy
        if self.strategy == 'spatial_pooling':
            X_processed = self._spatial_pooling(X)
        elif self.strategy == 'patch_extraction':
            X_processed = self._extract_patches(X)
        else:  # 'flatten'
            X_processed = X
        
        # Flatten
        if self.strategy == 'patch_extraction':
            X_flat = X_processed.reshape(X_processed.shape[0], -1)
        else:
            X_flat = X_processed.reshape(X_processed.shape[0], -1)
        
        # Apply feature selection
        if not fit_mode:
            X_flat = self._select_features(X_flat)
        
        # Handle single image input
        if original_ndim == 3 and X.shape[-1] in [1, 3, 4]:
            return X_flat[0]  # Return 1D array for single image
        
        return X_flat
    
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform in one operation."""
        return self.fit(X, y).transform(X)


# Advanced Pipeline Creation Functions
def create_image_resize_pipeline(target_size: Tuple[int, int] = (64, 64),
                                         interpolation: str = 'cubic',
                                         preserve_aspect_ratio: bool = False) -> Pipeline:
    """
    Creates an advanced pipeline for resizing images.
    """
    
    
    return Pipeline([
        ('resizer', ImageResizer(
            target_size=target_size,
            interpolation=interpolation,
            preserve_aspect_ratio=preserve_aspect_ratio
        ))
    ])


def create_image_normalization_pipeline(method: str = 'minmax',
                                               per_channel: bool = True,
                                               clip_outliers: bool = True) -> Pipeline:
    """
    Creates an advanced pipeline for normalizing image pixel values.
    """
    return Pipeline([
        ('normalizer', ImageNormalizer(
            method=method,
            per_channel=per_channel,
            clip_outliers=clip_outliers
        ))
    ])


def create_image_flatten_pipeline(target_size: Tuple[int, int] = (64, 64),
                                         normalize_method: str = 'divide255',
                                         flatten_strategy: str = 'flatten',
                                         max_features: Optional[int] = None) -> Pipeline:
    """
    Creates an advanced pipeline for resizing, normalizing, and flattening images.
    """
    
    
    steps = [
        ('resizer', ImageResizer(target_size=target_size)),
        ('normalizer', ImageNormalizer(method=normalize_method))
    ]
    
    steps.append(('flattener', ImageFlattener(
        strategy=flatten_strategy,
        max_features=max_features
    )))
    
    return Pipeline(steps)


def create_comprehensive_image_pipeline(target_size: Tuple[int, int] = (128, 128),
                                               resize_interpolation: str = 'cubic',
                                               normalize_method: str = 'minmax',
                                               flatten_strategy: str = 'flatten',
                                               apply_scaling: bool = True,
                                               scaling_method: str = 'standard',
                                               feature_selection: bool = False,
                                               max_features: Optional[int] = None) -> Pipeline:
    """
    Creates a comprehensive advanced image preprocessing pipeline.
    """
    
    
    steps = [
        ('resizer', ImageResizer(
            target_size=target_size,
            interpolation=resize_interpolation
        )),
        ('normalizer', ImageNormalizer(method=normalize_method))
    ]
    
    if flatten_strategy != 'none':
        steps.append(('flattener', ImageFlattener(
            strategy=flatten_strategy,
            max_features=max_features if feature_selection else None
        )))
        
        if apply_scaling:
            if scaling_method == 'standard':
                steps.append(('scaler', StandardScaler()))
            elif scaling_method == 'robust':
                steps.append(('scaler', RobustScaler()))
    
    return Pipeline(steps)



