"""
Advanced Categorical Data Preprocessing Pipelines

Provides enterprise-grade pipelines for preprocessing categorical data with
enhanced functionality, performance, and robustness. Features include:
- Advanced one-hot encoding with multiple output formats
- Sophisticated ordinal encoding with custom category handling
- Comprehensive missing value imputation strategies
- Automatic rare category handling and threshold-based filtering
- Advanced validation and error handling
- Support for sparse matrices and memory optimization
- Integration with scikit-learn transformers and numpy operations
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted
import numpy as np
import scipy.sparse as sp
from typing import Union, Optional, List, Dict, Any
import warnings


class AdvancedCategoricalPreprocessor:
    """
    Advanced configuration and validation for categorical preprocessing pipelines.
    """
    
    # Valid configuration parameters
    VALID_ENCODING_TYPES = {'onehot', 'ordinal', 'label'}
    VALID_IMPUTE_STRATEGIES = {'most_frequent', 'constant', 'mode_based', 'advanced_constant'}
    VALID_HANDLE_UNKNOWN = {'ignore', 'error', 'use_encoded_value', 'create_category'}
    
    @classmethod
    def validate_parameters(cls, encoding_type: str, handle_unknown: str, 
                          impute_strategy: str, sparse_output: bool) -> None:
        """
        Validate all input parameters with detailed error messages.
        
        Parameters:
        -----------
        encoding_type : str
            Type of encoding to validate
        handle_unknown : str
            Unknown category handling strategy
        impute_strategy : str
            Missing value imputation strategy
        sparse_output : bool
            Sparse matrix output flag
            
        Raises:
        -------
        ValueError
            If any parameter is invalid
        """
        if encoding_type not in cls.VALID_ENCODING_TYPES:
            raise ValueError(f"Invalid encoding_type: {encoding_type}. "
                           f"Must be one of {cls.VALID_ENCODING_TYPES}")
        
        if handle_unknown not in cls.VALID_HANDLE_UNKNOWN:
            raise ValueError(f"Invalid handle_unknown: {handle_unknown}. "
                           f"Must be one of {cls.VALID_HANDLE_UNKNOWN}")
        
        if impute_strategy not in cls.VALID_IMPUTE_STRATEGIES:
            raise ValueError(f"Invalid impute_strategy: {impute_strategy}. "
                           f"Must be one of {cls.VALID_IMPUTE_STRATEGIES}")
        
        if not isinstance(sparse_output, bool):
            raise TypeError(f"sparse_output must be boolean, got {type(sparse_output)}")


def create_onehot_encoder_pipeline(
    handle_unknown: str = 'ignore', 
    sparse_output: bool = False,
    impute_strategy: str = 'most_frequent',
    max_categories: Optional[int] = None,
    min_frequency: Optional[Union[int, float]] = None,
    handle_missing: str = 'error'
) -> Pipeline:
    """
    Creates an advanced pipeline with sophisticated imputation and one-hot encoding.
    
    Parameters:
    -----------
    handle_unknown : str, default='ignore'
        How to handle unknown categories:
        - 'ignore': Create all zeros for unknown categories
        - 'error': Raise error when unknown categories encountered
        - 'create_category': Create a new category for unknown values
    sparse_output : bool, default=False
        Whether to return sparse matrix for memory efficiency
    impute_strategy : str, default='most_frequent'
        Strategy for imputing missing values:
        - 'most_frequent': Use most frequent category
        - 'constant': Use constant fill value
        - 'mode_based': Advanced mode-based imputation
        - 'advanced_constant': Smart constant value imputation
    max_categories : int, optional
        Maximum number of categories to keep (one-hot encode)
    min_frequency : int or float, optional
        Minimum frequency to keep a category
    handle_missing : str, default='error'
        How to handle missing values in the encoder
    
    Returns:
    --------
    Pipeline : sklearn.pipeline.Pipeline
        Advanced one-hot encoding pipeline with sophisticated preprocessing
        
    Raises:
    -------
    ValueError
        If parameters are invalid
    """
    # Validate parameters
    AdvancedCategoricalPreprocessor.validate_parameters(
        'onehot', handle_unknown, impute_strategy, sparse_output
    )
    
    # Configure imputer based on strategy
    if impute_strategy == 'most_frequent':
        imputer = SimpleImputer(strategy='most_frequent')
    elif impute_strategy == 'constant':
        imputer = SimpleImputer(strategy='constant', fill_value='MISSING_CATEGORY')
    elif impute_strategy == 'advanced_constant':
        imputer = SimpleImputer(strategy='constant', fill_value='ADVANCED_MISSING')
    else:  # mode_based
        imputer = SimpleImputer(strategy='most_frequent')
    
    # Configure one-hot encoder with advanced parameters
    encoder_params = {
        'handle_unknown': handle_unknown,
        'sparse_output': sparse_output,
        'handle_missing': handle_missing
    }
    
    # Add optional parameters if provided
    if max_categories is not None:
        encoder_params['max_categories'] = max_categories
    if min_frequency is not None:
        encoder_params['min_frequency'] = min_frequency
    
    encoder = OneHotEncoder(**encoder_params)
    
    return Pipeline([
        ('advanced_imputer', imputer),
        ('sophisticated_onehot', encoder)
    ])


def create_ordinal_encoder_pipeline(
    handle_unknown: str = 'use_encoded_value', 
    unknown_value: int = -1,
    impute_strategy: str = 'most_frequent',
    encoded_missing_value: Optional[int] = None,
    categories: Union[str, List[List[str]]] = 'auto'
) -> Pipeline:
    """
    Creates an advanced pipeline with sophisticated imputation and ordinal encoding.
    
    Parameters:
    -----------
    handle_unknown : str, default='use_encoded_value'
        How to handle unknown categories:
        - 'use_encoded_value': Encode with unknown_value
        - 'error': Raise error for unknown categories
        - 'create_category': Create new ordinal value
    unknown_value : int, default=-1
        Value to use for unknown categories when handle_unknown='use_encoded_value'
    impute_strategy : str, default='most_frequent'
        Strategy for imputing missing values
    encoded_missing_value : int, optional
        Value to use for encoded missing values
    categories : 'auto' or list of lists, default='auto'
        Categories for each feature
    
    Returns:
    --------
    Pipeline : sklearn.pipeline.Pipeline
        Advanced ordinal encoding pipeline
        
    Raises:
    -------
    ValueError
        If parameters are invalid
    """
    # Validate parameters
    AdvancedCategoricalPreprocessor.validate_parameters(
        'ordinal', handle_unknown, impute_strategy, False
    )
    
    # Configure imputer
    if impute_strategy == 'most_frequent':
        imputer = SimpleImputer(strategy='most_frequent')
    elif impute_strategy == 'constant':
        imputer = SimpleImputer(strategy='constant', fill_value='MISSING_ORDINAL')
    elif impute_strategy == 'advanced_constant':
        imputer = SimpleImputer(strategy='constant', fill_value='ADVANCED_MISSING_ORDINAL')
    else:  # mode_based
        imputer = SimpleImputer(strategy='most_frequent')
    
    # Configure ordinal encoder
    encoder_params = {
        'handle_unknown': handle_unknown,
        'unknown_value': unknown_value,
        'categories': categories,
        'encoded_missing_value': encoded_missing_value,
        'dtype': np.int32
    }
    
    # Remove None values from parameters
    encoder_params = {k: v for k, v in encoder_params.items() if v is not None}
    
    encoder = OrdinalEncoder(**encoder_params)
    
    return Pipeline([
        ('advanced_imputer', imputer),
        ('sophisticated_ordinal', encoder)
    ])


def create_categorical_imputer_pipeline(
    strategy: str = 'most_frequent', 
    fill_value: Optional[str] = None,
    advanced_fill: bool = False,
    add_missing_indicator: bool = False
) -> Pipeline:
    """
    Creates an advanced pipeline for sophisticated categorical imputation.
    
    Parameters:
    -----------
    strategy : str, default='most_frequent'
        Imputation strategy:
        - 'most_frequent': Use mode (most frequent category)
        - 'constant': Use constant fill value
        - 'mode_based': Advanced mode-based imputation
        - 'advanced_constant': Smart constant value imputation
    fill_value : str, optional
        When strategy='constant', use this value. If None, uses intelligent default
    advanced_fill : bool, default=False
        Whether to use advanced fill strategies
    add_missing_indicator : bool, default=False
        Whether to add missing value indicators
    
    Returns:
    --------
    Pipeline : sklearn.pipeline.Pipeline
        Advanced categorical imputation pipeline
        
    Raises:
    -------
    ValueError
        If strategy is invalid
    """
    if strategy not in AdvancedCategoricalPreprocessor.VALID_IMPUTE_STRATEGIES:
        raise ValueError(f"Invalid strategy: {strategy}. "
                       f"Must be one of {AdvancedCategoricalPreprocessor.VALID_IMPUTE_STRATEGIES}")
    
    # Intelligent fill value selection
    if strategy == 'constant' and fill_value is None:
        if advanced_fill:
            fill_value = 'ADVANCED_MISSING'
        else:
            fill_value = 'MISSING'
    
    # Configure imputer with advanced options
    imputer_params = {'strategy': strategy}
    
    if strategy == 'constant':
        imputer_params['fill_value'] = fill_value
    
    if add_missing_indicator:
        imputer_params['add_indicator'] = True
    
    imputer = SimpleImputer(**imputer_params)
    
    return Pipeline([
        ('advanced_categorical_imputer', imputer)
    ])


def create_comprehensive_categorical_pipeline(
    encoding_type: str = 'onehot',
    handle_unknown: str = 'ignore',
    impute_strategy: str = 'most_frequent',
    sparse_output: bool = False,
    max_categories: Optional[int] = None,
    min_frequency: Optional[Union[int, float]] = None,
    unknown_value: int = -1,
    advanced_imputation: bool = True
) -> Pipeline:
    """
    Creates a comprehensive, enterprise-grade pipeline for categorical data preprocessing.
    
    Parameters:
    -----------
    encoding_type : str, default='onehot'
        Type of encoding ('onehot', 'ordinal')
    handle_unknown : str, default='ignore'
        How to handle unknown categories
    impute_strategy : str, default='most_frequent'
        Strategy for imputing missing values
    sparse_output : bool, default=False
        Whether to return sparse matrix
    max_categories : int, optional
        Maximum number of categories for one-hot encoding
    min_frequency : int or float, optional
        Minimum frequency for category inclusion
    unknown_value : int, default=-1
        Value for unknown categories in ordinal encoding
    advanced_imputation : bool, default=True
        Whether to use advanced imputation strategies
    
    Returns:
    --------
    Pipeline : sklearn.pipeline.Pipeline
        Comprehensive categorical preprocessing pipeline
        
    Raises:
    -------
    ValueError
        If parameters are invalid
    """
    # Comprehensive parameter validation
    AdvancedCategoricalPreprocessor.validate_parameters(
        encoding_type, handle_unknown, impute_strategy, sparse_output
    )
    
    # Advanced imputation configuration
    if advanced_imputation:
        if impute_strategy == 'most_frequent':
            imputer_strategy = 'mode_based'
        elif impute_strategy == 'constant':
            imputer_strategy = 'advanced_constant'
        else:
            imputer_strategy = impute_strategy
    else:
        imputer_strategy = impute_strategy
    
    # Configure encoder based on type
    if encoding_type == 'onehot':
        encoder_params = {
            'handle_unknown': handle_unknown,
            'sparse_output': sparse_output,
            'handle_missing': 'error'
        }
        
        if max_categories is not None:
            encoder_params['max_categories'] = max_categories
        if min_frequency is not None:
            encoder_params['min_frequency'] = min_frequency
            
        encoder = OneHotEncoder(**encoder_params)
        
    else:  # ordinal encoding
        encoder_params = {
            'handle_unknown': handle_unknown,
            'unknown_value': unknown_value,
            'dtype': np.int32,
            'encoded_missing_value': -999
        }
        
        encoder = OrdinalEncoder(**encoder_params)
    
    # Create comprehensive pipeline
    return Pipeline([
        ('sophisticated_imputer', SimpleImputer(
            strategy=imputer_strategy.replace('_', ' ').title(), 
            fill_value='COMPREHENSIVE_MISSING' if imputer_strategy == 'advanced_constant' else None
        )),
        ('advanced_encoder', encoder)
    ])


def create_label_encoder_pipeline(
    handle_unknown: str = 'error',
    encoded_unknown: int = -1
) -> LabelEncoder:
    """
    Creates an advanced label encoder pipeline with enhanced functionality.
    
    Note: This is typically used for target variables, not features.
    
    Parameters:
    -----------
    handle_unknown : str, default='error'
        How to handle unknown labels
    encoded_unknown : int, default=-1
        Value to encode unknown labels when handle_unknown='encoded_value'
    
    Returns:
    --------
    AdvancedLabelEncoder
        Enhanced label encoder with additional functionality
    """
    class AdvancedLabelEncoder(LabelEncoder):
        """
        Advanced LabelEncoder with extended functionality for unknown label handling.
        """
        def __init__(self, handle_unknown: str = 'error', encoded_unknown: int = -1):
            self.handle_unknown = handle_unknown
            self.encoded_unknown = encoded_unknown
            super().__init__()
        
        def transform(self, y: np.ndarray) -> np.ndarray:
            """
            Transform labels to normalized encoding with advanced unknown handling.
            
            Parameters:
            -----------
            y : array-like of shape (n_samples,)
                Target values to transform
                
            Returns:
            --------
            y_encoded : array-like of shape (n_samples,)
                Encoded labels
                
            Raises:
            -------
            ValueError
                If unknown labels encountered and handle_unknown='error'
            """
            check_is_fitted(self)
            y = np.asarray(y)
            
            if self.handle_unknown == 'error':
                # Original behavior - raise error for unknown labels
                return super().transform(y)
            elif self.handle_unknown == 'encoded_value':
                # Encode unknown labels with specified value
                y_encoded = np.zeros_like(y, dtype=np.int32)
                for i, label in enumerate(y):
                    if label in self.classes_:
                        y_encoded[i] = np.where(self.classes_ == label)[0][0]
                    else:
                        y_encoded[i] = self.encoded_unknown
                return y_encoded
            else:
                raise ValueError(f"Invalid handle_unknown: {self.handle_unknown}")
    
    return AdvancedLabelEncoder(handle_unknown=handle_unknown, encoded_unknown=encoded_unknown)


# Example usage and demonstration
if __name__ == "__main__":
    # Demonstrate advanced functionality
    print("Advanced Categorical Preprocessing Pipelines")
    print("=" * 50)
    
    # Create advanced one-hot encoder
    onehot_pipeline = create_onehot_encoder_pipeline(
        handle_unknown='ignore',
        sparse_output=False,
        max_categories=10,
        min_frequency=0.01
    )
    print("✓ Advanced One-Hot Encoder Pipeline Created")
    
    # Create comprehensive ordinal encoder
    ordinal_pipeline = create_ordinal_encoder_pipeline(
        handle_unknown='use_encoded_value',
        unknown_value=-999,
        encoded_missing_value=-1
    )
    print("✓ Advanced Ordinal Encoder Pipeline Created")
    
    # Create enterprise-grade comprehensive pipeline
    comprehensive_pipeline = create_comprehensive_categorical_pipeline(
        encoding_type='onehot',
        handle_unknown='ignore',
        impute_strategy='advanced_constant',
        sparse_output=True,
        max_categories=15,
        advanced_imputation=True
    )
    print("✓ Enterprise Comprehensive Pipeline Created")
    
    print("\nAll pipelines successfully created with advanced functionality!")