"""
Comprehensive tests for numerical preprocessing pipelines.
"""

import pytest
import numpy as np
from sklearn_preprocessing_pipelines.numerical import pipelines


class TestNumericalPipelines:
    """Test suite for numerical preprocessing pipelines."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Create sample data with some missing values and outliers
        self.sample_data = np.random.randn(100, 5)
        self.sample_data[10:15, 0] = np.nan  # Add missing values
        self.sample_data[20, 2] = 100  # Add outlier
    
    def test_standard_scaler_pipeline(self):
        """Test standard scaler pipeline."""
        pipeline = pipelines.create_standard_scaler_pipeline(
            impute_strategy='adaptive_median',
            scaler_type='standard',
            handle_outliers=True
        )
        
        assert pipeline is not None
        assert len(pipeline.steps) >= 2
        
        # Test fit_transform
        transformed = pipeline.fit_transform(self.sample_data)
        assert transformed is not None
        assert transformed.shape == self.sample_data.shape
        # Check that no NaN values remain
        assert not np.any(np.isnan(transformed))
    
    def test_minmax_scaler_pipeline(self):
        """Test MinMax scaler pipeline."""
        pipeline = pipelines.create_minmax_scaler_pipeline(
            feature_range=(0, 1),
            impute_strategy='adaptive_median',
            handle_outliers=True
        )
        
        assert pipeline is not None
        transformed = pipeline.fit_transform(self.sample_data)
        assert transformed is not None
        # Check values are in range (allowing for outliers)
        assert np.all((transformed >= -1) & (transformed <= 2))
    
    def test_robust_scaler_pipeline(self):
        """Test robust scaler pipeline."""
        pipeline = pipelines.create_robust_scaler_pipeline(
            impute_strategy='adaptive_median',
            outlier_method='isolation_forest'
        )
        
        assert pipeline is not None
        transformed = pipeline.fit_transform(self.sample_data)
        assert transformed is not None
        assert not np.any(np.isnan(transformed))
    
    def test_knn_imputer_pipeline(self):
        """Test KNN imputer pipeline."""
        pipeline = pipelines.create_knn_imputer_pipeline(
            n_neighbors=5,
            scaler_type='standard',
            outlier_handling=True
        )
        
        assert pipeline is not None
        transformed = pipeline.fit_transform(self.sample_data)
        assert transformed is not None
        assert not np.any(np.isnan(transformed))
    
    def test_power_transform_pipeline(self):
        """Test power transform pipeline."""
        # Use only positive data for box-cox
        positive_data = np.abs(self.sample_data) + 1
        
        pipeline = pipelines.create_power_transform_pipeline(
            method='auto',
            impute_strategy='adaptive_median',
            standardize=True
        )
        
        assert pipeline is not None
        transformed = pipeline.fit_transform(positive_data)
        assert transformed is not None
    
    def test_comprehensive_numerical_pipeline(self):
        """Test comprehensive numerical pipeline."""
        pipeline = pipelines.create_comprehensive_numerical_pipeline(
            impute_strategy='adaptive_median',
            scaler_type='outlier_robust',
            power_transform=False,
            handle_outliers=True
        )
        
        assert pipeline is not None
        transformed = pipeline.fit_transform(self.sample_data)
        assert transformed is not None


class TestAdvancedImputer:
    """Test AdvancedImputer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.data = np.random.randn(50, 3)
        self.data[5:10, 0] = np.nan
    
    def test_adaptive_median_strategy(self):
        """Test adaptive median imputation."""
        imputer = pipelines.AdvancedImputer(strategy='adaptive_median')
        imputer.fit(self.data)
        transformed = imputer.transform(self.data)
        
        assert not np.any(np.isnan(transformed))
        assert transformed.shape == self.data.shape
    
    def test_trimmed_mean_strategy(self):
        """Test trimmed mean imputation."""
        imputer = pipelines.AdvancedImputer(strategy='trimmed_mean')
        imputer.fit(self.data)
        transformed = imputer.transform(self.data)
        
        assert not np.any(np.isnan(transformed))
    
    def test_m_estimator_strategy(self):
        """Test M-estimator imputation."""
        imputer = pipelines.AdvancedImputer(strategy='m_estimator')
        imputer.fit(self.data)
        transformed = imputer.transform(self.data)
        
        assert not np.any(np.isnan(transformed))


class TestOutlierRobustScaler:
    """Test OutlierRobustScaler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.data = np.random.randn(100, 3)
        self.data[10, :] = [50, 50, 50]  # Add outliers
    
    def test_iqr_method(self):
        """Test IQR outlier detection."""
        scaler = pipelines.OutlierRobustScaler(method='iqr')
        scaler.fit(self.data)
        transformed = scaler.transform(self.data)
        
        assert transformed is not None
        assert transformed.shape == self.data.shape
    
    def test_zscore_method(self):
        """Test z-score outlier detection."""
        scaler = pipelines.OutlierRobustScaler(method='zscore')
        scaler.fit(self.data)
        transformed = scaler.transform(self.data)
        
        assert transformed is not None


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
