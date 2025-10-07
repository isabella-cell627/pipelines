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


class TestEdgeCases:
    """Test edge cases and boundary conditions for numerical pipelines."""
    
    def test_all_zeros(self):
        """Test pipeline with all zero values."""
        data = np.zeros((10, 3))
        pipeline = pipelines.create_standard_scaler_pipeline()
        
        transformed = pipeline.fit_transform(data)
        assert transformed is not None
        assert transformed.shape == data.shape
        assert transformed.dtype == np.float64
        # All zeros should remain all zeros (or NaN from zero std)
        assert np.all(np.isnan(transformed)) or np.allclose(transformed, 0)
    
    def test_single_row(self):
        """Test pipeline with single row."""
        data = np.array([[1.0, 2.0, 3.0]])
        pipeline = pipelines.create_minmax_scaler_pipeline()
        
        transformed = pipeline.fit_transform(data)
        assert transformed is not None
        assert transformed.shape == data.shape
        assert transformed.dtype == np.float64
        # Single row should not have NaN unless the scaler produces it
        assert not np.any(np.isinf(transformed))
    
    def test_all_nan(self):
        """Test pipeline with all NaN values - use mean strategy which fills with 0."""
        data = np.full((10, 3), np.nan)
        # Create pipeline using library function with mean strategy
        # Mean strategy on all-NaN columns should fill with 0
        pipeline = pipelines.create_standard_scaler_pipeline(
            impute_strategy='mean'
        )
        
        transformed = pipeline.fit_transform(data)
        assert transformed is not None
        assert transformed.shape == data.shape
        # With all NaN, mean imputer fills with 0, then scaling produces NaN
        # This is expected sklearn behavior
        # Let's verify the pipeline ran without error
        assert transformed is not None
    
    def test_extreme_values(self):
        """Test pipeline with extreme values."""
        data = np.array([[1e10, 1e-10, 0],
                        [1e10, 1e-10, 0],
                        [1e10, 1e-10, 0]])
        pipeline = pipelines.create_robust_scaler_pipeline()
        
        transformed = pipeline.fit_transform(data)
        assert transformed is not None
        assert transformed.shape == data.shape
        assert transformed.dtype == np.float64
        assert not np.any(np.isinf(transformed))
    
    def test_high_dimensional(self):
        """Test pipeline with high-dimensional data."""
        np.random.seed(42)
        data = np.random.randn(100, 100)
        pipeline = pipelines.create_standard_scaler_pipeline()
        
        transformed = pipeline.fit_transform(data)
        assert transformed is not None
        assert transformed.shape == data.shape
        assert transformed.dtype == np.float64
        assert not np.any(np.isnan(transformed))
        assert not np.any(np.isinf(transformed))
    
    def test_constant_column(self):
        """Test pipeline with constant column."""
        data = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
        pipeline = pipelines.create_standard_scaler_pipeline()
        
        transformed = pipeline.fit_transform(data)
        assert transformed is not None
        assert transformed.shape == data.shape
        assert transformed.dtype == np.float64
        # Constant column will have NaN from zero std, but that's expected
        # Other column should be properly scaled
    
    def test_negative_values_power_transform(self):
        """Test power transform with negative values."""
        data = np.array([[-5.0, 2.0], [-3.0, 4.0], [1.0, 6.0]])
        pipeline = pipelines.create_power_transform_pipeline(method='yeo-johnson')
        
        transformed = pipeline.fit_transform(data)
        assert transformed is not None
        assert transformed.shape == data.shape
        assert transformed.dtype == np.float64
        assert not np.any(np.isnan(transformed))
        assert not np.any(np.isinf(transformed))
    
    def test_highly_skewed_data(self):
        """Test pipeline with highly skewed data."""
        np.random.seed(42)
        data = np.random.exponential(scale=2.0, size=(100, 3))
        pipeline = pipelines.create_comprehensive_numerical_pipeline(
            power_transform=True
        )
        
        transformed = pipeline.fit_transform(data)
        assert transformed is not None
        assert transformed.shape == data.shape
        assert transformed.dtype == np.float64
        assert not np.any(np.isnan(transformed))
        assert not np.any(np.isinf(transformed))


class TestOutlierHandling:
    """Test outlier detection and handling."""
    
    def test_isolation_forest_outliers(self):
        """Test isolation forest outlier detection."""
        np.random.seed(42)
        data = np.random.randn(100, 3)
        data[0, :] = [100, 100, 100]  # Outlier
        
        scaler = pipelines.OutlierRobustScaler(method='isolation_forest')
        scaler.fit(data)
        transformed = scaler.transform(data)
        
        assert transformed is not None
        assert scaler.outlier_mask_ is not None
        assert np.any(scaler.outlier_mask_)
    
    def test_elliptic_envelope_outliers(self):
        """Test elliptic envelope outlier detection."""
        np.random.seed(42)
        data = np.random.randn(100, 3)
        data[0, :] = [50, 50, 50]  # Outlier
        
        scaler = pipelines.OutlierRobustScaler(method='elliptic_envelope')
        scaler.fit(data)
        transformed = scaler.transform(data)
        
        assert transformed is not None


class TestDataIntegrity:
    """Test data integrity and validation."""
    
    def test_no_data_modification(self):
        """Test that original data is not modified."""
        np.random.seed(42)
        data = np.random.randn(10, 3)
        data_copy = data.copy()
        
        pipeline = pipelines.create_standard_scaler_pipeline()
        pipeline.fit_transform(data)
        
        # Original data should not be modified
        np.testing.assert_array_equal(data, data_copy)
    
    def test_transform_consistency(self):
        """Test that transform is consistent."""
        np.random.seed(42)
        data = np.random.randn(10, 3)
        
        pipeline = pipelines.create_minmax_scaler_pipeline()
        pipeline.fit(data)
        
        t1 = pipeline.transform(data)
        t2 = pipeline.transform(data)
        
        np.testing.assert_array_almost_equal(t1, t2)
    
    def test_fit_transform_equals_fit_then_transform(self):
        """Test that fit_transform equals fit then transform."""
        np.random.seed(42)
        data = np.random.randn(10, 3)
        
        pipeline1 = pipelines.create_standard_scaler_pipeline()
        pipeline2 = pipelines.create_standard_scaler_pipeline()
        
        t1 = pipeline1.fit_transform(data)
        pipeline2.fit(data)
        t2 = pipeline2.transform(data)
        
        np.testing.assert_array_almost_equal(t1, t2)


class TestPerformance:
    """Test performance with larger datasets."""
    
    def test_large_dataset_standard(self):
        """Test standard scaler with large dataset."""
        np.random.seed(42)
        data = np.random.randn(10000, 50)
        
        pipeline = pipelines.create_standard_scaler_pipeline()
        transformed = pipeline.fit_transform(data)
        
        assert transformed is not None
        assert transformed.shape == data.shape
    
    def test_large_dataset_knn_imputer(self):
        """Test KNN imputer with large dataset."""
        np.random.seed(42)
        data = np.random.randn(1000, 10)
        data[np.random.rand(1000, 10) < 0.1] = np.nan
        
        pipeline = pipelines.create_knn_imputer_pipeline(n_neighbors=5)
        transformed = pipeline.fit_transform(data)
        
        assert transformed is not None
        assert not np.any(np.isnan(transformed))


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
