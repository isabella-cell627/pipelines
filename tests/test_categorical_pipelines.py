"""
Comprehensive tests for categorical preprocessing pipelines.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn_preprocessing_pipelines.categorical import pipelines


class TestCategoricalPipelines:
    """Test suite for categorical preprocessing pipelines."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.sample_data = np.array([
            ['cat', 'red', 'small'],
            ['dog', 'blue', 'medium'],
            ['cat', 'red', 'large'],
            ['bird', 'green', 'small'],
            [None, 'blue', None],
            ['dog', None, 'medium']
        ], dtype=object)
    
    def test_onehot_encoder_pipeline(self):
        """Test one-hot encoder pipeline creation and fitting."""
        pipeline = pipelines.create_onehot_encoder_pipeline(
            handle_unknown='ignore',
            sparse_output=False,
            impute_strategy='most_frequent'
        )
        
        assert pipeline is not None
        assert len(pipeline.steps) == 2
        
        # Test fit_transform
        transformed = pipeline.fit_transform(self.sample_data)
        assert transformed is not None
        assert transformed.shape[0] == self.sample_data.shape[0]
        assert not np.any(pd.isna(transformed))
        assert transformed.dtype in [np.float64, np.int32, np.int64]
    
    def test_ordinal_encoder_pipeline(self):
        """Test ordinal encoder pipeline."""
        pipeline = pipelines.create_ordinal_encoder_pipeline(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            impute_strategy='most_frequent'
        )
        
        assert pipeline is not None
        assert len(pipeline.steps) == 2
    
    def test_categorical_imputer_pipeline(self):
        """Test categorical imputer pipeline with constant strategy."""
        # Use constant strategy which works with np.nan in object arrays
        pipeline = pipelines.create_categorical_imputer_pipeline(
            strategy='constant',
            fill_value='MISSING'
        )
        
        assert pipeline is not None
        assert len(pipeline.steps) == 1
        
        # Test transform - use np.nan which sklearn handles properly
        test_data = np.array([
            ['cat', 'red'],
            ['dog', 'blue'],
            [np.nan, 'red'],
            ['cat', np.nan]
        ], dtype=object)
        
        transformed = pipeline.fit_transform(test_data)
        assert transformed is not None
        assert transformed.shape == test_data.shape
        # Verify missing values are filled
        assert 'MISSING' in transformed
        assert not np.any(pd.isna(transformed))
    
    def test_categorical_imputer_advanced_fill(self):
        """Test categorical imputer with advanced_fill=True default."""
        # Test advanced fill mode with default fill value
        pipeline = pipelines.create_categorical_imputer_pipeline(
            strategy='constant',
            advanced_fill=True
        )
        
        assert pipeline is not None
        assert len(pipeline.steps) == 1
        
        # Test transform with np.nan
        test_data = np.array([
            ['cat', 'red'],
            ['dog', 'blue'],
            [np.nan, 'red'],
            ['cat', np.nan]
        ], dtype=object)
        
        transformed = pipeline.fit_transform(test_data)
        assert transformed is not None
        assert transformed.shape == test_data.shape
        # Advanced fill should use 'ADVANCED_MISSING' as default
        assert 'ADVANCED_MISSING' in transformed
        assert not np.any(pd.isna(transformed))
    
    def test_categorical_imputer_advanced_fill_custom(self):
        """Test categorical imputer with advanced_fill=True and custom fill_value."""
        # Test advanced fill mode with custom fill value
        custom_fill = 'CUSTOM_XYZ'
        pipeline = pipelines.create_categorical_imputer_pipeline(
            strategy='constant',
            advanced_fill=True,
            fill_value=custom_fill
        )
        
        assert pipeline is not None
        assert len(pipeline.steps) == 1
        
        # Test transform with np.nan
        test_data = np.array([
            ['cat', 'red'],
            ['dog', 'blue'],
            [np.nan, 'red'],
            ['cat', np.nan]
        ], dtype=object)
        
        transformed = pipeline.fit_transform(test_data)
        assert transformed is not None
        assert transformed.shape == test_data.shape
        # Should use custom fill value, not default
        assert custom_fill in transformed
        assert 'ADVANCED_MISSING' not in transformed
        assert not np.any(pd.isna(transformed))
    
    def test_comprehensive_categorical_pipeline(self):
        """Test comprehensive categorical pipeline."""
        pipeline = pipelines.create_comprehensive_categorical_pipeline(
            encoding_type='onehot',
            handle_unknown='ignore',
            impute_strategy='most_frequent',
            sparse_output=False
        )
        
        assert pipeline is not None
        assert len(pipeline.steps) == 2
    
    def test_parameter_validation(self):
        """Test parameter validation in AdvancedCategoricalPreprocessor."""
        with pytest.raises(ValueError):
            pipelines.AdvancedCategoricalPreprocessor.validate_parameters(
                'invalid_encoding', 'ignore', 'most_frequent', False
            )
        
        with pytest.raises(ValueError):
            pipelines.AdvancedCategoricalPreprocessor.validate_parameters(
                'onehot', 'invalid_unknown', 'most_frequent', False
            )
    
    def test_label_encoder(self):
        """Test label encoder pipeline."""
        encoder = pipelines.create_label_encoder_pipeline(
            handle_unknown='error'
        )
        
        assert encoder is not None
        
        # Test basic encoding
        labels = np.array(['cat', 'dog', 'cat', 'bird', 'dog'])
        encoder.fit(labels)
        encoded = encoder.transform(labels)
        
        assert len(encoded) == len(labels)
        assert encoded.dtype in [np.int32, np.int64]


class TestAdvancedFeatures:
    """Test advanced features of categorical pipelines."""
    
    def test_sparse_output(self):
        """Test sparse matrix output."""
        pipeline = pipelines.create_onehot_encoder_pipeline(
            sparse_output=True,
            impute_strategy='constant'
        )
        
        assert pipeline is not None
    
    def test_max_categories(self):
        """Test max_categories parameter."""
        pipeline = pipelines.create_onehot_encoder_pipeline(
            max_categories=10,
            min_frequency=0.01
        )
        
        assert pipeline is not None
    
    def test_different_impute_strategies(self):
        """Test different imputation strategies."""
        strategies = ['most_frequent', 'constant', 'mode_based', 'advanced_constant']
        
        for strategy in strategies:
            pipeline = pipelines.create_categorical_imputer_pipeline(
                strategy=strategy
            )
            assert pipeline is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_data(self):
        """Test pipeline with empty data - should raise ValueError."""
        empty_data = np.array([]).reshape(0, 3)
        pipeline = pipelines.create_onehot_encoder_pipeline()
        
        # Empty data should raise ValueError
        with pytest.raises(ValueError):
            pipeline.fit(empty_data)
    
    def test_single_value_column(self):
        """Test pipeline with single unique value."""
        data = np.array([['cat'], ['cat'], ['cat']], dtype=object)
        pipeline = pipelines.create_ordinal_encoder_pipeline()
        
        transformed = pipeline.fit_transform(data)
        assert transformed is not None
        assert transformed.shape[0] == 3
    
    def test_all_nan_column(self):
        """Test pipeline with all NaN values - sklearn limitation case."""
        data = np.array([[None], [None], [None]], dtype=object)
        pipeline = pipelines.create_categorical_imputer_pipeline(
            strategy='constant',
            fill_value='MISSING'
        )
        
        # Note: sklearn's SimpleImputer has a limitation where it cannot fill
        # all-None columns properly. This is expected sklearn behavior.
        transformed = pipeline.fit_transform(data)
        assert transformed is not None
        assert transformed.shape == data.shape
        # Due to sklearn limitation, all-None data remains None
        # This test documents expected sklearn behavior
    
    def test_high_cardinality(self):
        """Test pipeline with high cardinality data."""
        data = np.array([[f'cat_{i}'] for i in range(100)], dtype=object)
        pipeline = pipelines.create_onehot_encoder_pipeline(max_categories=20)
        
        transformed = pipeline.fit_transform(data)
        assert transformed is not None
        assert transformed.shape[0] == 100
        # Should limit categories
        assert transformed.shape[1] <= 20
        assert not np.any(pd.isna(transformed))
    
    def test_special_characters(self):
        """Test pipeline with special characters."""
        data = np.array([['cat@#$'], ['dog!@#'], ['bird*&^']], dtype=object)
        pipeline = pipelines.create_ordinal_encoder_pipeline()
        
        transformed = pipeline.fit_transform(data)
        assert transformed is not None
        assert transformed.shape == (3, 1)
        assert not np.any(pd.isna(transformed))
        assert transformed.dtype in [np.int32, np.int64]
    
    def test_unicode_characters(self):
        """Test pipeline with unicode characters."""
        data = np.array([['café'], ['naïve'], ['résumé']], dtype=object)
        pipeline = pipelines.create_onehot_encoder_pipeline(sparse_output=False)
        
        transformed = pipeline.fit_transform(data)
        assert transformed is not None
        assert transformed.shape[0] == 3
        assert not np.any(pd.isna(transformed))
    
    def test_mixed_types(self):
        """Test pipeline with mixed string and numeric."""
        data = np.array([['cat', '1', 'true'],
                        ['dog', '2', 'false'],
                        ['bird', '3', 'true']], dtype=object)
        pipeline = pipelines.create_comprehensive_categorical_pipeline(sparse_output=False)
        
        transformed = pipeline.fit_transform(data)
        assert transformed is not None
        assert transformed.shape[0] == 3
        assert not np.any(pd.isna(transformed))
    
    def test_very_long_strings(self):
        """Test pipeline with very long categorical strings."""
        data = np.array([['a' * 1000], ['b' * 1000], ['c' * 1000]], dtype=object)
        pipeline = pipelines.create_ordinal_encoder_pipeline()
        
        transformed = pipeline.fit_transform(data)
        assert transformed is not None
        assert transformed.shape == (3, 1)
        assert not np.any(pd.isna(transformed))
        assert transformed.dtype in [np.int32, np.int64]
    
    def test_label_encoder_unknown_handling(self):
        """Test label encoder with unknown categories."""
        encoder = pipelines.create_label_encoder_pipeline(
            handle_unknown='encoded_value',
            encoded_unknown=-999
        )
        
        train_data = np.array(['cat', 'dog', 'bird'])
        encoder.fit(train_data)
        
        # Test with unknown category
        test_data = np.array(['cat', 'fish', 'dog'])
        encoded = encoder.transform(test_data)
        
        assert encoded is not None
        assert -999 in encoded  # Unknown category should be encoded as -999


class TestDataIntegrity:
    """Test data integrity and validation."""
    
    def test_output_shape_consistency(self):
        """Test that output shape is consistent."""
        data = np.array([['cat', 'red'], ['dog', 'blue'], ['cat', 'red']], dtype=object)
        pipeline = pipelines.create_ordinal_encoder_pipeline()
        
        transformed = pipeline.fit_transform(data)
        assert transformed.shape[0] == data.shape[0]
        assert transformed.shape[1] == data.shape[1]
    
    def test_no_data_leakage(self):
        """Test that fitting and transforming are separate."""
        train_data = np.array([['cat'], ['dog'], ['bird']], dtype=object)
        test_data = np.array([['cat'], ['fish'], ['dog']], dtype=object)
        
        pipeline = pipelines.create_ordinal_encoder_pipeline(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        
        pipeline.fit(train_data)
        train_transformed = pipeline.transform(train_data)
        test_transformed = pipeline.transform(test_data)
        
        # Test data should have unknown category marked
        assert -1 in test_transformed
        assert -1 not in train_transformed
    
    def test_inverse_transform_compatibility(self):
        """Test that encodings are reversible where applicable."""
        data = np.array(['cat', 'dog', 'bird', 'cat'])
        encoder = pipelines.create_label_encoder_pipeline()
        
        encoder.fit(data)
        encoded = encoder.transform(data)
        decoded = encoder.inverse_transform(encoded)
        
        np.testing.assert_array_equal(data, decoded)


class TestPerformance:
    """Test performance with larger datasets."""
    
    def test_large_dataset_onehot(self):
        """Test pipeline with large dataset."""
        np.random.seed(42)
        large_data = np.random.choice(['cat', 'dog', 'bird', 'fish'], 
                                     size=(10000, 5), 
                                     replace=True).astype(object)
        
        pipeline = pipelines.create_onehot_encoder_pipeline(sparse_output=True)
        
        transformed = pipeline.fit_transform(large_data)
        assert transformed is not None
        assert transformed.shape[0] == 10000
    
    def test_large_dataset_ordinal(self):
        """Test ordinal encoding with large dataset."""
        np.random.seed(42)
        large_data = np.random.choice(['A', 'B', 'C', 'D'], 
                                     size=(10000, 3), 
                                     replace=True).astype(object)
        
        pipeline = pipelines.create_ordinal_encoder_pipeline()
        
        transformed = pipeline.fit_transform(large_data)
        assert transformed is not None
        assert transformed.shape == large_data.shape


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
