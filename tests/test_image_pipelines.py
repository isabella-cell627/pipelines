"""
Comprehensive tests for image preprocessing pipelines.
"""

import pytest
import numpy as np
from sklearn_preprocessing_pipelines.image import pipelines


class TestImagePipelines:
    """Test suite for image preprocessing pipelines."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Create sample image data (batch of 10 images, 32x32 RGB)
        self.sample_images = np.random.rand(10, 32, 32, 3)
        self.sample_grayscale = np.random.rand(10, 32, 32)
    
    def test_image_resizer(self):
        """Test ImageResizer."""
        resizer = pipelines.ImageResizer(
            target_size=(64, 64),
            interpolation='cubic',
            anti_aliasing=True
        )
        
        resizer.fit(self.sample_images)
        resized = resizer.transform(self.sample_images)
        
        assert resized is not None
        assert resized.shape[1:3] == (64, 64)
    
    def test_image_resizer_preserve_aspect(self):
        """Test ImageResizer with aspect ratio preservation."""
        resizer = pipelines.ImageResizer(
            target_size=(64, 64),
            preserve_aspect_ratio=True
        )
        
        resizer.fit(self.sample_images)
        resized = resizer.transform(self.sample_images)
        
        assert resized is not None
    
    def test_image_normalizer_minmax(self):
        """Test ImageNormalizer with minmax."""
        normalizer = pipelines.ImageNormalizer(
            method='minmax',
            per_channel=True
        )
        
        normalizer.fit(self.sample_images)
        normalized = normalizer.transform(self.sample_images)
        
        assert normalized is not None
        assert np.all(normalized >= 0) and np.all(normalized <= 1)
    
    def test_image_normalizer_standard(self):
        """Test ImageNormalizer with standard scaling."""
        normalizer = pipelines.ImageNormalizer(
            method='standard',
            per_channel=False
        )
        
        normalizer.fit(self.sample_images)
        normalized = normalizer.transform(self.sample_images)
        
        assert normalized is not None
    
    def test_image_normalizer_divide255(self):
        """Test ImageNormalizer with divide by 255."""
        # Create uint8 images
        images_uint8 = (np.random.rand(10, 32, 32, 3) * 255).astype(np.uint8)
        
        normalizer = pipelines.ImageNormalizer(
            method='divide255'
        )
        
        normalizer.fit(images_uint8)
        normalized = normalizer.transform(images_uint8)
        
        assert normalized is not None
        assert np.all(normalized >= 0) and np.all(normalized <= 1)
    
    def test_image_flattener(self):
        """Test ImageFlattener."""
        flattener = pipelines.ImageFlattener(
            strategy='flatten'
        )
        
        flattener.fit(self.sample_images)
        flattened = flattener.transform(self.sample_images)
        
        assert flattened is not None
        assert flattened.ndim == 2
        assert flattened.shape[0] == self.sample_images.shape[0]
    
    def test_image_resizer_interpolations(self):
        """Test different interpolation methods."""
        methods = ['nearest', 'linear', 'cubic']
        
        for method in methods:
            resizer = pipelines.ImageResizer(
                target_size=(48, 48),
                interpolation=method
            )
            
            resized = resizer.fit_transform(self.sample_images)
            assert resized.shape[1:3] == (48, 48)
    
    def test_image_normalizer_methods(self):
        """Test different normalization methods."""
        methods = ['minmax', 'standard', 'robust', 'divide255', 'mean']
        
        for method in methods:
            normalizer = pipelines.ImageNormalizer(method=method)
            normalized = normalizer.fit_transform(self.sample_images)
            assert normalized is not None


class TestImagePreprocessingIntegration:
    """Integration tests for image pipelines."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.images = np.random.rand(5, 64, 64, 3)
    
    def test_full_preprocessing_pipeline(self):
        """Test a complete preprocessing pipeline."""
        from sklearn.pipeline import Pipeline
        
        # Create a complete pipeline
        pipeline = Pipeline([
            ('resizer', pipelines.ImageResizer(target_size=(32, 32))),
            ('normalizer', pipelines.ImageNormalizer(method='minmax')),
            ('flattener', pipelines.ImageFlattener(strategy='flatten'))
        ])
        
        result = pipeline.fit_transform(self.images)
        assert result is not None
        assert result.ndim == 2


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
