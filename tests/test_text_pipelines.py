"""
Comprehensive tests for text preprocessing pipelines.
"""

import pytest
import numpy as np
from sklearn_preprocessing_pipelines.text import pipelines


class TestTextPipelines:
    """Test suite for text preprocessing pipelines."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_texts = [
            "Hello WORLD!!! This is a test text with numbers 123.",
            "Another example with URL: https://example.com and email test@email.com",
            "Simple text for processing",
            "Text with      multiple   spaces     and 456 numbers!",
            "Special characters: café, naïve, résumé"
        ]
    
    def test_count_vectorizer_pipeline(self):
        """Test count vectorizer pipeline."""
        pipeline = pipelines.create_advanced_count_vectorizer_pipeline(
            max_features=100,
            ngram_range=(1, 2),
            min_df=1
        )
        
        assert pipeline is not None
        assert len(pipeline.steps) == 3
        
        # Test fit_transform
        transformed = pipeline.fit_transform(self.sample_texts)
        assert transformed is not None
        assert transformed.shape[0] == len(self.sample_texts)
    
    def test_tfidf_pipeline(self):
        """Test TF-IDF pipeline."""
        pipeline = pipelines.create_advanced_tfidf_pipeline(
            max_features=100,
            ngram_range=(1, 2),
            min_df=1
        )
        
        assert pipeline is not None
        transformed = pipeline.fit_transform(self.sample_texts)
        assert transformed is not None
        assert transformed.shape[0] == len(self.sample_texts)
    
    def test_robust_text_pipeline(self):
        """Test robust text pipeline."""
        pipeline = pipelines.create_robust_text_pipeline(
            vectorizer_type='tfidf',
            max_features=100,
            ngram_range=(1, 2),
            advanced_cleaning=True
        )
        
        assert pipeline is not None
        transformed = pipeline.fit_transform(self.sample_texts)
        assert transformed is not None
    
    def test_production_text_pipeline(self):
        """Test production text pipeline."""
        pipeline = pipelines.create_production_text_pipeline(
            max_features=100,
            ngram_range=(1, 2),
            min_df=1
        )
        
        assert pipeline is not None
        transformed = pipeline.fit_transform(self.sample_texts)
        assert transformed is not None


class TestAdvancedTextCleaner:
    """Test AdvancedTextCleaner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_texts = [
            "Hello WORLD!!!",
            "Test with 123 numbers",
            "Special @#$ characters"
        ]
    
    def test_lowercase(self):
        """Test lowercase conversion."""
        cleaner = pipelines.AdvancedTextCleaner(lowercase=True)
        cleaner.fit(self.sample_texts)
        cleaned = cleaner.transform(self.sample_texts)
        
        assert all(isinstance(text, str) for text in cleaned)
        assert any('hello' in text.lower() for text in cleaned)
    
    def test_remove_special_chars(self):
        """Test special character removal."""
        cleaner = pipelines.AdvancedTextCleaner(
            remove_special_chars=True,
            remove_numbers=False
        )
        cleaner.fit(self.sample_texts)
        cleaned = cleaner.transform(self.sample_texts)
        
        assert all(isinstance(text, str) for text in cleaned)
    
    def test_remove_numbers(self):
        """Test number removal."""
        cleaner = pipelines.AdvancedTextCleaner(
            remove_numbers=True
        )
        cleaner.fit(self.sample_texts)
        cleaned = cleaner.transform(self.sample_texts)
        
        assert all(isinstance(text, str) for text in cleaned)


class TestTextNormalizer:
    """Test TextNormalizer class."""
    
    def test_url_replacement(self):
        """Test URL replacement."""
        normalizer = pipelines.TextNormalizer(replace_urls=True)
        texts = ["Check out https://example.com for more info"]
        
        normalized = normalizer.fit_transform(texts)
        assert 'URL' in normalized[0]
    
    def test_email_replacement(self):
        """Test email replacement."""
        normalizer = pipelines.TextNormalizer(replace_emails=True)
        texts = ["Contact us at test@example.com"]
        
        normalized = normalizer.fit_transform(texts)
        assert 'EMAIL' in normalized[0]


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
