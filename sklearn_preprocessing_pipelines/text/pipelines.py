"""
Advanced Text Data Preprocessing Pipelines

Provides industrial-strength pipelines for preprocessing text data with
enhanced vectorization, comprehensive text cleaning, and advanced features.
Includes configurable components for production-ready text processing.
"""

import re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from scipy import sparse
import logging
from typing import Union, List, Optional, Callable
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedTextCleaner(BaseEstimator, TransformerMixin):
    """
    Advanced text cleaning transformer with comprehensive preprocessing capabilities.
    
    Features:
    - Configurable text normalization
    - Advanced pattern matching and replacement
    - Unicode handling
    - Custom preprocessing functions
    - Batch processing optimization
    """
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_special_chars: bool = True,
                 remove_numbers: bool = False,
                 remove_extra_whitespace: bool = True,
                 preserve_case_for_entities: bool = False,
                 custom_patterns: Optional[List[tuple]] = None,
                 strip_accents: bool = True,
                 min_token_length: int = 1):
        """
        Initialize the advanced text cleaner.
        
        Parameters:
        -----------
        lowercase : bool, default=True
            Convert text to lowercase
        remove_special_chars : bool, default=True
            Remove special characters and punctuation
        remove_numbers : bool, default=False
            Remove numerical digits
        remove_extra_whitespace : bool, default=True
            Normalize whitespace
        preserve_case_for_entities : bool, default=False
            Preserve case for potential named entities
        custom_patterns : list of tuples, optional
            Custom regex patterns and replacements [(pattern, replacement), ...]
        strip_accents : bool, default=True
            Remove accent characters
        min_token_length : int, default=1
            Minimum token length after cleaning
        """
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.remove_numbers = remove_numbers
        self.remove_extra_whitespace = remove_extra_whitespace
        self.preserve_case_for_entities = preserve_case_for_entities
        self.custom_patterns = custom_patterns or []
        self.strip_accents = strip_accents
        self.min_token_length = min_token_length
        
        # Pre-compile regex patterns for performance
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficient processing."""
        self.patterns = []
        
        if self.remove_special_chars:
            # Enhanced special character removal preserving basic word structure
            self.patterns.append((re.compile(r'[^\w\s@#]|_'), ' '))
        
        if self.remove_numbers:
            self.patterns.append((re.compile(r'\d+'), ' '))
            
        if self.remove_extra_whitespace:
            self.patterns.append((re.compile(r'\s+'), ' '))
            
        # Add custom patterns
        for pattern, replacement in self.custom_patterns:
            self.patterns.append((re.compile(pattern), replacement))
            
        # Accent stripping pattern (basic implementation)
        if self.strip_accents:
            self.patterns.append((re.compile(r'[̀-ͯ]'), ''))
    
    def _clean_single_text(self, text: str) -> str:
        """Clean a single text string with all configured rules."""
        if not isinstance(text, str) or not text.strip():
            return ""
            
        cleaned_text = text
        
        # Apply regex patterns
        for pattern, replacement in self.patterns:
            cleaned_text = pattern.sub(replacement, cleaned_text)
            
        # Case handling with entity preservation option
        if self.lowercase and not self.preserve_case_for_entities:
            cleaned_text = cleaned_text.lower()
            
        # Remove short tokens if specified
        if self.min_token_length > 1:
            tokens = cleaned_text.split()
            tokens = [token for token in tokens if len(token) >= self.min_token_length]
            cleaned_text = ' '.join(tokens)
            
        return cleaned_text.strip()
    
    def fit(self, X, y=None):
        """Fit transformer (no operation for cleaner)."""
        logger.info("Fitting AdvancedTextCleaner...")
        return self
    
    def transform(self, X):
        """Transform text data with advanced cleaning."""
        logger.info("Transforming text data with AdvancedTextCleaner...")
        
        if not isinstance(X, (list, np.ndarray)):
            raise ValueError("Input must be list or numpy array")
            
        X_cleaned = []
        total_texts = len(X)
        
        for i, text in enumerate(X):
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{total_texts} texts...")
                
            cleaned_text = self._clean_single_text(text)
            X_cleaned.append(cleaned_text)
            
        logger.info(f"Successfully cleaned {total_texts} texts")
        return np.array(X_cleaned)


class TextNormalizer(BaseEstimator, TransformerMixin):
    """
    Advanced text normalizer for consistent text representation.
    
    Features:
    - Advanced Unicode normalization
    - Character encoding normalization
    - Emoji and symbol handling
    - URL and email replacement
    """
    
    def __init__(self, 
                 replace_urls: bool = True,
                 replace_emails: bool = True,
                 replace_phone_numbers: bool = True,
                 normalize_unicode: bool = True):
        """
        Initialize text normalizer.
        
        Parameters:
        -----------
        replace_urls : bool, default=True
            Replace URLs with placeholder
        replace_emails : bool, default=True
            Replace email addresses with placeholder
        replace_phone_numbers : bool, default=True
            Replace phone numbers with placeholder
        normalize_unicode : bool, default=True
            Normalize Unicode characters
        """
        self.replace_urls = replace_urls
        self.replace_emails = replace_emails
        self.replace_phone_numbers = replace_phone_numbers
        self.normalize_unicode = normalize_unicode
        
        # Pre-compile patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?(\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})')
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Normalize text data."""
        X_normalized = []
        
        for text in X:
            if not isinstance(text, str):
                X_normalized.append("")
                continue
                
            normalized_text = text
            
            if self.replace_urls:
                normalized_text = self.url_pattern.sub(' URL ', normalized_text)
            if self.replace_emails:
                normalized_text = self.email_pattern.sub(' EMAIL ', normalized_text)
            if self.replace_phone_numbers:
                normalized_text = self.phone_pattern.sub(' PHONE ', normalized_text)
            if self.normalize_unicode:
                # Basic Unicode normalization - in practice, you might use unicodedata
                normalized_text = normalized_text.encode('ascii', 'ignore').decode('ascii')
                
            X_normalized.append(normalized_text)
            
        return np.array(X_normalized)


def create_advanced_count_vectorizer_pipeline(
    max_features: int = 5000, 
    ngram_range: tuple = (1, 2),
    min_df: Union[int, float] = 2,
    max_df: Union[int, float] = 0.95,
    binary: bool = False,
    dtype: type = np.float64,
    strip_accents: bool = True,
    remove_extra_whitespace: bool = True
) -> Pipeline:
    """
    Creates an advanced count vectorization pipeline with comprehensive preprocessing.
    
    Parameters:
    -----------
    max_features : int, default=5000
        Maximum number of features (vocabulary size)
    ngram_range : tuple, default=(1, 2)
        The lower and upper boundary of n-grams range
    min_df : int or float, default=2
        Minimum document frequency (int for count, float for proportion)
    max_df : int or float, default=0.95
        Maximum document frequency
    binary : bool, default=False
        If True, all non-zero counts are set to 1
    dtype : type, default=np.float64
        Type of the matrix returned by vectorizer
    strip_accents : bool, default=True
        Remove accents during preprocessing
    remove_extra_whitespace : bool, default=True
        Normalize whitespace
    
    Returns:
    --------
    Pipeline : sklearn.pipeline.Pipeline
        Advanced count vectorization pipeline
    """
    return Pipeline([
        ('normalizer', TextNormalizer()),
        ('cleaner', AdvancedTextCleaner(
            strip_accents=strip_accents,
            remove_extra_whitespace=remove_extra_whitespace,
            min_token_length=2
        )),
        ('vectorizer', CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            binary=binary,
            dtype=dtype,
            stop_words='english',
            token_pattern=r'(?u)\b\w\w+\b'  # At least 2 characters
        ))
    ])


def create_advanced_tfidf_pipeline(
    max_features: int = 10000,
    ngram_range: tuple = (1, 3),
    min_df: Union[int, float] = 2,
    max_df: Union[int, float] = 0.9,
    norm: str = 'l2',
    use_idf: bool = True,
    smooth_idf: bool = True,
    sublinear_tf: bool = True,
    dtype: type = np.float64,
    strip_accents: bool = True,
    lowercase: bool = True
) -> Pipeline:
    """
    Creates a production-ready TF-IDF pipeline with advanced features.
    
    Parameters:
    -----------
    max_features : int, default=10000
        Maximum number of features
    ngram_range : tuple, default=(1, 3)
        N-gram range (unigrams to trigrams)
    min_df : int or float, default=2
        Minimum document frequency
    max_df : int or float, default=0.9
        Maximum document frequency
    norm : {'l1', 'l2'}, default='l2'
        Norm used to normalize term vectors
    use_idf : bool, default=True
        Enable inverse-document-frequency reweighting
    smooth_idf : bool, default=True
        Smooth idf weights by adding one to document frequencies
    sublinear_tf : bool, default=True
        Apply sublinear tf scaling (1 + log(tf))
    dtype : type, default=np.float64
        Type of the matrix returned by vectorizer
    strip_accents : bool, default=True
        Remove accents during preprocessing
    lowercase : bool, default=True
        Convert all characters to lowercase
    
    Returns:
    --------
    Pipeline : sklearn.pipeline.Pipeline
        Advanced TF-IDF pipeline
    """
    return Pipeline([
        ('normalizer', TextNormalizer()),
        ('cleaner', AdvancedTextCleaner(
            lowercase=lowercase,
            strip_accents=strip_accents,
            remove_extra_whitespace=True,
            min_token_length=2
        )),
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
            dtype=dtype,
            stop_words='english',
            token_pattern=r'(?u)\b\w\w+\b',  # At least 2 characters
            analyzer='word'
        ))
    ])


def create_robust_text_pipeline(
    vectorizer_type: str = 'tfidf',
    max_features: int = 8000,
    ngram_range: tuple = (1, 2),
    advanced_cleaning: bool = True,
    **kwargs
) -> Pipeline:
    """
    Creates a robust text preprocessing pipeline with configurable vectorization.
    
    Parameters:
    -----------
    vectorizer_type : {'tfidf', 'count'}, default='tfidf'
        Type of vectorizer to use
    max_features : int, default=8000
        Maximum number of features
    ngram_range : tuple, default=(1, 2)
        N-gram range
    advanced_cleaning : bool, default=True
        Use advanced text cleaning
    **kwargs : dict
        Additional parameters passed to vectorizer
    
    Returns:
    --------
    Pipeline : sklearn.pipeline.Pipeline
        Robust text preprocessing pipeline
    """
    steps = []
    
    # Add normalizer
    steps.append(('normalizer', TextNormalizer()))
    
    # Add cleaner
    if advanced_cleaning:
        steps.append(('cleaner', AdvancedTextCleaner(
            min_token_length=2,
            remove_extra_whitespace=True,
            strip_accents=True
        )))
    
    # Add vectorizer
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            **kwargs
        )
    elif vectorizer_type == 'count':
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.95,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")
    
    steps.append(('vectorizer', vectorizer))
    
    return Pipeline(steps)


def create_production_text_pipeline(
    max_features: int = 15000,
    ngram_range: tuple = (1, 3),
    min_df: int = 3,
    max_df: float = 0.85,
    norm: str = 'l2',
    sublinear_tf: bool = True
) -> Pipeline:
    """
    Creates a production-grade text preprocessing pipeline optimized for performance.
    
    Parameters:
    -----------
    max_features : int, default=15000
        Maximum number of features
    ngram_range : tuple, default=(1, 3)
        N-gram range
    min_df : int, default=3
        Minimum document frequency
    max_df : float, default=0.85
        Maximum document frequency
    norm : str, default='l2'
        Normalization method
    sublinear_tf : bool, default=True
        Use sublinear term frequency scaling
    
    Returns:
    --------
    Pipeline : sklearn.pipeline.Pipeline
        Production-ready text preprocessing pipeline
    """
    return Pipeline([
        ('normalizer', TextNormalizer(
            replace_urls=True,
            replace_emails=True,
            replace_phone_numbers=True
        )),
        ('cleaner', AdvancedTextCleaner(
            lowercase=True,
            remove_special_chars=True,
            remove_extra_whitespace=True,
            strip_accents=True,
            min_token_length=2,
            preserve_case_for_entities=False
        )),
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            norm=norm,
            sublinear_tf=sublinear_tf,
            use_idf=True,
            smooth_idf=True,
            stop_words='english',
            token_pattern=r'(?u)\b\w{2,}\b',  # At least 2 characters
            dtype=np.float32,  # Use float32 for memory efficiency
            analyzer='word'
        ))
    ])


# Example usage and testing
if __name__ == "__main__":
    # Sample text data for testing
    sample_texts = [
        "Hello WORLD!!! This is a test text with numbers 123 and symbols @#%.",
        "Another example with URL: https://example.com and email test@email.com",
        "Simple text for processing",
        "Text with      multiple   spaces     and 123 numbers!",
        "Special characters: café, naïve, résumé"
    ]
    
    # Test advanced pipelines
    print("Testing Advanced Text Processing Pipelines...")
    
    # Test production pipeline
    production_pipeline = create_production_text_pipeline(max_features=1000)
    transformed = production_pipeline.fit_transform(sample_texts)
    print(f"Production pipeline output shape: {transformed.shape}")
    print(f"Feature names sample: {production_pipeline.named_steps['tfidf'].get_feature_names_out()[:10]}")
    
    # Test robust pipeline with count vectorizer
    count_pipeline = create_robust_text_pipeline(vectorizer_type='count', max_features=500)
    count_transformed = count_pipeline.fit_transform(sample_texts)
    print(f"Count vectorizer output shape: {count_transformed.shape}")
    
    print("All pipelines tested successfully!")