"""
Scikit-Learn Preprocessing Pipelines Library

A comprehensive library providing reusable preprocessing pipeline templates
for various data types including numerical, categorical, mixed, text, 
time series, and image data.
"""

from sklearn_preprocessing_pipelines.numerical import pipelines as numerical_pipelines
from sklearn_preprocessing_pipelines.categorical import pipelines as categorical_pipelines
from sklearn_preprocessing_pipelines.mixed import pipelines as mixed_pipelines
from sklearn_preprocessing_pipelines.text import pipelines as text_pipelines
from sklearn_preprocessing_pipelines.timeseries import pipelines as timeseries_pipelines
from sklearn_preprocessing_pipelines.image import pipelines as image_pipelines

__version__ = "1.0.0"
__all__ = [
    "numerical_pipelines",
    "categorical_pipelines", 
    "mixed_pipelines",
    "text_pipelines",
    "timeseries_pipelines",
    "image_pipelines"
]
