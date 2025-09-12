"""
Module for configuration management
"""

TARGET_COLUMN = "price"
FEATURE_COLUMNS = [
    'bedrooms',
    'bathrooms',
    'sqft_living',
    'sqft_lot',
    'floors',
    'sqft_above',
    'sqft_basement'
]

COLUMN_SELECTION = FEATURE_COLUMNS + [TARGET_COLUMN] + ['zipcode']