"""
Module for configuration management
"""

TARGET_COLUMN = "price"
FEATURE_COLUMNS = [
    # base
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_basement",

    # quality and neighborhood
    "grade",
    "view",
    "waterfront",
    "sqft_living15",
    "condition",

    # demographics
    "medn_hshld_incm_amt",
    "medn_incm_per_prsn_amt",
    "hous_val_amt",
    "per_urbn",
    "per_sbrbn",
    "per_farm",
    "per_non_farm",
    "per_less_than_9",
    "per_9_to_12",
    "per_hsd",
    "per_some_clg",
    "per_assoc",
    "per_bchlr",
    "per_prfsnl",
]


COLUMN_SELECTION = FEATURE_COLUMNS + [TARGET_COLUMN] + ['zipcode']