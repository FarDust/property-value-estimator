-- Property Value Estimator Feature Query
-- Replicates the logic from mle-project-challenge-2/create_model.py load_data function

SELECT 
    -- House characteristics features (from SALES_COLUMN_SELECTION)
    hs.bedrooms,
    hs.bathrooms,
    hs.sqft_living,
    hs.sqft_lot,
    hs.floors,
    hs.sqft_above,
    hs.sqft_basement,
    
    -- Demographics features
    zd.ppltn_qty,
    zd.urbn_ppltn_qty,
    zd.sbrbn_ppltn_qty,
    zd.farm_ppltn_qty,
    zd.non_farm_qty,
    zd.medn_hshld_incm_amt,
    zd.medn_incm_per_prsn_amt,
    zd.hous_val_amt,
    zd.edctn_less_than_9_qty,
    zd.edctn_9_12_qty,
    zd.edctn_high_schl_qty,
    zd.edctn_some_clg_qty,
    zd.edctn_assoc_dgre_qty,
    zd.edctn_bchlr_dgre_qty,
    zd.edctn_prfsnl_qty,
    zd.per_urbn,
    zd.per_sbrbn,
    zd.per_farm,
    zd.per_non_farm,
    zd.per_less_than_9,
    zd.per_9_to_12,
    zd.per_hsd,
    zd.per_some_clg,
    zd.per_assoc,
    zd.per_bchlr,
    zd.per_prfsnl,
    
    -- Target variable (last column following ML conventions)
    hs.price

FROM raw_house_sales hs
LEFT JOIN raw_zipcode_demographics zd 
    ON hs.zipcode = zd.zipcode;

-- Note: zipcode excluded from SELECT (dropped after merge in Python)
-- Note: Using raw tables preserves time-series data integrity
