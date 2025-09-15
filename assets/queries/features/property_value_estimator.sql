SELECT
  -- House base
  hs.bedrooms,
  hs.bathrooms,
  hs.sqft_living,
  hs.sqft_lot,
  hs.floors,
  hs.sqft_basement,

  -- House quality and neighborhood
  hs.grade,
  hs.view,
  hs.waterfront,
  hs.sqft_living15,
  hs.condition,

  -- Demographics
  zd.medn_hshld_incm_amt,
  zd.medn_incm_per_prsn_amt,
  zd.hous_val_amt,
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

  -- Target last
  hs.price
FROM raw_house_sales hs
LEFT JOIN raw_zipcode_demographics zd
  ON hs.zipcode = zd.zipcode;



-- Note: zipcode excluded from SELECT (dropped after merge in Python)
-- Note: Using raw tables preserves time-series data integrity
