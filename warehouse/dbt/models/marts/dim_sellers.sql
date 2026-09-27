-- Grain: one row per seller.
select
    s.seller_id,
    s.zip_code_prefix,
    s.city,
    s.state,
    g.lat,
    g.lng
from {{ ref('stg_sellers') }} s
left join {{ ref('stg_geolocation') }} g using (zip_code_prefix)
