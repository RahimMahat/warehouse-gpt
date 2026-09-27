-- Grain: one row per item within an order.
select
    {{ dbt.concat(["oi.order_id", "'-'", "cast(oi.order_item_id as varchar)"]) }} as order_item_key,
    oi.order_id,
    oi.order_item_id,
    oi.product_id,
    oi.seller_id,
    o.customer_unique_id,
    o.customer_state,
    s.state                         as seller_state,
    p.category_name,
    o.order_status,
    o.is_canceled,
    o.purchased_at,
    o.purchase_date,
    oi.shipping_limit_at,
    oi.price,
    oi.freight_value,
    oi.price + oi.freight_value     as item_total_value,
    o.review_score
from {{ ref('stg_order_items') }} oi
join {{ ref('fct_orders') }} o using (order_id)
left join {{ ref('stg_products') }} p using (product_id)
left join {{ ref('stg_sellers') }} s using (seller_id)
