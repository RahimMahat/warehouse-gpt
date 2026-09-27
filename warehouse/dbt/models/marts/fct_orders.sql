-- Grain: one row per order.
with items as (
    select
        order_id,
        count(*)                   as item_count,
        count(distinct seller_id)  as seller_count,
        count(distinct product_id) as product_count,
        sum(price)                 as items_value,
        sum(freight_value)         as freight_value
    from {{ ref('stg_order_items') }}
    group by order_id
),

payments as (
    select
        order_id,
        sum(payment_value)                                   as payment_value,
        max(payment_installments)                            as max_installments,
        count(*)                                             as payment_count,
        arg_max(payment_type, payment_value)                 as primary_payment_type
    from {{ ref('stg_order_payments') }}
    group by order_id
)

select
    o.order_id,
    c.customer_unique_id,
    o.customer_id,
    c.state                                                     as customer_state,
    c.city                                                      as customer_city,
    o.order_status,
    o.purchased_at,
    cast(o.purchased_at as date)                                as purchase_date,
    o.approved_at,
    o.delivered_to_carrier_at,
    o.delivered_to_customer_at,
    o.estimated_delivery_at,
    o.order_status = 'delivered' and o.delivered_to_customer_at is not null
                                                                as is_delivered,
    o.order_status in ('canceled', 'unavailable')               as is_canceled,
    date_diff('day', o.purchased_at, o.delivered_to_customer_at)
                                                                as delivery_days,
    case
        when o.delivered_to_customer_at is null then null
        else o.delivered_to_customer_at > o.estimated_delivery_at + interval 1 day
    end                                                         as is_late,
    coalesce(i.item_count, 0)                                   as item_count,
    coalesce(i.seller_count, 0)                                 as seller_count,
    coalesce(i.product_count, 0)                                as product_count,
    coalesce(i.items_value, 0)                                  as items_value,
    coalesce(i.freight_value, 0)                                as freight_value,
    coalesce(i.items_value, 0) + coalesce(i.freight_value, 0)   as order_value,
    p.payment_value,
    p.payment_count,
    p.primary_payment_type,
    p.max_installments,
    r.review_score
from {{ ref('stg_orders') }} o
left join {{ ref('stg_customers') }} c using (customer_id)
left join items i using (order_id)
left join payments p using (order_id)
left join {{ ref('stg_order_reviews') }} r using (order_id)
