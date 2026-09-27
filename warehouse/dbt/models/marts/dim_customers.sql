-- Grain: one row per real customer (customer_unique_id).
-- Olist issues a new customer_id per order, so counting customer_id overcounts people.
with orders as (
    select * from {{ ref('fct_orders') }}
),

latest_location as (
    select
        customer_unique_id,
        arg_max(customer_state, purchased_at) as state,
        arg_max(customer_city, purchased_at)  as city
    from orders
    group by customer_unique_id
)

select
    o.customer_unique_id,
    l.state,
    l.city,
    min(o.purchased_at)                                          as first_order_at,
    max(o.purchased_at)                                          as last_order_at,
    count(*)                                                     as order_count,
    count(*) filter (where not o.is_canceled)                    as valid_order_count,
    coalesce(sum(o.order_value) filter (where not o.is_canceled), 0)
                                                                 as lifetime_value,
    count(*) > 1                                                 as is_repeat_customer,
    avg(o.review_score)                                          as avg_review_score
from orders o
join latest_location l using (customer_unique_id)
group by all
