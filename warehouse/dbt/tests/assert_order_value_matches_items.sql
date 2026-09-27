-- Order-level items_value must equal the sum of item prices in fct_order_items.
select o.order_id
from {{ ref('fct_orders') }} o
left join (
    select order_id, sum(price) as items_value
    from {{ ref('fct_order_items') }}
    group by order_id
) i using (order_id)
where coalesce(i.items_value, 0) <> o.items_value
