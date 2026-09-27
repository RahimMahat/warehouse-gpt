-- Grain: one row per payment (an order can be paid with several methods / vouchers).
select
    {{ dbt.concat(["op.order_id", "'-'", "cast(op.payment_sequential as varchar)"]) }} as payment_key,
    op.order_id,
    op.payment_sequential,
    op.payment_type,
    op.payment_installments,
    op.payment_value,
    o.customer_unique_id,
    o.customer_state,
    o.order_status,
    o.purchased_at,
    o.purchase_date
from {{ ref('stg_order_payments') }} op
join {{ ref('fct_orders') }} o using (order_id)
