-- Paid amount should match billed amount (items + freight) for delivered orders.
-- Installment interest and vouchers cause small legitimate gaps, so this only warns,
-- and only when more than ~1% of orders disagree by more than 1% / 1 BRL.
{{ config(severity='warn', warn_if='>1000') }}
select order_id, order_value, payment_value
from {{ ref('fct_orders') }}
where is_delivered
  and payment_value is not null
  and abs(payment_value - order_value) > greatest(1, 0.01 * order_value)
