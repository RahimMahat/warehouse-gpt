select * from {{ source('silver', 'order_payments') }}
