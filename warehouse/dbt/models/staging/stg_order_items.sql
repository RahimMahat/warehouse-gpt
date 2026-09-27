select * from {{ source('silver', 'order_items') }}
