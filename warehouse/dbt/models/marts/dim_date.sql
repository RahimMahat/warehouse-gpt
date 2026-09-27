-- Grain: one row per calendar day covering the dataset (2016-2018).
select
    cast(d as date)                     as date_day,
    year(d)                             as year,
    quarter(d)                          as quarter,
    month(d)                            as month,
    strftime(d, '%Y-%m')                as year_month,
    monthname(d)                        as month_name,
    week(d)                             as iso_week,
    isodow(d)                           as iso_day_of_week,
    dayname(d)                          as day_name,
    isodow(d) in (6, 7)                 as is_weekend
from range(date '2016-01-01', date '2019-01-01', interval 1 day) t(d)
