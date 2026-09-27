"""Registry of source tables: raw file, silver primary key, and silver grain."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceTable:
    name: str
    raw_file: str
    primary_key: tuple[str, ...]
    description: str


SOURCES: dict[str, SourceTable] = {
    t.name: t
    for t in [
        SourceTable(
            "orders",
            "olist_orders_dataset.csv",
            ("order_id",),
            "One row per order with lifecycle timestamps.",
        ),
        SourceTable(
            "order_items",
            "olist_order_items_dataset.csv",
            ("order_id", "order_item_id"),
            "One row per item within an order.",
        ),
        SourceTable(
            "order_payments",
            "olist_order_payments_dataset.csv",
            ("order_id", "payment_sequential"),
            "One row per payment used for an order.",
        ),
        SourceTable(
            "order_reviews",
            "olist_order_reviews_dataset.csv",
            ("order_id",),
            "Latest customer review per order.",
        ),
        SourceTable(
            "customers",
            "olist_customers_dataset.csv",
            ("customer_id",),
            "One row per order-level customer id (customer_unique_id is the person).",
        ),
        SourceTable(
            "sellers", "olist_sellers_dataset.csv", ("seller_id",), "One row per marketplace seller."
        ),
        SourceTable(
            "products",
            "olist_products_dataset.csv",
            ("product_id",),
            "One row per product with English category name.",
        ),
        SourceTable(
            "geolocation",
            "olist_geolocation_dataset.csv",
            ("zip_code_prefix",),
            "One row per 5-digit zip prefix with centroid coordinates.",
        ),
        SourceTable(
            "category_translation",
            "product_category_name_translation.csv",
            ("product_category_name",),
            "Portuguese to English category names.",
        ),
    ]
}
