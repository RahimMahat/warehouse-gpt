"""Silver layer: typed, deduplicated, normalized Delta tables + DQ gates."""

from __future__ import annotations

from collections.abc import Callable

import structlog
from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from warehouse_gpt.config import Settings, get_settings
from warehouse_gpt.pipelines import dq
from warehouse_gpt.pipelines.spark_utils import literal_df
from warehouse_gpt.pipelines.tables import SOURCES

log = structlog.get_logger(__name__)

ORDER_STATUSES = [
    "created",
    "approved",
    "invoiced",
    "processing",
    "shipped",
    "delivered",
    "canceled",
    "unavailable",
]
PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card", "not_defined"]
BR_STATES = [
    "AC",
    "AL",
    "AM",
    "AP",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MG",
    "MS",
    "MT",
    "PA",
    "PB",
    "PE",
    "PI",
    "PR",
    "RJ",
    "RN",
    "RO",
    "RR",
    "RS",
    "SC",
    "SE",
    "SP",
    "TO",
]
# Two categories are missing from the official translation file.
MISSING_TRANSLATIONS = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_food_processors",
}

_ACCENTED = "áàâãäéèêëíìîïóòôõöúùûüçñÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇÑ"
_PLAIN = "aaaaaeeeeiiiiooooouuuucnAAAAAEEEEIIIIOOOOOUUUUCN"

Bronze = Callable[[str], DataFrame]


def normalize_text(col: str) -> Column:
    """Lowercase, strip accents and collapse whitespace: 'São  Paulo' -> 'sao paulo'."""
    stripped = F.translate(F.col(col), _ACCENTED, _PLAIN)
    return F.regexp_replace(F.lower(F.trim(stripped)), r"\s+", " ")


def zip_prefix(col: str) -> Column:
    return F.lpad(F.trim(F.col(col)), 5, "0")


def ts(col: str) -> Column:
    # Source timestamps are naive Brazil local time; keep them timezone-free (NTZ) so
    # downstream engines never shift them by the session time zone.
    return F.to_timestamp_ntz(F.col(col), F.lit("yyyy-MM-dd HH:mm:ss"))


def _orders(b: Bronze) -> DataFrame:
    return (
        b("orders")
        .select(
            "order_id",
            "customer_id",
            F.lower(F.trim("order_status")).alias("order_status"),
            ts("order_purchase_timestamp").alias("purchased_at"),
            ts("order_approved_at").alias("approved_at"),
            ts("order_delivered_carrier_date").alias("delivered_to_carrier_at"),
            ts("order_delivered_customer_date").alias("delivered_to_customer_at"),
            ts("order_estimated_delivery_date").alias("estimated_delivery_at"),
        )
        .dropDuplicates(["order_id"])
    )


def _order_items(b: Bronze) -> DataFrame:
    return (
        b("order_items")
        .select(
            "order_id",
            F.col("order_item_id").cast("int").alias("order_item_id"),
            "product_id",
            "seller_id",
            ts("shipping_limit_date").alias("shipping_limit_at"),
            F.col("price").cast("decimal(12,2)").alias("price"),
            F.col("freight_value").cast("decimal(12,2)").alias("freight_value"),
        )
        .dropDuplicates(["order_id", "order_item_id"])
    )


def _order_payments(b: Bronze) -> DataFrame:
    return (
        b("order_payments")
        .select(
            "order_id",
            F.col("payment_sequential").cast("int").alias("payment_sequential"),
            F.lower(F.trim("payment_type")).alias("payment_type"),
            F.col("payment_installments").cast("int").alias("payment_installments"),
            F.col("payment_value").cast("decimal(12,2)").alias("payment_value"),
        )
        .dropDuplicates(["order_id", "payment_sequential"])
    )


def _order_reviews(b: Bronze) -> DataFrame:
    # The raw file has duplicate review_ids and multiple reviews per order.
    # Silver grain = one review per order: the most recently answered one.
    df = b("order_reviews").select(
        "review_id",
        "order_id",
        F.col("review_score").cast("int").alias("review_score"),
        F.nullif(F.trim("review_comment_title"), F.lit("")).alias("review_title"),
        F.nullif(F.trim("review_comment_message"), F.lit("")).alias("review_message"),
        ts("review_creation_date").alias("review_created_at"),
        ts("review_answer_timestamp").alias("review_answered_at"),
    )
    w = Window.partitionBy("order_id").orderBy(
        F.col("review_answered_at").desc_nulls_last(), F.col("review_id")
    )
    return df.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")


def _customers(b: Bronze) -> DataFrame:
    return (
        b("customers")
        .select(
            "customer_id",
            "customer_unique_id",
            zip_prefix("customer_zip_code_prefix").alias("zip_code_prefix"),
            normalize_text("customer_city").alias("city"),
            F.upper(F.trim("customer_state")).alias("state"),
        )
        .dropDuplicates(["customer_id"])
    )


def _sellers(b: Bronze) -> DataFrame:
    return (
        b("sellers")
        .select(
            "seller_id",
            zip_prefix("seller_zip_code_prefix").alias("zip_code_prefix"),
            normalize_text("seller_city").alias("city"),
            F.upper(F.trim("seller_state")).alias("state"),
        )
        .dropDuplicates(["seller_id"])
    )


def _category_translation(b: Bronze) -> DataFrame:
    base = b("category_translation")
    extra = literal_df(
        base.sparkSession,
        list(MISSING_TRANSLATIONS.items()),
        "product_category_name string, product_category_name_english string",
    )
    return (
        base.select("product_category_name", "product_category_name_english")
        .unionByName(extra)
        .dropDuplicates(["product_category_name"])
    )


def _products(b: Bronze, translation: DataFrame) -> DataFrame:
    p = b("products").select(
        "product_id",
        F.col("product_category_name").alias("category_name_pt"),
        # The source misspells "length" as "lenght"; fixed here once.
        F.col("product_name_lenght").cast("int").alias("product_name_length"),
        F.col("product_description_lenght").cast("int").alias("product_description_length"),
        F.col("product_photos_qty").cast("int").alias("product_photos_qty"),
        F.col("product_weight_g").cast("int").alias("product_weight_g"),
        F.col("product_length_cm").cast("int").alias("product_length_cm"),
        F.col("product_height_cm").cast("int").alias("product_height_cm"),
        F.col("product_width_cm").cast("int").alias("product_width_cm"),
    )
    t = translation.withColumnRenamed("product_category_name", "category_name_pt")
    return (
        p.join(F.broadcast(t), "category_name_pt", "left")
        .withColumn("category_name", F.coalesce(F.col("product_category_name_english"), F.lit("unknown")))
        .withColumn("category_name_pt", F.coalesce(F.col("category_name_pt"), F.lit("unknown")))
        .drop("product_category_name_english")
        .dropDuplicates(["product_id"])
    )


def _geolocation(b: Bronze) -> DataFrame:
    # ~1M raw rows with many points per zip prefix and some coordinates outside Brazil.
    # Collapse to one centroid per prefix, using only points inside Brazil's bounding box.
    g = (
        b("geolocation")
        .select(
            zip_prefix("geolocation_zip_code_prefix").alias("zip_code_prefix"),
            F.col("geolocation_lat").cast("double").alias("lat"),
            F.col("geolocation_lng").cast("double").alias("lng"),
            normalize_text("geolocation_city").alias("city"),
            F.upper(F.trim("geolocation_state")).alias("state"),
        )
        .filter("lat BETWEEN -34 AND 5.5 AND lng BETWEEN -74 AND -34")
    )
    mode_w = Window.partitionBy("zip_code_prefix").orderBy(F.desc("n"), "city", "state")
    modal_place = (
        g.groupBy("zip_code_prefix", "city", "state")
        .agg(F.count("*").alias("n"))
        .withColumn("_rn", F.row_number().over(mode_w))
        .filter("_rn = 1")
        .select("zip_code_prefix", "city", "state")
    )
    centroids = g.groupBy("zip_code_prefix").agg(
        F.round(F.avg("lat"), 6).alias("lat"),
        F.round(F.avg("lng"), 6).alias("lng"),
        F.count("*").alias("source_points"),
    )
    return centroids.join(modal_place, "zip_code_prefix")


def _checks() -> dict[str, list[dq.Check]]:
    return {
        "orders": [
            dq.min_rows(90_000),
            dq.not_null("order_id", "customer_id", "purchased_at"),
            dq.unique("order_id"),
            dq.accepted_values("order_status", ORDER_STATUSES),
            dq.expression(
                "delivered_orders_have_delivery_date",
                "order_status = 'delivered' AND delivered_to_customer_at IS NULL",
            ),
            dq.expression("delivery_not_before_purchase", "delivered_to_customer_at < purchased_at"),
        ],
        "order_items": [
            dq.not_null("order_id", "product_id", "seller_id", "price"),
            dq.unique("order_id", "order_item_id"),
            dq.in_range("price", lo=0),
            dq.in_range("freight_value", lo=0),
        ],
        "order_payments": [
            dq.not_null("order_id", "payment_type", "payment_value"),
            dq.unique("order_id", "payment_sequential"),
            dq.accepted_values("payment_type", PAYMENT_TYPES),
            dq.in_range("payment_value", lo=0),
            dq.expression("payment_type_defined", "payment_type = 'not_defined'"),
            dq.expression("no_zero_value_payments", "payment_value = 0"),
        ],
        "order_reviews": [
            dq.not_null("order_id", "review_score"),
            dq.unique("order_id"),
            dq.in_range("review_score", lo=1, hi=5),
        ],
        "customers": [
            dq.not_null("customer_id", "customer_unique_id"),
            dq.unique("customer_id"),
            dq.accepted_values("state", BR_STATES),
        ],
        "sellers": [
            dq.not_null("seller_id"),
            dq.unique("seller_id"),
            dq.accepted_values("state", BR_STATES),
        ],
        "products": [
            dq.not_null("product_id", "category_name"),
            dq.unique("product_id"),
            dq.expression("products_have_known_category", "category_name = 'unknown'"),
        ],
        "geolocation": [dq.unique("zip_code_prefix"), dq.not_null("lat", "lng")],
        "category_translation": [dq.unique("product_category_name")],
    }


def build_silver(
    spark: SparkSession, settings: Settings | None = None, fail_on_error: bool = True
) -> dict[str, int]:
    settings = settings or get_settings()

    def bronze(name: str) -> DataFrame:
        return spark.read.format("delta").load(str(settings.bronze_dir / name))

    translation = _category_translation(bronze)
    frames: dict[str, DataFrame] = {
        "orders": _orders(bronze),
        "order_items": _order_items(bronze),
        "order_payments": _order_payments(bronze),
        "order_reviews": _order_reviews(bronze),
        "customers": _customers(bronze),
        "sellers": _sellers(bronze),
        "products": _products(bronze, translation),
        "geolocation": _geolocation(bronze),
        "category_translation": translation,
    }
    assert set(frames) == set(SOURCES), "silver builders out of sync with SOURCES"

    checks = _checks()
    counts: dict[str, int] = {}
    all_results: list[dq.CheckResult] = []
    for name, df in frames.items():
        target = str(settings.silver_dir / name)
        df.write.format("delta").mode("overwrite").option("overwriteSchema", True).save(target)
        written = spark.read.format("delta").load(target)
        counts[name] = written.count()
        log.info("silver.written", table=name, rows=counts[name])
        all_results += dq.run_checks(name, written, checks.get(name, []))

    run_id = dq.persist_results(spark, all_results, str(settings.silver_dir / "dq_results"))
    passed = sum(r.passed for r in all_results)
    log.info("dq.summary", run_id=run_id, passed=passed, total=len(all_results))
    if fail_on_error:
        dq.raise_on_errors(all_results)
    return counts
