# ADR-001: Pin Spark 4.0 + Delta 4.0, run on JDK 21

**Status:** accepted

## Context
The project must run on a Windows laptop with no admin changes and at zero cost. Several combinations were tried:

| Combination | Result |
|---|---|
| PySpark 4.2 on system Java 24 | `UnsupportedOperationException: getSubject is not supported` (Hadoop uses an API removed in Java 23+) |
| PySpark 4.2 / 4.1 + delta-spark 4.4 on JDK 21 | `NoSuchMethodError: ParserInterface.$init$`, a binary mismatch between Delta and Spark |
| **PySpark 4.0.x + delta-spark 4.0.x on JDK 21** | ✅ works; DuckDB's `delta_scan` reads the tables |

## Decision
- Pin `pyspark>=4.0.1,<4.1` and `delta-spark>=4.0,<4.1`.
- Use a portable JDK 21 plus Hadoop `winutils.exe`/`hadoop.dll`, configured through `WGPT_JAVA_HOME` and `WGPT_HADOOP_HOME`. `spark_session.py` sets them per process.
- Do not use `createDataFrame(<python list>)` or Python UDFs. Both route through Python worker processes, which crashed on Windows. Small literal frames are built on the JVM instead (`spark_utils.literal_df`). This also avoids serialization overhead.

## Consequences
- Revisit when Delta publishes a release built against Spark ≥ 4.1 GA.
- On Linux/CI only a JDK 17/21 is needed; the Hadoop native libs are Windows-only.
