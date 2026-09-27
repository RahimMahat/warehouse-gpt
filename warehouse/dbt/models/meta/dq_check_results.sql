-- Latest result of every Spark data-quality check. The agent reads this to add caveats.
select
    run_id,
    run_at,
    table_name,
    check_name,
    severity,
    passed,
    failing_rows
from {{ ref('stg_dq_results') }}
qualify run_at = max(run_at) over ()
