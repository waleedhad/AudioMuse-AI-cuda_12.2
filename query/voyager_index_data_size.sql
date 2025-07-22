SELECT 
    pg_size_pretty(pg_total_relation_size('public.voyager_index_data')) AS total_size,
    pg_total_relation_size('public.voyager_index_data') / 1024 / 1024 AS size_mb
;
