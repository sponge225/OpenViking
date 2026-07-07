# Query-ready memory profile script

Example:

```bash
python memory_profile/profile_ov_memory.py --config config/versionrag_config.yaml --query-count 3
```

The script starts or reuses `openviking-server` for the configured dataset store and samples the server process tree as the main memory metric.

