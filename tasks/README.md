Tasks root directory.

Create per-subset folders:
- ID/task_variants/*.json
- ID/referencePics/<taskId>/*
- OOD/task_variants/*.json
- OOD/referencePics/<taskId>/*

The evaluator will prefer these paths when `--run-all-tasks` is used.