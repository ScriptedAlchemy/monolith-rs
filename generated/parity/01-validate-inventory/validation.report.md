# Parity Inventory Validation Report

## Checklist Convention

- Inferred convention: unknown
- Expected example checklist path: `monolith-rs/parity/monolith/__init__.md`

## Counts

- Python files (`monolith/**/*.py`): 334 (expected: 334)
- Parity checklist files (`monolith-rs/parity/**/*.md`): 0
- Mapped checklist files: 0
- Parity index exists: no (`monolith-rs/PYTHON_PARITY_INDEX.md`)
- Parity index rows: 0
- Parity index python rows: 0

## Invariants

- pythonFileCountMatchesExpected: true
- checklistFileCountMatchesPythonFileCount: false
- checklistCoversAllPythonFiles: false
- noUnmatchedChecklistFiles: false
- indexRowCountMatchesPythonFileCount: false
- indexRowCountMatchesChecklistCount: false
- parityIndexCoversAllPythonFiles: false

## Missing Parity Checklists

- Total missing checklists: 334
- First 50 missing checklist targets:
  1. `monolith/__init__.py`
  2. `monolith/agent_service/__init__.py`
  3. `monolith/agent_service/agent.py`
  4. `monolith/agent_service/agent_base.py`
  5. `monolith/agent_service/agent_client.py`
  6. `monolith/agent_service/agent_controller.py`
  7. `monolith/agent_service/agent_controller_test.py`
  8. `monolith/agent_service/agent_service.py`
  9. `monolith/agent_service/agent_service_test.py`
  10. `monolith/agent_service/agent_v1.py`
  11. `monolith/agent_service/agent_v3.py`
  12. `monolith/agent_service/agent_v3_test.py`
  13. `monolith/agent_service/backends.py`
  14. `monolith/agent_service/backends_test.py`
  15. `monolith/agent_service/client.py`
  16. `monolith/agent_service/constants.py`
  17. `monolith/agent_service/data_def.py`
  18. `monolith/agent_service/data_def_test.py`
  19. `monolith/agent_service/mocked_tfserving.py`
  20. `monolith/agent_service/mocked_tfserving_test.py`
  21. `monolith/agent_service/mocked_zkclient.py`
  22. `monolith/agent_service/mocked_zkclient_test.py`
  23. `monolith/agent_service/model_manager.py`
  24. `monolith/agent_service/model_manager_test.py`
  25. `monolith/agent_service/replica_manager.py`
  26. `monolith/agent_service/replica_manager_test.py`
  27. `monolith/agent_service/resource_utils.py`
  28. `monolith/agent_service/resource_utils_test.py`
  29. `monolith/agent_service/run.py`
  30. `monolith/agent_service/svr_client.py`
  31. `monolith/agent_service/tfs_client.py`
  32. `monolith/agent_service/tfs_client_test.py`
  33. `monolith/agent_service/tfs_monitor.py`
  34. `monolith/agent_service/tfs_monitor_test.py`
  35. `monolith/agent_service/tfs_wrapper.py`
  36. `monolith/agent_service/utils.py`
  37. `monolith/agent_service/utils_test.py`
  38. `monolith/agent_service/zk_mirror.py`
  39. `monolith/agent_service/zk_mirror_test.py`
  40. `monolith/base_runner.py`
  41. `monolith/common/python/mem_profiling.py`
  42. `monolith/core/__init__.py`
  43. `monolith/core/auto_checkpoint_feed_hook.py`
  44. `monolith/core/base_embedding_host_call.py`
  45. `monolith/core/base_embedding_host_call_test.py`
  46. `monolith/core/base_embedding_task.py`
  47. `monolith/core/base_host_call.py`
  48. `monolith/core/base_layer.py`
  49. `monolith/core/base_layer_test.py`
  50. `monolith/core/base_model_params.py`

## Extra Parity Checklists

- Total extra checklists: 0
- First 50 extra checklists: (none)

## Parity Index Coverage

- Parity index does not exist; skipping missing/extra python path checks in index.

## Warnings

- Missing monolith-rs/PYTHON_PARITY_INDEX.md.
- No parity checklists matched monolith-rs/parity/**/*.md.
- Could not infer parity checklist naming convention from existing files; using default expectations.
- Missing parity checklists for 334 python files.
