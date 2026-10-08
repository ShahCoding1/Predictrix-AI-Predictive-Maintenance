# Production and Research Roadmap

## Implemented
- [x] Validated UCI-compatible ingestion and synthetic demo generator
- [x] Three ML baselines with leakage-safe preprocessing
- [x] Validation-based selection and held-out test evaluation
- [x] FastAPI prediction, batch inference, local perturbation endpoints
- [x] Responsive dashboard and comparison metrics
- [x] Tests, Dockerfile, CI workflow, setup documentation

## Next research release
- [ ] SHAP explanations (separate from reference perturbation)
- [ ] Hyperparameter search nested inside training split
- [ ] Probability calibration and threshold tuning on validation only
- [ ] Per-product-type and subgroup error analysis
- [ ] Precision-recall/ROC calibration plot exports
- [ ] Repeated-seed or cross-validation uncertainty estimates
- [ ] C-MAPSS temporal modelling with engine-wise splits

## Next platform release
- [ ] React/TypeScript SPA with accessible component library
- [ ] PostgreSQL experiment and inference audit history
- [ ] Authentication and authorization
- [ ] MLflow experiment tracking
- [ ] Upload validation and asynchronous training queue
- [ ] Model registry, drift monitoring, observability
- [ ] Rate limiting, secrets management, deployment hardening
