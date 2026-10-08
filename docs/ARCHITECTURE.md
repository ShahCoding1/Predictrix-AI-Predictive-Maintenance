# Architecture

Browser (static HTML/CSS/JavaScript) → FastAPI REST service → trusted local scikit-learn pipeline → probability, class, local reference perturbation.

Training is an offline CLI process: CSV → schema validation → stratified fit/validation/test → model selection on validation PR-AUC → refit on fit+validation → single held-out test evaluation → joblib artifact and JSON report.

No authentication or persistent prediction history is included. Run only on a trusted development environment or behind appropriate access controls. Never load untrusted `.joblib` files because pickle-based serialization can execute code.
