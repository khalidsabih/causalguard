# Configuration

`configs/experiments/base.yaml` is the main runnable configuration.

The experiment runner accepts `--config` and `--set key=value` overrides, e.g.:

```bash
causalguard-run --config configs/experiments/base.yaml --set drift_type=covariate --set trigger=periodic
```
