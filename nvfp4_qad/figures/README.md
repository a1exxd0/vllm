# figures/

Generated files — not tracked by git (see `.gitignore`).

## Method-explainer figures (synthetic)

These illustrate codec properties, not training results:

```bash
python -m nvfp4_qad.figures
```

Writes `fig1_nvfp4_quantizer.png` and `fig2_quant_error_vs_scale.png`.

## Training dashboard figures (real run)

```bash
# During or after training:
python -m nvfp4_qad.dashboard runs/nvfp4_qad.jsonl
```

Writes `training_loss.png`, `scale_evolution.png`, `eval_vs_step.png`,
`eval_vs_context.png` next to the log file.

## Layout preview (synthetic sample data)

```bash
python -m nvfp4_qad.dashboard   # no argument → writes watermarked SAMPLE preview
```
