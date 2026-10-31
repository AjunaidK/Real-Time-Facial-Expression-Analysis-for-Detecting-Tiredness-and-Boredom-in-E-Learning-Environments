# Trained models (optional)

The artefact runs **without** trained models by using a transparent, calibrated
heuristic detector (see `app/js/detector.js`). This lets demonstrate the full
system immediately. When trained deep models will be available, drop the converted
TensorFlow.js exports here and the app will pick them up automatically.

Expected layout after conversion:

```
app/models/
  tiredness/
    model.json
    group1-shard1of1.bin
  boredom/
    model.json
    group1-shard1of1.bin
```

## Producing these files

1. Train (see `server/src/training/`):
   ```bash
   python -m src.training.train_tiredness --data data/processed/uta_tiredness.npz --out models_out/tiredness
   python -m src.training.train_boredom   --data data/processed/daisee_boredom.npz --out models_out/boredom
   ```
2. Convert to TensorFlow.js:
   ```bash
   python -m src.training.export_tfjs \
       --tiredness models_out/tiredness/tiredness_keras.keras \
       --boredom   models_out/boredom/boredom_keras.keras \
       --out app/models
   ```

The detector prefers a trained score where available and falls back to the
heuristic per-signal, so a tiredness-only model still works (boredom stays
heuristic until its model is present).

## Reporting note

When you add trained models, compare their output against the heuristic baseline
already in the artefact — the heuristic is a legitimate interpretable baseline for
Chapter 5, and "trained model beats rule-based baseline" is exactly the kind of
result the evaluation chapter wants.
