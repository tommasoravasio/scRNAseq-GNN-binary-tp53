# GNN_approach_for_TP53_mutation

## Model Configuration

The parameters for running baseline model designs (i.e., when not using Optuna for hyperparameter fine-tuning) are managed through JSON configuration files located in the `configs/` directory.

### How it Works

The `src/model_constructor.py` script, when run with `--fine_tuning no`, accepts a `--config` argument that specifies the path to a JSON configuration file. This file contains various parameters that define the model architecture, training process, data paths, and other settings.

The `main_baseline()` function within `src/model_constructor.py` reads these parameters and uses them to initialize and train the model.

### Example Configuration

An example configuration file is provided at `configs/baseline_default.json`. This file shows the structure and available parameters:

```json
{
  "epochs": 50,
  "batch_size": 16,
  "ID_model": "GraphNorm_combat",
  "use_adamW": true,
  "model_type": "gat",
  "use_graphnorm": true,
  "feature_selection": "target",
  "graphs_path": "graphs_target_combat",
  "early_stopping": false,
  "hidden_channels": 64,
  "dropout_rate": 0.2,
  "lr": 0.0001,
  "loss_weight": false,
  "weight_decay": 1e-4,
  "heads": 1,
  "use_third_layer": false
}
```

### Running with a Configuration

The `jobs/model_run.sh` script is set up to run a baseline model using a default configuration:

```bash
python src/model_constructor.py --fine_tuning no --config configs/baseline_default.json
```

To run a different model design:
1.  Create a new JSON configuration file (e.g., `configs/my_custom_config.json`) in the `configs/` directory.
2.  Modify the parameters in this new file as needed.
3.  Update the `jobs/model_run.sh` script to point to your new configuration file:
    ```bash
    python src/model_constructor.py --fine_tuning no --config configs/my_custom_config.json
    ```
    Alternatively, you can pass the `--config` argument directly if running manually.

This approach allows for easier management and reproduction of different model designs without modifying the core Python script.