import optuna
import json
import sys
import os
from functools import partial # For passing extra arguments to objective

# Assuming train_model and load_graphs will be imported from model_constructor
# If model_constructor.py is in the same directory (src), this should work.
# If running scripts from the root directory, src.model_constructor might be needed.
from model_constructor import train_model, load_graphs


def _suggest_hyperparameter(trial, param_config):
    """Helper function to suggest a hyperparameter based on its configuration."""
    param_type = param_config["type"]
    if param_type == "categorical":
        return trial.suggest_categorical(param_config["name"], param_config["choices"])
    elif param_type == "float":
        log = param_config.get("log", False)
        if "step" in param_config:
            return trial.suggest_float(param_config["name"], param_config["low"], param_config["high"], step=param_config["step"], log=log)
        else:
            return trial.suggest_float(param_config["name"], param_config["low"], param_config["high"], log=log)
    elif param_type == "int":
        log = param_config.get("log", False) # Optuna's suggest_int doesn't directly support log=True in older versions, but good to have
        if "step" in param_config:
             return trial.suggest_int(param_config["name"], param_config["low"], param_config["high"], step=param_config["step"], log=log)
        else:
            return trial.suggest_int(param_config["name"], param_config["low"], param_config["high"], log=log)
    else:
        raise ValueError(f"Unsupported hyperparameter type: {param_type}")


def objective(trial, optuna_config, train_PyG, test_PyG):
    """Objective function for Optuna hyperparameter optimization.

    This function is called by Optuna for each trial. It samples
    hyperparameters based on the 'optuna_params' section of the config,
    trains the model, and returns the F1 score.
    """
    # These are fixed parameters for the training runs within this Optuna study
    # They come from the 'fixed_params' section of the Optuna JSON config
    fixed_params = optuna_config["fixed_params"]

    epochs = fixed_params["epochs"]
    batch_size = fixed_params["batch_size"]
    feature_selection = fixed_params["feature_selection"]
    # graphs_path = fixed_params["graphs_path"] # Data is already loaded and passed
    model_type = fixed_params["model_type"]
    use_graphnorm = fixed_params["use_graphnorm"]
    use_adamW = fixed_params.get("use_adamW", True) # Default to True if not specified
    early_stopping = fixed_params.get("early_stopping", False)

    # Hyperparameters to be tuned by Optuna
    # These are defined in the 'hyperparameters' section of the Optuna JSON config
    tuned_params = {}
    for param_name, param_details in optuna_config["hyperparameters"].items():
        # Ensure "name" in param_details matches param_name for clarity, or just use param_name
        # Using param_name directly as the key for trial.suggest_
        # We need to map the param_name from config to the actual argument name in train_model if they differ
        # For now, assuming they are the same (e.g., "lr" in config maps to "lr" in train_model)

        # Correctly use param_details which contains type, choices/low/high etc.
        # The key `param_name` is what we use for `trial.suggest_categorical(param_name, ...)`
        # The `param_details` should have a "name" field that matches `param_name` if _suggest_hyperparameter expects it.
        # Let's simplify: _suggest_hyperparameter will use `param_name` as the name for suggestion.
        # It will take `param_details` (which is `param_config` in its scope) for type, choices etc.

        # We need to pass the name of the hyperparameter to suggest_
        # The `param_details` should contain everything needed for the suggestion type
        # E.g., {"type": "categorical", "choices": [32, 64, 128]} for hidden_channels
        # The _suggest_hyperparameter function needs `param_name` to call `trial.suggest_categorical(param_name, ...)`

        # The structure of optuna_config["hyperparameters"] is:
        # "hyperparameters": {
        #   "hidden_channels": {"type": "categorical", "choices": [32, 64, 128]},
        #   "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": true}
        # }
        # So, param_name is "hidden_channels", param_details is {"type": ..., "choices": ...}

        # We need to ensure _suggest_hyperparameter gets the actual name for the trial.
        # Let's adjust _suggest_hyperparameter to take `name` and `config_dict`

        # param_details already contains "name" field.
        # Example: "hidden_channels": {"name": "hidden_channels", "type": "categorical", "choices": [32, 64, 128]}
        # This is a bit redundant. Let's assume the key itself is the name.

        # Re-thinking the call to _suggest_hyperparameter:
        # It should be: trial.suggest_float(param_name, param_details["low"], param_details["high"], ...)
        # So, _suggest_hyperparameter needs `trial`, `param_name_for_trial`, and `param_config_for_type_and_values`

        # Let's make _suggest_hyperparameter take `trial`, `name_str`, `details_dict`
        # where name_str is "lr", and details_dict is {"type": "float", "low": 1e-4, ...}

        # current_param_config = {"name": param_name, **param_details} # Constructing what _suggest_hyperparameter expects
        # tuned_params[param_name] = _suggest_hyperparameter(trial, current_param_config)

        # Simpler: _suggest_hyperparameter takes trial, name, and the config dict for that param
        tuned_params[param_name] = _suggest_hyperparameter(trial, {"name": param_name, **param_details})

    # Call train_model with combined fixed and tuned parameters
    model = train_model(
        train_PyG=train_PyG,
        test_PyG=test_PyG,

        # Fixed parameters from optuna_config["fixed_params"]
        epochs=epochs,
        batch_size=batch_size,
        ID_model=f"optuna_{optuna_config['study_name']}_{trial.number}", # Unique ID for each trial
        model_type=model_type,
        use_graphnorm=use_graphnorm,
        feature_selection=feature_selection,
        use_adamW=use_adamW, # from fixed_params
        early_stopping=early_stopping, # from fixed_params

        # Tuned hyperparameters from optuna_config["hyperparameters"]
        # Pass them directly as kwargs
        **tuned_params
    )

    # Retrieve the metric to optimize (e.g., f1_score)
    # The path to summary_metrics.json needs to be constructed carefully
    # Results/{feature_selection}/{model_type}_results/{ID_model}/summary_metrics.json
    results_dir = f"Results/{feature_selection}/{model_type}_results/optuna_{optuna_config['study_name']}_{trial.number}"
    metrics_path = f"{results_dir}/summary_metrics.json"

    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print(f"Error: summary_metrics.json not found at {metrics_path} for trial {trial.number}", file=sys.stderr)
        # Return a value indicating failure, e.g., a very low F1 score or raise an error
        # Optuna by default tries to minimize, so for maximization, a low score is like a failure.
        # If direction is "maximize", returning 0.0 or -1.0 would be appropriate.
        # If the study might prune based on intermediate values, this needs careful handling.
        return 0.0 # Or handle as per Optuna's error handling/pruning

    metric_to_optimize = optuna_config.get("metric_to_optimize", "f1_score")
    return metrics[metric_to_optimize]


def run_optuna_study(optuna_config_path):
    """
    Loads Optuna configuration from a JSON file, sets up, and runs the Optuna study.
    """
    print(f"Loading Optuna configuration from: {optuna_config_path}")
    try:
        with open(optuna_config_path, 'r') as f:
            optuna_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Optuna configuration file not found at {optuna_config_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from Optuna configuration file {optuna_config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Optuna configuration loaded: {optuna_config}")

    study_name = optuna_config.get("study_name", "optuna_study")
    direction = optuna_config.get("direction", "maximize")
    n_trials = optuna_config.get("n_trials", 20)

    # Load data once before starting the study
    # Data paths are part of "fixed_params" in the Optuna config
    fixed_params = optuna_config["fixed_params"]
    graphs_path_suffix = fixed_params["graphs_path"] # e.g., "graphs_HVG_"
    train_data_path = f"data/{graphs_path_suffix}/train"
    test_data_path = f"data/{graphs_path_suffix}/test"

    print(f"Loading training data from: {train_data_path}")
    train_PyG = load_graphs(train_data_path)
    print(f"Loading test data from: {test_data_path}")
    test_PyG = load_graphs(test_data_path)

    # Create a partial function for the objective, passing the loaded config and data
    # This way, Optuna's study.optimize can call objective(trial) correctly
    objective_with_args = partial(objective, optuna_config=optuna_config, train_PyG=train_PyG, test_PyG=test_PyG)

    # TODO: Add Sampler and Pruner configuration from JSON if needed
    # sampler_config = optuna_config.get("sampler")
    # pruner_config = optuna_config.get("pruner")
    # sampler = ... if sampler_config else None
    # pruner = ... if pruner_config else None

    study = optuna.create_study(study_name=study_name, direction=direction) #, sampler=sampler, pruner=pruner)
    study.optimize(objective_with_args, n_trials=n_trials)

    print("\nStudy statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    # print(f"  Number of pruned trials: {len(study.get_trials(deepcopy=False, states=[TrialState.PRUNED]))}")
    # print(f"  Number of complete trials: {len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]))}")


    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value ({optuna_config.get('metric_to_optimize', 'f1_score')}): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save study results (optional)
    # study_results_path = f"Results/{optuna_config['fixed_params']['feature_selection']}/{optuna_config['fixed_params']['model_type']}_results/{study_name}_study_results.csv"
    # os.makedirs(os.path.dirname(study_results_path), exist_ok=True)
    # df_results = study.trials_dataframe()
    # df_results.to_csv(study_results_path, index=False)
    # print(f"\nOptuna study results saved to {study_results_path}")


if __name__ == '__main__':
    # This part allows testing optuna_utils.py directly
    # It requires a command-line argument for the Optuna config file
    if len(sys.argv) > 1:
        optuna_config_arg = sys.argv[1]
        print(f"Running Optuna study with configuration: {optuna_config_arg}")
        run_optuna_study(optuna_config_path=optuna_config_arg)
    else:
        print("Usage: python src/optuna_utils.py <path_to_optuna_config.json>")
        # Example: Create a dummy config for quick testing if needed
        # print("No config path provided. You can create a sample 'optuna_config.json' and run:")
        # print("python src/optuna_utils.py configs/optuna_gat_config.json")
    pass
