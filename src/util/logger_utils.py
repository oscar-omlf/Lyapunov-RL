import os
import json
import logging

def setup_run_directory_and_logging(config: dict, base_log_dir: str = "logs", evaluation: bool = False):
    
    try:
        env_name = config["environment"]
    except KeyError:
        raise KeyError("The configuration dictionary must include an 'environment' key.")
    
    try:
        model_name = config["model_name"]
    except KeyError:
        raise KeyError("The configuration dictionary must include a 'model_name' key.")
    
    # Create a folder at base_log_dir/env_name/model_name
    model_dir = os.path.join(base_log_dir, env_name, model_name)    
    os.makedirs(model_dir, exist_ok=True)

    existing_runs = [d for d in os.listdir(model_dir) if d.startswith("run_") and os.path.isdir(os.path.join(model_dir, d))]
    if not existing_runs:
        run_number = 0
    else:
        max_run = max([int(d.split('_')[1]) for d in existing_runs])
        run_number = max_run + 1

    run_dir = os.path.join(model_dir, f"run_{run_number}/")
    os.makedirs(run_dir)
    print(f"Starting new run in: {run_dir}")

    logger = logging.getLogger(f"{model_name}_run_{run_number}")
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    if evaluation is False:
        log_file = 'training.log'
    else:
        log_file = 'model_evaluation.log'

    log_file_path = os.path.join(run_dir, log_file)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(stream_handler)
    
    logger.info("Starting Run")
    logger.info(f"Run Directory: {run_dir}")
    config_str = json.dumps(config, indent=4, default=str)
    if evaluation is False:
        logger.info(f"Hyperparameters:\n{config_str}")
    
    return run_dir, logger
