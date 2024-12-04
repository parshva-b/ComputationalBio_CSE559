import os
import subprocess
import itertools
import time
import json

# Define hyperparameter combinations
hyperparameters = {
    'batch_size': [32, 64, 128],
    'drop_rate': [0.25, 0.5, 0.35],
    'lin_size': [512, 1024, 2048],
}

# Function to create the command for each hyperparameter combination
def create_command(params):
    root_name = f"model_{params['batch_size']}_{params['drop_rate']}_{params['lin_size']}_5heads_blosum45"
    model_name = f"{root_name}.ckpt"
    log_file = f"logs_{root_name}.txt"

    command = [
        "python3", "main2.py",
        "--infile", "../epi_split/train.csv",
        "--indepfile", "../epi_split/test.csv",
        "--batch_size", str(params['batch_size']),
        "--drop_rate", str(params['drop_rate']),
        "--lin_size", str(params['lin_size']),
        "--mode", "train",
        "--save_model", "True",
        "--model_name", model_name,
        "--cuda", "True",
    ]
    
    # Redirect output to log file
    # command_str = ' '.join(command) + f' > {log_file}'
    # return command_str
    return command, log_file

# Log the training run
def log_run(params, start_time, end_time):
    log_data = {
        'hyperparameters': params,
        'start_time': start_time,
        'end_time': end_time
    }

    log_filename = "training_log.json"
    if os.path.exists(log_filename):
        with open(log_filename, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_data)
    
    with open(log_filename, "w") as f:
        json.dump(logs, f, indent=4)

# Run the training for all hyperparameter combinations
def run_training():
    # Create the hyperparameter grid
    param_grid = list(itertools.product(*hyperparameters.values()))

    for param_set in param_grid:
        params = dict(zip(hyperparameters.keys(), param_set))
        
        command, log_file = create_command(params)
        start_time = time.time()
        
        # Run the training with the current hyperparameter set
        print(f"Training with parameters: {params}")
        with open(log_file, 'w') as log:
            subprocess.run(command, stdout=log, stderr=log)

        # Assuming we save performance in a file, read and evaluate performance
        end_time = time.time()
        # performance = evaluate_performance(params)  # Replace with actual function to extract performance
        
        # Log the run
        log_run(params, start_time, end_time)

        # # Save best model based on performance
        # if not best_performance or performance['accuracy'] > best_performance['accuracy']:
        #     best_performance = performance
        #     best_params = params

    # Save the best hyperparameters to a file
    # with open("best_hyperparameters.json", "w") as f:
    #     json.dump(best_params, f, indent=4)

    # print(f"Best Hyperparameters: {best_params}")
    # print(f"Best Performance: {best_performance}")

# Simulated performance evaluation (to be replaced with actual logic)
def evaluate_performance(params):
    # This function would read the performance from the log or output files after training
    # For now, it's just a placeholder returning a random performance
    performance = {
        'accuracy': round(0.5 + (0.5 * (params['lr'] / 0.001))),  # Just an example formula
        'loss': round(0.2 + (0.1 * (params['batch_size'] / 32))),  # Example loss calculation
    }
    return performance

if __name__ == '__main__':
    run_training()
