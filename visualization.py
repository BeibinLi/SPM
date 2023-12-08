import os
import json
import matplotlib.pyplot as plt
import numpy as np

window_size = 99
max_steps = 5000


# Function to calculate moving average with handling null values
def moving_average(data, window_size):
    result = []
    for i in range(len(data)):
        window_data = data[max(i - window_size //
                               2, 0):min(i + window_size // 2 + 1, len(data))]
        # Filter out null values
        filtered_data = [x for x in window_data if x is not None]
        if filtered_data != []:
            result.append(np.mean(filtered_data))
        else:
            result.append(None)
    return np.array(result)


# Experiment directories
expdirs = [
    # "results/015_rl_finetune/", "results/016_rl_finetune/",
    # "results/030_rl_finetune/", "results/032_rl_finetune/", "results/033_rl_finetune/",
    # "results/034_rl_finetune/", "results/035_rl_finetune/", "results/037_rl_finetune/", "results/038_rl_finetune/",
    # "results/042_rl_finetune/", "results/043_rl_finetune/", "results/044_rl_finetune/", "results/045_rl_finetune/",
    "results/050_rl_finetune/"
]

# Initialize dictionaries to store data for plotting
loss_data = {}
cost_data = {}

# Process each experiment directory
for expdir in expdirs:
    loss_data[expdir] = []
    cost_data[expdir] = []

    # Iterate through checkpoint folders
    for checkpoint_folder in os.listdir(expdir):
        checkpoint_path = os.path.join(expdir, checkpoint_folder, 'logs.json')

        # Check if logs.json exists
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as file:
                logs = json.load(file)

                # Extract 'iter', 'loss', and 'cost' values
                for log in logs:
                    iter_val = log['iter']
                    loss_val = log['loss'] if 'loss' in log and log[
                        'loss'] is not None else None
                    cost_val = log['cost'] if 'cost' in log and log[
                        'cost'] is not None else None

                    # Append data for plotting
                    if iter_val < max_steps:
                        loss_data[expdir].append((iter_val, loss_val))
                        cost_data[expdir].append((iter_val, cost_val))

    loss_data[expdir].sort(key=lambda x: x[0])
    cost_data[expdir].sort(key=lambda x: x[0])

# Create plots
fig, ax = plt.subplots(2, 1, figsize=(10, 12))

# Plotting loss vs iter
for expdir, data in loss_data.items():
    data = np.array(data)
    iters, losses = data[:, 0], data[:, 1]
    smoothed_losses = moving_average(losses, window_size)

    indexes = np.where(smoothed_losses != None)[0]

    ax[0].plot(iters[indexes],
               smoothed_losses[indexes],
               label=f'{expdir.split("/")[-2]} - Smoothed',
               linestyle='--')

ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('Loss')
ax[0].set_title('Loss vs Iteration for each experiment')
ax[0].legend()

# Plotting cost vs iter
for expdir, data in cost_data.items():
    data = np.array(data)
    iters, costs = data[:, 0], data[:, 1]
    smoothed_costs = moving_average(costs, window_size)

    indexes = np.where(smoothed_costs != None)[0]

    ax[1].plot(iters[indexes],
               smoothed_costs[indexes],
               label=f'{expdir.split("/")[-2]} - Smoothed',
               linestyle='--')

ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('Cost')
ax[1].set_title('Cost vs Iteration for each experiment')
ax[1].legend()

plt.tight_layout()
plt.savefig("visualization.pdf", format="pdf", bbox_inches="tight")
