import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib as mpl

script_name = ".".join(os.path.basename(__file__).split('.')[:-1])
# Set scientific style
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['axes.edgecolor'] = '#333333'

# Read data from the Excel file, specifically the Fig2:recall-rule-ver-diff-models sheet
excel_path = 'Table for genrm.xlsx'
df = pd.read_excel(excel_path, sheet_name='Fig2recall-rule-ver-diff-models')

# Specify the order for displaying verifiers
ordered_verifiers = ['qwen-math-7b',  
                     'Qwen-2.5-32B', 'DeepSeek-R1-Distill-Qwen-7B-new','DeepSeek-R1-Distill-Qwen-32B-new']
# ordered_verifiers = ['DeepSeek-R1-Distill-Qwen-32B-new',
                    #  'Qwen-2.5-32B', 'DeepSeek-R1-Distill-Qwen-7B-new','qwen-math-7b']
# Specify the order for displaying tasks
ordered_tasks = ['math', 'deepscaler', 'orz', 'skywork', 'avg']

task_name = {
    "math": "Math",
    "deepscaler": "DeepscaleR",
    "orz": "ORZ-Math",
    "skywork": "Skywork",
    "avg": "Avg"
}

# Define custom colors and configurations for each verifier
verifier_configs = {
    'DeepSeek-R1-Distill-Qwen-32B-new': {
        'color': '#A64036',  # red
        'display_name': 'DS-Distill-Qwen-32B',
        'hatch': '',
        'edgecolor': '#333333',
        'linewidth': 0.5
    },
    'DeepSeek-R1-Distill-Qwen-7B-new': {
        'color': '#354E6B',  # blue
        'display_name': 'DS-Distill-Qwen-7B',
        'hatch': '',
        'edgecolor': '#333333',
        'linewidth': 0.5
    },
    'Qwen-2.5-32B': {
        'color': '#e38f54',  # orange
        'display_name': 'Qwen2.5-32B-Instruct',
        'hatch': '',
        'edgecolor': '#333333',
        'linewidth': 0.5
    },
    'qwen-math-7b': {
        'color': '#4182A4',  # teal
        'display_name': 'Qwen2.5-Math-7B-Instruct',
        'hatch': '',
        'edgecolor': '#333333',
        'linewidth': 0.5
    }
}

# Extract all tasks from the DataFrame (excluding 'Verifier' column)
all_tasks = df.columns[1:].tolist()

# Ensure all requested tasks exist in the data
for task in ordered_tasks:
    if task != 'avg' and task not in all_tasks:
        raise ValueError(f"Task '{task}' not found in the data")

# Create a dictionary to map verifier names to their data
verifier_data = {}
for _, row in df.iterrows():
    verifier = row['Verifier']
    # Create a dictionary for this verifier with task -> value mapping
    task_values = {}
    for i, task in enumerate(all_tasks):
        # Use recall directly (don't convert to FNR)
        task_values[task] = row.values[i+1]  # +1 to skip the Verifier column
    
    # Reorder the data according to ordered_tasks
    ordered_values = []
    for task in ordered_tasks:
        if task == 'avg':
            # Calculate average for the displayed tasks (excluding avg itself)
            displayed_tasks = [t for t in ordered_tasks if t != 'avg']
            avg_value = sum(task_values[t] for t in displayed_tasks) / len(displayed_tasks)
            ordered_values.append(avg_value)
        else:
            ordered_values.append(task_values[task])
    
    verifier_data[verifier] = ordered_values

# Ensure all requested verifiers exist in the data
for verifier in ordered_verifiers:
    if verifier not in verifier_data:
        raise ValueError(f"Verifier '{verifier}' not found in the data")

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 5))

# Set width of bars
bar_width = 0.18
# Reduce spacing between dataset groups by using smaller steps
indices = np.arange(0, len(ordered_tasks) * 0.83, 0.83)  # Using 0.83 as step size like in fig1

# Create bars for each verifier in the specified order
for i, verifier in enumerate(ordered_verifiers):
    config = verifier_configs.get(verifier, {})
    color = config.get('color', f'C{i}')
    display_name = config.get('display_name', verifier)
    hatch = config.get('hatch', '')
    edgecolor = config.get('edgecolor', '#333333')
    linewidth = config.get('linewidth', 0.5)
    
    ax.bar(indices + (i - len(ordered_verifiers)/2 + 0.5) * bar_width, 
           verifier_data[verifier], 
           bar_width,
           label=display_name, 
           color=color, 
           hatch=hatch, 
           edgecolor=edgecolor,
           linewidth=linewidth)

# Set labels and title
ax.set_ylabel('Recall Rate', fontsize=23)
ax.set_xticks(indices)
ax.set_xticklabels([task_name[task] for task in ordered_tasks], fontsize=23)
ax.tick_params(axis='y', labelsize=23)
ax.set_ylim(0.85, 1.04)  # Adjusted for Recall values
# Set custom y-ticks that only go up to 1.0
ax.set_yticks([0.85, 0.90, 0.95, 1.00])
# Add grid
ax.grid(axis='both', linestyle='--', alpha=0.3)

# Add legend with custom styling
ax.legend(fontsize=19, 
          frameon=True, 
          framealpha=0.9, 
          edgecolor='#cccccc', 
          loc='upper right',  # Position at the upper right
          borderaxespad=0)

# Ensure values are visible by adding text annotations
def autolabel(idx, verifier):
    values = verifier_data[verifier]
    config = verifier_configs.get(verifier, {})
    
    for j, val in enumerate(values):
        ax.annotate(f'{val:.2f}',
                xy=(indices[j] + (idx - len(ordered_verifiers)/2 + 0.5) * bar_width, val),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=15)

for i, verifier in enumerate(ordered_verifiers):
    autolabel(i, verifier)

# Add border
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_color('#333333')

# Create directory for figures if it doesn't exist
os.makedirs('Figures', exist_ok=True)

# Adjust layout and save
plt.tight_layout()
plt.savefig(f'Figures/{script_name}_recall_rule-ver-diff-model.png', dpi=500, bbox_inches='tight')
plt.savefig(f'Figures/{script_name}_recall_rule-ver-diff-model.pdf', dpi=500, bbox_inches='tight')
plt.show()
