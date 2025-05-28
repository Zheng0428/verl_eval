import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib as mpl

# Set scientific style
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['axes.edgecolor'] = '#333333'

# Read data from the Excel file, specifically the Fig1fnr-rule-vers sheet
excel_path = '/Users/bytedance/Desktop/verl/plot/Table for genrm.xlsx'
df = pd.read_excel(excel_path, sheet_name='Fig1fnr-rule-vers')

# Specify the order for displaying verifiers
ordered_verifiers = ['Human', 'Qwen', 'HuggingFace', 'VERL']

# Specify the order for displaying tasks
ordered_tasks = ['math', 'deepscaler', 'orz', 'skywork', 'avg']

task_name  = {
    "math": "Math",
    "deepscaler": "DeepscaleR",
    "orz": "ORZ-Math",
    "skywork": "Skywork-OR1",
    "avg": "Avg"
}


# Define custom colors and configurations for each verifier
verifier_configs = {
    'Human': {
        'color': '#A64036',  # red
        'display_name': 'Human',
        'hatch': '',
        'edgecolor': '#333333',
        'linewidth': 0.5
    },
    'HuggingFace': {
        'color': '#354E6B',  # blue
        'display_name': 'HuggingFace',
        'hatch': '',
        'edgecolor': '#333333',
        'linewidth': 0.5
    },
    'VERL': {
        'color': '#e38f54',  # orange
        'display_name': 'VERL',
        'hatch': '',
        'edgecolor': '#333333',
        'linewidth': 0.5
    },
    'Qwen': {
        'color':'#4182A4',  # teal
        'display_name': 'Qwen',
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
# Instead of np.arange which uses step=1.0 by default
indices = np.arange(0, len(ordered_tasks) * 0.8, 0.8)  # Using 0.7 as step size instead of 1.0

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
# ax.set_xlabel('Tasks', fontsize=22)
ax.set_ylabel('False Negative Rate (FNR)', fontsize=20)
# No title for scientific style
ax.set_xticks(indices)
ax.set_xticklabels([task_name[task] for task in ordered_tasks], fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.set_ylim(0, 0.25)
# Add grid
ax.grid(axis='both', linestyle='--', alpha=0.3)

# Add legend with custom styling
ax.legend(fontsize=18, frameon=True, framealpha=0.9, edgecolor='#cccccc', loc='upper left')

# Ensure values are visible by adding text annotations
def autolabel(idx, verifier):
    values = verifier_data[verifier]
    config = verifier_configs.get(verifier, {})
    
    for j, val in enumerate(values):
        # if val == 0:
        #     continue
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
plt.savefig('Figures/fig1_fnr_bar.png', dpi=500, bbox_inches='tight')
plt.savefig('Figures/fig1_fnr_bar.pdf', dpi=500, bbox_inches='tight')
plt.show() 