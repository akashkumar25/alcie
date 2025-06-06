# 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create directory for saving figures
output_dir = "/home/akkumar/ALCIE/graphs"
os.makedirs(output_dir, exist_ok=True)

# Load data files
diversity_file_path = "/home/akkumar/ALCIE/diversity_all_metrics_evaluation.xlsx"
random_file_path = "/home/akkumar/ALCIE/random_all_metrics_evaluation.xlsx"
uncertainty_file_path = "/home/akkumar/ALCIE/uncertainty_all_metrics_evaluation.xlsx"
random_no_delete_file_path = "/home/akkumar/ALCIE/random_no_delete_all_metrics_evaluation.xlsx"

# Load BLEU-4 and BERTScore F1 sheets
bleu_diversity_df = pd.read_excel(diversity_file_path, sheet_name="BLEU-4")
bleu_random_df = pd.read_excel(random_file_path, sheet_name="BLEU-4")
bleu_uncertainty_df = pd.read_excel(uncertainty_file_path, sheet_name="BLEU-4")
bleu_random_no_delete_df = pd.read_excel(random_no_delete_file_path, sheet_name="BLEU-4")

bert_diversity_df = pd.read_excel(diversity_file_path, sheet_name="BERTScore F1")
bert_random_df = pd.read_excel(random_file_path, sheet_name="BERTScore F1")
bert_uncertainty_df = pd.read_excel(uncertainty_file_path, sheet_name="BERTScore F1")
bert_random_no_delete_df = pd.read_excel(random_no_delete_file_path, sheet_name="BERTScore F1")

# Extract categories
categories = bleu_diversity_df.columns[1:]
n_clusters = len(categories)

# Define colors for each method
sns.set_palette("Blues", n_clusters)
colors_diversity = sns.color_palette("Blues", n_clusters)
sns.set_palette("Oranges", n_clusters)
colors_random = sns.color_palette("Oranges", n_clusters)
sns.set_palette("Greens", n_clusters)
colors_uncertainty = sns.color_palette("Greens", n_clusters)
sns.set_palette("Greys", n_clusters)
colors_random_no_delete = sns.color_palette("Greys", n_clusters)

# Fill missing values
bleu_diversity_df.fillna(0, inplace=True)
bleu_random_df.fillna(0, inplace=True)
bleu_uncertainty_df.fillna(0, inplace=True)
bleu_random_no_delete_df.fillna(0, inplace=True)
bert_diversity_df.fillna(0, inplace=True)
bert_random_df.fillna(0, inplace=True)
bert_uncertainty_df.fillna(0, inplace=True)
bert_random_no_delete_df.fillna(0, inplace=True)

# Function to plot stacked bar charts
def plot_stacked_bar_chart(df_list, labels, colors, metric_name, output_filename):
    fig, ax = plt.subplots(figsize=(14, 8))
    x_indexes = np.arange(n_clusters)
    bar_width = 0.2
    bottoms = [np.zeros(n_clusters) for _ in df_list]
    
    for i, df in enumerate(df_list):
        for j, tested_cluster in enumerate(categories):
            bars = ax.bar(x_indexes + (i - len(df_list) / 2) * (bar_width), df.iloc[:, j+1], width=bar_width, bottom=bottoms[i],
                          label=f"{labels[i]} - {tested_cluster}" if j == 0 else "", alpha=0.9, color=colors[i][j])
            
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + height/2, f'{height:.2f}', ha='center', va='center', fontsize=10, color='black')
            
            bottoms[i] += df.iloc[:, j+1]
    
    ax.set_xlabel("Trained Clusters", fontsize=12)
    ax.set_ylabel(f"{metric_name} Score", fontsize=12)
    ax.set_title(f"Final Stacked Bar Chart of {metric_name} Across Clusters\n(Diversity vs Random vs Uncertainty vs Random No Delete)", fontsize=14, fontweight='bold')
    ax.set_xticks(x_indexes)
    ax.set_xticklabels(categories, rotation=45, fontsize=10)
    ax.legend(loc='upper right', fontsize=9, ncol=2, frameon=True, title="Color Families:\n- Blue → Diversity\n- Orange → Random\n- Green → Uncertainty\n- Purple → Random No Delete")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(f"{output_dir}/{output_filename}", dpi=300, bbox_inches='tight')
    plt.show()

# Generate and save BLEU-4 stacked bar chart
plot_stacked_bar_chart(
    [bleu_diversity_df, bleu_random_df, bleu_uncertainty_df, bleu_random_no_delete_df],
    ["Diversity", "Random", "Uncertainty", "Random No Delete"],
    [colors_diversity, colors_random, colors_uncertainty, colors_random_no_delete],
    "BLEU-4 Retention",
    "bleu4_stacked_bar_retention_all_methods.png"
)

# Generate and save BERTScore F1 stacked bar chart
plot_stacked_bar_chart(
    [bert_diversity_df, bert_random_df, bert_uncertainty_df, bert_random_no_delete_df],
    ["Diversity", "Random", "Uncertainty", "Random No Delete"],
    [colors_diversity, colors_random, colors_uncertainty, colors_random_no_delete],
    "BERTScore F1",
    "bertscore_f1_stacked_bar_retention_all_methods.png"
)
