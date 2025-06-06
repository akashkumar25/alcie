import os
import json
import pandas as pd

# Define directories and cluster information
evaluation_dir = "/home/akkumar/ALCIE/evaluation/blip2/temp/hybrid"
clusters = ["accessories", "bottoms", "dresses", "outerwear", "shoes", "tops"]

# Metrics to extract and their sheet names
metrics_to_extract = {
    "BLEU4": "BLEU-4",
    "ROUGE_L": "ROUGE-L",
    "METEOR": "METEOR",
    "BERTScore_Precision_Avg": "BERTScore Precision",
    "BERTScore_Recall_Avg": "BERTScore Recall",
    "BERTScore_F1_Avg": "BERTScore F1"
}

# Initialize dictionaries to store DataFrames for each metric
results = {metric: pd.DataFrame(index=[cluster.capitalize() for cluster in clusters], 
                                columns=[cluster.capitalize() for cluster in clusters], 
                                dtype=float) for metric in metrics_to_extract}

# Loop through each test cluster
for test_cluster in clusters:
    # Loop through each training cluster (including current and later clusters)
    for train_cluster in clusters[clusters.index(test_cluster):]:
        # File path for evaluation results
        eval_file = os.path.join(evaluation_dir, f"evaluation_results_{test_cluster}_on_{train_cluster}.json")

        if os.path.exists(eval_file):
            with open(eval_file, "r") as f:
                data = json.load(f)

            # Extract and store each metric rounded to 4 decimal points
            for metric, sheet_name in metrics_to_extract.items():
                if metric in data:
                    metric_score = round(data[metric], 4)
                    results[metric].at[test_cluster.capitalize(), train_cluster.capitalize()] = metric_score


# Save results to an Excel file
output_excel_path = "/home/akkumar/ALCIE/blip2_hybrid2_metrics_evaluation.xlsx"
with pd.ExcelWriter(output_excel_path) as writer:
    for metric, sheet_name in metrics_to_extract.items():
        results[metric].to_excel(writer, sheet_name=sheet_name)

print(f"All metrics evaluation saved to: {output_excel_path}")
