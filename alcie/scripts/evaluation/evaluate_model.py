import json
import torch
import os
import base64
import argparse
import sys
from PIL import Image
from torchvision import transforms
from io import BytesIO
import evaluate
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from transformers import OFATokenizer


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from ofa import OFAModelForCaption

# ========================
# IMAGE PREPROCESSING
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 480
patch_resize_transform = transforms.Compose([
    transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

def preprocess_image(image_content):
    image = Image.open(BytesIO(base64.urlsafe_b64decode(image_content))).convert("RGB")
    image = patch_resize_transform(image).unsqueeze(0)
    return image.to(device)

# ========================
# MODEL LOADING FUNCTION
# ========================
def load_model(model_path):
    tokenizer = OFATokenizer.from_pretrained(model_path)
    model = OFAModelForCaption.from_pretrained(model_path, use_cache=True).to(device)
    return tokenizer, model

# ========================
# GENERATE CAPTION FUNCTION
# ========================
def generate_caption(image_contents, tokenizer, model):
    batch = [preprocess_image(content) for content in image_contents]
    batch = torch.cat(batch, dim=0)
    txt = [" what does the image describe?"] * len(image_contents)
    inputs = tokenizer(txt, return_tensors="pt", padding=True).input_ids.to(device)
    gen = model.generate(inputs, patch_images=batch, num_beams=5, no_repeat_ngram_size=3, max_length=110)
    captions = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return captions

# ========================
# LOAD IMAGE DATA FUNCTION
# ========================
def load_image_data(file_path):
    image_id2content = {}
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            image_id, image_content = line.strip().split('\t')
            image_id2content[image_id] = image_content
    return image_id2content

# ========================
# METRICS LOADING
# ========================
metrics = {
    "BLEU": evaluate.load("bleu"),
    "ROUGE": evaluate.load("rouge", trust_remote_code=True),
    "METEOR": evaluate.load("meteor"),
    "BERTScore": evaluate.load("bertscore")
}

# ========================
# PROCESS METRICS FUNCTION
# ========================
def process_metrics(data):
    bleu4 = data["BLEU"]["precisions"][-1] if "BLEU" in data and "precisions" in data["BLEU"] else None
    rougeL = data["ROUGE"]["rougeL"] if "ROUGE" in data and "rougeL" in data["ROUGE"] else None
    meteor = data["METEOR"]["meteor"] if "METEOR" in data and "meteor" in data["METEOR"] else None
    bert_precision_avg = (
        sum(data["BERTScore"]["precision"]) / len(data["BERTScore"]["precision"])
        if "BERTScore" in data and "precision" in data["BERTScore"]
        else None
    )
    bert_recall_avg = (
        sum(data["BERTScore"]["recall"]) / len(data["BERTScore"]["recall"])
        if "BERTScore" in data and "recall" in data["BERTScore"]
        else None
    )
    bert_f1_avg = (
        sum(data["BERTScore"]["f1"]) / len(data["BERTScore"]["f1"])
        if "BERTScore" in data and "f1" in data["BERTScore"]
        else None
    )
    return {
        "BLEU4": bleu4,
        "ROUGE_L": rougeL,
        "METEOR": meteor,
        "BERTScore_Precision_Avg": bert_precision_avg,
        "BERTScore_Recall_Avg": bert_recall_avg,
        "BERTScore_F1_Avg": bert_f1_avg
    }

# ========================
# MAIN EVALUATION SCRIPT
# ========================
def main():
    parser = argparse.ArgumentParser(description="Evaluate image captions for different clusters.")
    parser.add_argument("--sampling_method", type=str, required=True, help="Sampling method: random, diversity, uncertainty")
    args = parser.parse_args()
    
    # AL = ['diversity', 'uncertainity']

    # Paths based on sampling method
    sampling_method = args.sampling_method
    # if sampling_method not in AL:
        # base_model_path = f"/home/akkumar/ALCIE/fine_tuned_models/{sampling_method}/OFA_trained_model_memory"
    # else:
    base_model_path = f"/home/akkumar/ALCIE/fine_tuned_models/{sampling_method}/OFA_trained_model_memory"
    
    print("Evaluating Model:",base_model_path)
    
    test_data_path = "/home/akkumar/ALCIE/cluster_facade_training_combined"
    eval_results_path = f"/home/akkumar/ALCIE/evaluation/{sampling_method}"
    pred_results_path = f"/home/akkumar/ALCIE/prediction/{sampling_method}"
    os.makedirs(eval_results_path, exist_ok=True)
    os.makedirs(pred_results_path, exist_ok=True)

    cluster_names = ["accessories", "bottoms", "dresses", "outerwear", "shoes", "tops"]

    for test_cluster in cluster_names:
        # Load test data for the current cluster
        test_caption_file = os.path.join(test_data_path, test_cluster, "test_caption.jsonl")
        test_image_file = os.path.join(test_data_path, test_cluster, "test_image.tsv")

        print(f"\n=== Evaluating Test Cluster: {test_cluster} ===")

        # Load image data
        with ThreadPoolExecutor() as executor:
            future_image_data = executor.submit(load_image_data, test_image_file)
            image_id2content = future_image_data.result()

        # Load test captions
        with open(test_caption_file, 'r', encoding='utf-8', errors='ignore') as f:
            test_lines = f.readlines()

        # Evaluate this test dataset using all models trained on current and following clusters
        for model_cluster in cluster_names[cluster_names.index(test_cluster):]:
            print(f"Using Model: {model_cluster} to evaluate {test_cluster}")

            model_path = os.path.join(base_model_path, model_cluster, "checkpoint-best")
            tokenizer, model = load_model(model_path)

            predictions, references, captions_predictions = [], [], []

            # Process in batches
            batch_size = 32
            for i in tqdm(range(0, len(test_lines), batch_size), desc=f"Processing {test_cluster} with {model_cluster}"):
                batch_lines = test_lines[i:i + batch_size]
                batch_items = [json.loads(line) for line in batch_lines]
                batch_image_ids = [item['image_id'] for item in batch_items]
                batch_descriptions = [item['text'][0] if isinstance(item['text'], list) else item['text'] for item in batch_items]
                batch_image_contents = [image_id2content[image_id] for image_id in batch_image_ids if image_id in image_id2content]

                # Generate captions
                batch_captions = generate_caption(batch_image_contents, tokenizer, model)

                for image_id, description, caption in zip(batch_image_ids, batch_descriptions, batch_captions):
                    predictions.append(caption)
                    references.append([description])
                    captions_predictions.append({
                        "image_id": image_id,
                        "caption": description,
                        "prediction": caption
                    })

            # Compute metrics
            results = {}
            for metric_name, metric in metrics.items():
                if metric_name == "BERTScore":
                    results[metric_name] = metric.compute(predictions=predictions, references=references, lang="en")
                else:
                    results[metric_name] = metric.compute(predictions=predictions, references=references)

            # Process and save evaluation results
            processed_results = process_metrics(results)
            eval_output_path = os.path.join(eval_results_path, f"evaluation_results_{test_cluster}_on_{model_cluster}.json")
            with open(eval_output_path, 'w') as f:
                json.dump(processed_results, f, indent=4)

            # Save predictions
            pred_output_path = os.path.join(pred_results_path, f"captions_predictions_{test_cluster}_on_{model_cluster}.json")
            with open(pred_output_path, 'w') as f:
                json.dump(captions_predictions, f, indent=4)

            print(f"Evaluation completed for {test_cluster} on {model_cluster}. Results saved.")

    print("\n Evaluation for all clusters completed!")

if __name__ == "__main__":
    main()
