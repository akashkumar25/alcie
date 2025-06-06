import argparse
import json
import torch
import os
import clip
import sys
from os.path import join
from transformers import set_seed, TrainerCallback, TrainerState, TrainerControl, EarlyStoppingCallback, Trainer, HfArgumentParser, TrainingArguments
from sklearn.model_selection import train_test_split
from loguru import logger
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# Import memory buffer and sampling trainers dynamically
from scripts.memory_managements import DiversityMemoryReplayTrainer as DiversityTrainer, MemoryBufferDiversity
from scripts.memory_managements import RandomMemoryReplayTrainer as RandomTrainer, MemoryBufferRandom
from scripts.memory_managements import CertaintyMemoryReplayTrainer as CertainTrainer, MemoryBufferCertainty
from scripts.memory_managements import UncertainityMemoryReplayTrainer as UncertaintyTrainer, MemoryBufferUncertainty
from scripts.memory_managements import HybridMemoryReplayTrainer as HybridTrainer, MemoryBufferHybrid

from scripts.data_processing import CaptionDatasetBlip2, CaptionCollatorBlip2, CaptionArgumentsBlip2

class BasicTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, num_items_in_batch=None):
        inputs = {k: v for k, v in inputs.items() if k not in ["captions", "image_ids"]}
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def training_step(self, model, inputs, num_items_in_batch=None):
        inputs = {k: v for k, v in inputs.items() if k not in ["captions", "image_ids"]}
        return super().training_step(model, inputs, num_items_in_batch)


# Function to configure logger dynamically based on mode
def setup_logger(training_mode):
    log_filename = {
        "random": "/home/akkumar/ALCIE/logs/blip2/training_RS.log",
        "random_no_delete": "/home/akkumar/ALCIE/logs/blip2/training_RS_no_delete.log",
        "diversity": "/home/akkumar/ALCIE/logs/blip2/training_DSP.log",
        "uncertainty": "/home/akkumar/ALCIE/logs/blip2/training_US.log",
        "basic": "/home/akkumar/ALCIE/logs/blip2/training_basic.log",
        "hybrid": "/home/akkumar/ALCIE/logs/blip2/training_hybrid.log",
    }[training_mode]

    logger.add(log_filename, format="{time} {level} {message}", level="INFO")
    logger.info(f"Using {training_mode} training mode. Logs saved to {log_filename}")


# Custom callback for logging metrics
class LogMetricsCallback(TrainerCallback):
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if len(state.log_history) == 0:
            logger.info("No log entries in state.log_history at epoch end.")
            return
        logs = state.log_history[-1]
        # logger.info(f"Epoch {state.epoch} - Loss: {logs['loss']}")
        if 'eval_loss' in logs:
            logger.info(f"Epoch {state.epoch} - Eval_Loss: {logs['eval_loss']}")


# Dataset splitting function
def split_dataset(dataset, test_size=0.1):
    train_indices, eval_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=42)
    return torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, eval_indices)

# def split_dataset(dataset, subset_fraction=0.1, test_size=0.1):
#     total_size = len(dataset)
#     subset_size = int(total_size * subset_fraction)

#     # Randomly select 10% of the dataset
#     subset_indices = torch.randperm(total_size)[:subset_size]

#     # Split into train/eval
#     train_indices, eval_indices = train_test_split(subset_indices.tolist(), test_size=test_size, random_state=42)

#     return torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, eval_indices)


# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='train_ofa.json', help="Path to training arguments file")
    parser.add_argument("--memory_file", type=str, default='memory_buffer.pkl', help="Path to memory buffer file")
    parser.add_argument("--training_mode", type=str, choices=["basic", "random", "random_no_delete", "diversity", "uncertainty","hybrid"], required=True, help="Training mode")
    parser.add_argument("--use_memory_replay", action='store_true', help="Enable memory replay during training")
    parser.add_argument("--current_cluster", type=int, default=1, help="Current cluster number for memory management")
    parser.add_argument("--total_capacity", type=int, default=200, help="Total memory buffer capacity")
    parser.add_argument("--delete_percent", type=float, default=0, help="Percentage of memory deleted for current cluster")

    args = parser.parse_args()

    train_args_file = args.train_args_file
    memory_file = args.memory_file
    use_memory_replay = args.use_memory_replay
    current_cluster = args.current_cluster
    total_capacity = args.total_capacity
    delete_percent = args.delete_percent
    training_mode = args.training_mode

    # Setup logger based on training mode
    setup_logger(training_mode)

    # Load training arguments
    parser = HfArgumentParser((CaptionArgumentsBlip2, TrainingArguments))
    args, training_args = parser.parse_json_file(json_file=train_args_file)

    os.makedirs(training_args.output_dir, exist_ok=True)
    set_seed(training_args.seed)

    model_name = args.model_name_or_path
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

    # # Define PEFT config for LoRA
    # peft_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    #     bias="none",
    #     target_modules=["q_proj", "v_proj", "k_proj"]  # or modules from Q-Former/decoder
    # )
    # model = get_peft_model(model, peft_config)
    # model.enable_input_require_grads()
    # model.print_trainable_parameters()
    # Freezing encoder weights if specified
    if args.freeze_encoder and hasattr(model, "vision_model"):
        logger.info("Freezing BLIP-2 vision encoder...")
        for param in model.vision_model.parameters():
            param.requires_grad = False  # Completely freeze vision encoder
    
    if args.freeze_qformer and hasattr(model, "qformer"):
        logger.info("Freezing Q-Former...")
        for param in model.qformer.parameters():
            param.requires_grad = False
    
    else:
        for name, param in model.named_parameters():
            if "qformer" in name or "language_model" in name:
                param.requires_grad = True


    if hasattr(model, "qformer"):
        for name, module in model.qformer.named_modules():
            if hasattr(module, "dropout"):
                if isinstance(module.dropout, torch.nn.Dropout):
                    module.dropout.p = 0.2
                elif isinstance(module.dropout, float):
                    module.dropout = 0.2  # overwrite float

            if hasattr(module, "attention_dropout"):
                if isinstance(module.attention_dropout, torch.nn.Dropout):
                    module.attention_dropout.p = 0.2
                elif isinstance(module.attention_dropout, float):
                    module.attention_dropout = 0.2


    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        for name, module in model.language_model.model.named_modules():
            if hasattr(module, "dropout"):
                if isinstance(module.dropout, torch.nn.Dropout):
                    module.dropout.p = 0.1
                elif isinstance(module.dropout, float):
                    module.dropout = 0.1

            if hasattr(module, "attn_dropout"):
                if isinstance(module.attn_dropout, torch.nn.Dropout):
                    module.attn_dropout.p = 0.1
                elif isinstance(module.attn_dropout, float):
                    module.attn_dropout = 0.1




    # Load dataset and split it
    dataset = CaptionDatasetBlip2(args.train_caption_file, args.train_image_file)
    train_dataset, eval_dataset = split_dataset(dataset, test_size=0.1)

    # Collator for batching data
    data_collator = CaptionCollatorBlip2(processor)
    
    logger.info("Inspecting batch shapes before training...")
    train_loader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator)

    for batch in train_loader:
        print(batch.keys())
        logger.info(f"pixel_values: {batch['pixel_values'].shape}")
        if 'input_ids' in batch:
            logger.info(f"input_ids: {batch['input_ids'].shape}")
            print(batch['input_ids'][0])
        if 'labels' in batch:
            print("Labels:", batch['labels'][0])
            decoded_labels = processor.tokenizer.decode([token for token in batch['labels'][0] if token != -100])
            print("Decoded label caption (without -100):", decoded_labels)
        break  # only print first batch
        
    if training_mode == "basic":
        trainer = BasicTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=processor.tokenizer,
            callbacks=[
                LogMetricsCallback(),
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.0
                )
            ]
        )
    else:
        if training_mode =='diversity':
            memory_buffer = MemoryBufferDiversity(sampling_strategy=training_mode)
        elif training_mode == 'random' or training_mode == 'random_no_delete':
            memory_buffer = MemoryBufferRandom(sampling_strategy=training_mode)
        elif training_mode == 'certainty':
            memory_buffer = MemoryBufferCertainty(sampling_strategy=training_mode)
        elif training_mode == 'uncertainty':
            memory_buffer = MemoryBufferUncertainty(sampling_strategy=training_mode)
        else:
            memory_buffer = MemoryBufferHybrid(sampling_strategy=training_mode)
            
        memory_buffer.set_total_capacity(total_capacity)

        if os.path.exists(memory_file):
            memory_buffer.load(memory_file)
            logger.info("Loaded existing memory buffer.")

        # Handle memory deletion differently based on the mode
        if training_mode != "random_no_delete" and current_cluster > 1:
            memory_buffer.free_space_for_new_cluster(current_cluster, delete_percent)
            logger.info(f"Freed {delete_percent * 100}% of memory for cluster {current_cluster}")

        clip_model, preprocess = None, None
        if training_mode == "diversity" or training_mode == "hybrid":
            clip_model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")

        trainer_class = {
            "random": RandomTrainer,
            "random_no_delete": RandomTrainer,  # Use same trainer but skip deletion logic
            "diversity": DiversityTrainer,
            "uncertainty": UncertaintyTrainer,
            "hybrid": HybridTrainer,
        }[training_mode]

        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=processor.tokenizer,
            memory_buffer=memory_buffer,
            replay_freq=200,
            use_memory_replay=use_memory_replay,
            clip_model=clip_model,
            preprocess=preprocess,
            current_cluster=current_cluster,
            callbacks=[
                LogMetricsCallback(),
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.0
                )
            ]
        )

    logger.info("Starting training")
    train_result = trainer.train()
    metrics = train_result.metrics
    logger.info(f"Training completed with metrics: {metrics}")
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.save_model(join(training_args.output_dir, 'checkpoint-final'))
    processor.save_pretrained(join(training_args.output_dir, 'checkpoint-final'))

    if training_mode != "basic":
        temp_memory_file = memory_file + ".new"
        memory_buffer.save(temp_memory_file)
        os.replace(temp_memory_file, memory_file)
        logger.info(f"Updated memory buffer at {memory_file}")

    logger.info("Saving best model checkpoint")
    trainer.save_model(join(training_args.output_dir, 'checkpoint-best'))
    processor.save_pretrained(join(training_args.output_dir, 'checkpoint-best'))
    logger.info("Training process completed successfully")

    if eval_dataset:
        logger.info("Starting final evaluation")
        metrics = trainer.evaluate(eval_dataset)
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("Training completed successfully")


if __name__ == '__main__':
    main()
    