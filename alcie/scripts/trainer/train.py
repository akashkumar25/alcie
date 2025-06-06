# training script for OFA model with various memory management strategies


import argparse
import json
import torch
import os
import clip
import sys
from os.path import join
from transformers import OFATokenizer, HfArgumentParser, TrainingArguments, set_seed, TrainerCallback, TrainerState, TrainerControl, Trainer
from sklearn.model_selection import train_test_split
from loguru import logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from ofa import OFAModelForCaption

# Import memory buffer and sampling trainers dynamically
# from scripts.memory_managements import MemoryBuffer
from scripts.memory_managements import DiversityMemoryReplayTrainer as DiversityTrainer, MemoryBufferDiversity
from scripts.memory_managements import RandomMemoryReplayTrainer as RandomTrainer, MemoryBufferRandom
from scripts.memory_managements import CertaintyMemoryReplayTrainer as CertainTrainer, MemoryBufferCertainty
from scripts.memory_managements import UncertainityMemoryReplayTrainer as UncertaintyTrainer, MemoryBufferUncertainty
from scripts.memory_managements import HybridMemoryReplayTrainer as HybridTrainer, MemoryBufferHybrid

from scripts.data_processing import CaptionDataset, CaptionArguments, CaptionCollator

class BasicTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Exclude 'caption' from the input batch
        inputs = {k: v for k, v in inputs.items() if k != 'caption'}
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def training_step(self, model, inputs):
        # Exclude 'caption' from the input batch during training as well
        inputs = {k: v for k, v in inputs.items() if k != 'caption'}
        return super().training_step(model, inputs)

# Function to configure logger dynamically based on mode
def setup_logger(training_mode):
    log_filename = {
        "random": "/home/akkumar/ALCIE/logs/training_RS.log",
        "random_no_delete": "/home/akkumar/ALCIE/logs/training_RS_no_delete.log",
        "diversity": "/home/akkumar/ALCIE/logs/training_DSP.log",
        "certainty": "/home/akkumar/ALCIE/logs/training_CS.log",
        "uncertainty": "/home/akkumar/ALCIE/logs/training_US.log",
        "basic": "/home/akkumar/ALCIE/logs/training_basic.log",
        "hybrid": "/home/akkumar/ALCIE/logs/training_hybrid.log"
    }[training_mode]

    logger.add(log_filename, format="{time} {level} {message}", level="INFO")
    logger.info(f"Using {training_mode} training mode. Logs saved to {log_filename}")


# Custom callback for logging metrics
class LogMetricsCallback(TrainerCallback):
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        logs = state.log_history[-1]
        logger.info(f"Epoch {state.epoch} - Loss: {logs['loss']}")
        if 'eval_loss' in logs:
            logger.info(f"Epoch {state.epoch} - Eval_Loss: {logs['eval_loss']}")


# Dataset splitting function
def split_dataset(dataset, test_size=0.1):
    train_indices, eval_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=42)
    return torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, eval_indices)


# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='train_ofa.json', help="Path to training arguments file")
    parser.add_argument("--memory_file", type=str, default='memory_buffer.pkl', help="Path to memory buffer file")
    parser.add_argument("--training_mode", type=str, choices=["basic", "random", "random_no_delete", "diversity", "certainty","uncertainty", "hybrid"], required=True, help="Training mode")
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
    parser = HfArgumentParser((CaptionArguments, TrainingArguments))
    args, training_args = parser.parse_json_file(json_file=train_args_file)

    os.makedirs(training_args.output_dir, exist_ok=True)
    set_seed(training_args.seed)

    tokenizer = OFATokenizer.from_pretrained(args.model_name_or_path)
    model = OFAModelForCaption.from_pretrained(args.model_name_or_path)

    # Freezing encoder weights if specified
    if args.freeze_encoder:
        for name, param in model.encoder.named_parameters():
            if 'embed_tokens' in name and not args.freeze_word_embed:
                param.requires_grad = True
            else:
                param.requires_grad = False

    full_dataset = CaptionDataset(args.train_caption_file, args.train_image_file)
    train_dataset, eval_dataset = split_dataset(full_dataset, test_size=0.1)

    data_collator = CaptionCollator(tokenizer=tokenizer, max_seq_length=args.max_seq_length)

    if training_mode == "basic":
        trainer = BasicTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[
                LogMetricsCallback()
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
        elif training_mode == 'hybrid':
            memory_buffer = MemoryBufferHybrid(sampling_strategy=training_mode)


        memory_buffer.set_total_capacity(total_capacity)

        if os.path.exists(memory_file):
            memory_buffer.load(memory_file)
            logger.info("Loaded existing memory buffer.")

        # Handle memory deletion differently based on the mode
        if training_mode != "random_no_delete" and current_cluster > 1:
            memory_buffer.free_space_for_new_cluster(current_cluster, delete_percent)
            logger.info(f"Freed {delete_percent * 100}% of memory for cluster {current_cluster}")
            print(f"Freed {delete_percent * 100}% of memory for cluster {current_cluster}")

        clip_model, preprocess = None, None
        if training_mode == "diversity" or "certainty" or "uncertainty":
            clip_model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")

        trainer_class = {
            "random": RandomTrainer,
            "random_no_delete": RandomTrainer,  # Use same trainer but skip deletion logic
            "diversity": DiversityTrainer,
            "certainty":CertainTrainer,
            "uncertainty": UncertaintyTrainer,
            "hybrid": HybridTrainer
        }[training_mode]

        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            memory_buffer=memory_buffer,
            replay_freq=200,
            use_memory_replay=use_memory_replay,
            clip_model=clip_model,
            preprocess=preprocess,
            current_cluster=current_cluster,
            callbacks=[
                LogMetricsCallback(),
            ]
        )

    logger.info("Starting training")
    train_result = trainer.train()
    metrics = train_result.metrics
    logger.info(f"Training completed with metrics: {metrics}")
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info("Saving best model checkpoint")
    trainer.save_model(join(training_args.output_dir, 'checkpoint-best'))
    logger.info("Training process completed successfully")

    if training_mode != "basic":
        temp_memory_file = memory_file + ".new"
        memory_buffer.save(temp_memory_file)
        os.replace(temp_memory_file, memory_file)
        logger.info(f"Updated memory buffer at {memory_file}")

    
if __name__ == '__main__':
    main()
