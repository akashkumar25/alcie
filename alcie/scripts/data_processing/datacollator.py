# datacollator.py

from typing import Any, Dict, List
import torch
from loguru import logger
from transformers import Blip2Processor
from torch.nn.utils.rnn import pad_sequence

class CaptionCollatorBlip2:
    def __init__(self, processor, max_target_length=130):
        self.processor = processor
        self.max_target_length = max_target_length

    def __call__(self, features):
        # Filter out samples without valid image
        valid_features = [f for f in features if f['patch_image'] is not None]
        if not valid_features:
            raise ValueError("No valid samples in batch.")

        images = [f['patch_image'] for f in valid_features]
        captions = [f['caption'] for f in valid_features]
        image_ids = [f['image_id'] for f in valid_features]

        # Process both images and captions
        encoding = self.processor(
            images=images,
            text=captions,
            padding="longest",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
            add_special_tokens=True
        )

        # Labels with padding tokens masked as -100
        labels = encoding["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        encoding["labels"] = labels

        # Optionally log raw data for analysis
        encoding["captions"] = captions
        encoding["image_ids"] = image_ids

        return encoding


class CaptionCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        captions, patch_images, image_ids = [], [], []
        for data in features:
            # Skip images if preprocessing failed
            if data['patch_image'] is None:
                continue
            if 'caption' not in data:
                logger.error(f"Missing 'caption' key in batch data: {data}")
            captions.append(data['caption'])
            patch_images.append(data['patch_image'])
            image_ids.append(data['image_id'])
        
        # Get encoder inputs
        input_ids = self.tokenizer(
            ['what does the image describe?'] * len(captions), return_tensors="pt", max_length=self.max_seq_length, truncation=True, padding=True
        ).input_ids
        patch_images = torch.cat(patch_images, dim=0)

        # Get decoder inputs
        inputs = self.tokenizer(
            captions, return_tensors="pt", max_length=self.max_seq_length, truncation=True, padding=True
        )
        decoder_input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        inputs = {
            'input_ids': input_ids,
            'patch_images': patch_images,
            'decoder_input_ids': decoder_input_ids,
            'attention_mask': attention_mask,
            'captions': captions,  # Add raw captions here
            'image_ids': image_ids,  # Include image_ids
            'return_loss': True
        }
        return inputs
