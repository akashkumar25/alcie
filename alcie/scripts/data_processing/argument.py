# arguments.py

from dataclasses import dataclass, field

@dataclass
class CaptionArguments:
    """
    Custom parameters
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    max_seq_length: int = field(metadata={"help": "Maximum input length"})
    train_caption_file: str = field(metadata={"help": "Training caption file"})
    train_image_file: str = field(metadata={"help": "Training image file"})
    test_caption_file: str = field(metadata={"help": "Testing caption file"})
    test_image_file: str = field(metadata={"help": "Testing image file"})
    model_name_or_path: str = field(metadata={"help": "Pre-trained model path"})
    freeze_encoder: bool = field(metadata={"help": "Whether to freeze encoder weights and fine-tune decoder only"})
    freeze_word_embed: bool = field(
        metadata={"help": "Whether to freeze encoder's word embedding weights. Since OFA model's encoder and decoder share word embeddings, freeze_encoder will freeze word embeddings. When freeze_word_embed=False, word embeddings will be trained together"}
    )

@dataclass
class CaptionArgumentsBlip2:
    """
    Arguments pertaining specifically to BLIP-2 for fashion captioning or similar tasks.
    """
    max_seq_length: int = field(
        metadata={"help": "Maximum text input length for BLIP-2."}
    )
    train_caption_file: str = field(
        default="",
        metadata={"help": "Path to the training captions (JSONL or similar)."}
    )
    train_image_file: str = field(
        default="",
        metadata={"help": "Path to the training images (TSV or similar)."}
    )
    test_caption_file: str = field(
        default="",
        metadata={"help": "Path to the test captions (JSONL or similar)."}
    )
    test_image_file: str = field(
        default="",
        metadata={"help": "Path to the test images (TSV or similar)."}
    )
    model_name_or_path: str = field(
        default="",
        metadata={"help": "Pretrained BLIP-2 model checkpoint name or local path."}
    )
    freeze_encoder: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to freeze the BLIP-2 vision encoder (e.g., ViT). "
                "If True, the vision model's parameters will be set to require_grad=False."
            )
        }
    )
    freeze_qformer: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to freeze the Q-Former component of BLIP-2. "
                "If True, Q-Former parameters will be set to require_grad=False."
            )
        }
    )
