# dataset.py

import json
from torch.utils.data import Dataset
from tqdm import tqdm
import base64
from io import BytesIO
from PIL import Image
from torchvision import transforms
from loguru import logger
from json.decoder import JSONDecodeError
from tqdm import tqdm

class CaptionDatasetBlip2(Dataset):
    """
    Loads base64-encoded images from a .tsv and corresponding captions from a .jsonl,
    optimized for BLIP-2.
    """
    def __init__(self, caption_file, image_file):
        logger.info('Loading data from: {} and {}'.format(caption_file, image_file))

        # 1. Read images from a TSV (image_id -> base64)
        image_id2content = {}
        with open(image_file, 'rb') as f:
            raw_data = f.read()
            lines = raw_data.decode('utf-8', errors='ignore').splitlines()
            for line in tqdm(lines, desc="Loading images"):
                try:
                    image_id, image_content = line.split('\t')
                    image_id2content[image_id] = image_content
                except ValueError:
                    logger.warning(f"Skipping malformed line in image file: {line[:50]}...")

        # 2. Read captions from JSONL
        data_list = []
        with open(caption_file, 'rb') as f:
            raw_data = f.read()
            lines = raw_data.decode('utf-8', errors='ignore').splitlines()
            for line_number, line in enumerate(tqdm(lines, desc="Loading captions"), 1):
                try:
                    line_data = json.loads(line)
                    image_id = line_data['image_id']
                    caption = line_data['text']
                    if image_id in image_id2content:
                        data = {
                            'caption': caption.strip(),
                            'image_base64': image_id2content[image_id],
                            'image_id': image_id
                        }
                        data_list.append(data)
                    else:
                        logger.warning(f"Image ID {image_id} not found in image file. Skipping...")
                except JSONDecodeError as e:
                    logger.error(f"JSON decode error on line {line_number}: {e}")
                    logger.error(f"Problematic line: {line[:50]}...")
                except KeyError as e:
                    logger.error(f"Missing key on line {line_number}: {e}")
                    logger.error(f"Problematic line: {line[:50]}...")

        logger.info('Number of data points: {}'.format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        row = self.data_list[index]
        caption = row['caption']
        image_base64 = row['image_base64']
        image_id = row['image_id']

        patch_image = None
        try:
            image = Image.open(BytesIO(base64.urlsafe_b64decode(image_base64))).convert("RGB")
        except Exception as e:
            logger.error(f"Error opening image (ID={image_id}): {e}")
            patch_image = None

        return {
            'patch_image': image,  # Preprocessed image tensor
            'caption': caption,
            'image_id': image_id
        }




class CaptionDataset(Dataset):
    def __init__(self, caption_file, image_file):
        logger.info('Loading data from: {} and {}'.format(caption_file, image_file))
        
        # Read the content of each image
        image_id2content = {}
        with open(image_file, 'rb') as f:
            raw_data = f.read()
            lines = raw_data.decode('utf-8', errors='ignore').splitlines()
            for line in tqdm(lines, desc="Loading images"):
                try:
                    image_id, image_content = line.split('\t')
                    image_id2content[image_id] = image_content
                except ValueError:
                    logger.warning(f"Skipping malformed line in image file: {line[:50]}...")

        # Read all captions for each image to get all training data
        data_list = []
        with open(caption_file, 'rb') as f:
            raw_data = f.read()
            lines = raw_data.decode('utf-8', errors='ignore').splitlines()
            for line_number, line in enumerate(tqdm(lines, desc="Loading captions"), 1):
                try:
                    line_data = json.loads(line)
                    image_id = line_data['image_id']
                    caption = line_data['text']
                    if image_id in image_id2content:
                        data = {'caption': caption, 'image_base64': image_id2content[image_id], 'image_id': image_id}
                        data_list.append(data)
                    else:
                        logger.warning(f"Image ID {image_id} not found in image file. Skipping...")
                except JSONDecodeError as e:
                    logger.error(f"JSON decode error on line {line_number}: {e}")
                    logger.error(f"Problematic line: {line[:50]}...")
                except KeyError as e:
                    logger.error(f"Missing key on line {line_number}: {e}")
                    logger.error(f"Problematic line: {line[:50]}...")

        logger.info('Number of data points: {}'.format(len(data_list)))
        self.data_list = data_list

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 224
        patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.patch_resize_transform = patch_resize_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        row = self.data_list[index]
        caption = row['caption'].strip()
        image_base64 = row['image_base64']
        image_id = row['image_id']
        
        # Log if caption is missing
        if 'caption' not in row:
            logger.error(f"Missing 'caption' key in dataset for image ID: {image_id}")

        # Load and preprocess the image
        try:
            image = Image.open(BytesIO(base64.urlsafe_b64decode(image_base64)))
            patch_image = self.patch_resize_transform(image).unsqueeze(0)
            # print(f"Image shape after transformation: {patch_image.shape}")  # Print the shape
        except Exception as e:
            # Image loading failed
            logger.error('Error opening image, image_id: {}'.format(image_id))
            logger.exception(e)
            patch_image = None

        data = {'patch_image': patch_image, 'caption': caption, 'image_id': image_id}
        return data