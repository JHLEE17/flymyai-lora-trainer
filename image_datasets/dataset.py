import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random

def throw_one(probability: float) -> int:
    return 1 if random.random() < probability else 0


def image_resize(img, max_size=512):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h))

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def crop_to_aspect_ratio(image, ratio="16:9"):
    width, height = image.size
    ratio_map = {
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1)
    }
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    cropped_img = image.crop(crop_box)
    return cropped_img


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512, caption_type='txt',
                 random_ratio=False, caption_dropout_rate=0.1, cached_text_embeddings=None,
                 cached_image_embeddings=None, txt_cache_dir=None, img_cache_dir=None):
        # Get image files with proper filtering
        image_files = []
        for filename in os.listdir(img_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                full_path = os.path.join(img_dir, filename)
                if os.path.isfile(full_path):  # Ensure it's actually a file
                    image_files.append(full_path)
        
        self.images = sorted(image_files)
        print(f"Found {len(self.images)} image files in {img_dir}")
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.caption_dropout_rate = caption_dropout_rate
        self.cached_text_embeddings = cached_text_embeddings
        self.cached_image_embeddings = cached_image_embeddings
        self.txt_cache_dir = txt_cache_dir
        self.img_cache_dir = img_cache_dir
        print('cached_text_embeddings', type(cached_text_embeddings))
        
    def __len__(self):
        return 999999

    def __getitem__(self, idx):
        try:
            idx = random.randint(0, len(self.images) - 1)
            if self.cached_image_embeddings is None and self.img_cache_dir is None:
                img = Image.open(self.images[idx]).convert('RGB')
                if self.random_ratio:
                    ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                    if ratio != "default":
                        img = crop_to_aspect_ratio(img, ratio)
                img = image_resize(img, self.img_size)
                w, h = img.size
                new_w = (w // 32) * 32
                new_h = (h // 32) * 32
                img = img.resize((new_w, new_h))
                img = torch.from_numpy((np.array(img) / 127.5) - 1)
                img = img.permute(2, 0, 1)
            elif self.img_cache_dir is not None:
                img_filename = os.path.basename(self.images[idx]) + '.pt'
                img = torch.load(os.path.join(self.img_cache_dir, img_filename))
            else:
                img_filename = os.path.basename(self.images[idx])
                img = self.cached_image_embeddings[img_filename]
            # Create corresponding text file path safely
            image_path = self.images[idx]
            txt_path = os.path.splitext(image_path)[0] + '.' + self.caption_type
            if self.cached_text_embeddings is None and self.txt_cache_dir is None:
                try:
                    if not os.path.exists(txt_path):
                        print(f"Warning: Text file not found: {txt_path}")
                        # Return empty prompt if text file doesn't exist
                        return img, " "
                    prompt = open(txt_path, 'r', encoding='utf-8').read().strip()
                    if throw_one(self.caption_dropout_rate):
                        return img, " "
                    else:
                        return img, prompt
                except Exception as e:
                    print(f"Error reading text file {txt_path}: {e}")
                    return img, " "
            elif self.txt_cache_dir is not None:
                if throw_one(self.caption_dropout_rate):
                    empty_cache_path = os.path.join(self.txt_cache_dir, 'empty_embedding.pt')
                    txt_embs = torch.load(empty_cache_path)
                    return img, txt_embs['prompt_embeds'], txt_embs['prompt_embeds_mask']
                else:
                    # Get the base filename and create cache path
                    txt_filename = os.path.basename(txt_path) + '.pt'
                    cache_path = os.path.join(self.txt_cache_dir, txt_filename)
                    txt_embs = torch.load(cache_path)
                    return img, txt_embs['prompt_embeds'], txt_embs['prompt_embeds_mask']
            else:
                txt_filename = os.path.basename(txt_path)
                if throw_one(self.caption_dropout_rate):
                    return img, self.cached_text_embeddings['empty_embedding']['prompt_embeds'], self.cached_text_embeddings['empty_embedding']['prompt_embeds_mask']
                else:
                    return img, self.cached_text_embeddings[txt_filename]['prompt_embeds'], self.cached_text_embeddings[txt_filename]['prompt_embeds_mask']
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))

def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
