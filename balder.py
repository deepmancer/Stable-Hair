#!/usr/bin/env python
"""
balder.py - Generate bald versions of face images using Stable-Hair's bald converter.

Usage:
    python balder.py <input_images_dir> <output_dir>

For each image file in input_images_dir, generates a bald version and saves it
to output_dir with the same filename but .png extension.
"""

import os
import sys
import argparse
import glob

import torch
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from diffusers import UniPCMultistepScheduler
from diffusers.models import UNet2DConditionModel
from ref_encoder.latent_controlnet import ControlNetModel
from utils.pipeline_cn import StableDiffusionControlNetPipeline


def get_image_files(directory):
    """Get all image files from a directory."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(image_files)


class BaldConverter:
    def __init__(self, config="./configs/hair_transfer.yaml", device="cuda", weight_dtype=torch.float16):
        print("Initializing Bald Converter Pipeline...")
        self.config = OmegaConf.load(config)
        self.device = device
        self.weight_dtype = weight_dtype

        # Load UNet for controlnet initialization
        unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_model_path, subfolder="unet"
        ).to(device)

        # Load bald converter
        bald_converter = ControlNetModel.from_unet(unet).to(device)
        _state_dict = torch.load(self.config.bald_converter_path)
        bald_converter.load_state_dict(_state_dict, strict=False)
        bald_converter.to(dtype=weight_dtype)
        del unet

        # Create pipeline for hair removal
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=bald_converter,
            safety_checker=None,
            torch_dtype=weight_dtype,
        )
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )
        self.pipeline = self.pipeline.to(device)

        print("Initialization Done!")

    def get_bald(self, image, scale=0.9, size=512):
        """
        Generate a bald version of the input image.
        
        Args:
            image: PIL Image or path to image
            scale: controlnet conditioning scale (default: 0.9)
            size: output size (default: 512)
        
        Returns:
            PIL Image of the bald version
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        image = image.resize((size, size))
        W, H = image.size

        bald_image = self.pipeline(
            prompt="",
            negative_prompt="",
            num_inference_steps=30,
            guidance_scale=1.5,
            width=W,
            height=H,
            image=image,
            controlnet_conditioning_scale=scale,
            generator=None,
        ).images[0]

        return bald_image


def main():
    parser = argparse.ArgumentParser(description='Generate bald versions of face images')
    parser.add_argument('--input_images_dir', type=str, help='Directory containing input images', default="/workspace/outputs/aligned_image")
    parser.add_argument('--output_dir', type=str, help='Directory to save bald images', default="/workspace/baselines_outputs/bald/stable_hair")
    parser.add_argument('--config', type=str, default='./configs/hair_transfer.yaml',
                        help='Path to config file (default: ./configs/hair_transfer.yaml)')
    parser.add_argument('--scale', type=float, default=0.9,
                        help='ControlNet conditioning scale (default: 0.9)')
    parser.add_argument('--size', type=int, default=512,
                        help='Output image size (default: 512)')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'fp32'],
                        help='Model precision (default: fp16)')
    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_images_dir):
        print(f"Error: Input directory '{args.input_images_dir}' does not exist.")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get image files
    image_files = get_image_files(args.input_images_dir)
    if not image_files:
        print(f"No image files found in '{args.input_images_dir}'")
        sys.exit(1)

    print(f"Found {len(image_files)} image(s) to process.")

    # Initialize model
    weight_dtype = torch.float16 if args.dtype == 'fp16' else torch.float32
    converter = BaldConverter(config=args.config, weight_dtype=weight_dtype)

    # Process each image
    for i, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        output_filename = f"{name_without_ext}.png"
        output_path = os.path.join(args.output_dir, output_filename)

        print(f"[{i+1}/{len(image_files)}] Processing: {filename}")

        try:
            bald_image = converter.get_bald(image_path, scale=args.scale, size=args.size)
            bald_image.save(output_path)
            print(f"  Saved: {output_path}")

        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue

    print(f"\nDone! Processed {len(image_files)} image(s).")


if __name__ == '__main__':
    main()
