import argparse
import torch
from PIL import Image
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import os
import cv2
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from diffusers import DDIMScheduler, UniPCMultistepScheduler
from diffusers.models import UNet2DConditionModel
from ref_encoder.latent_controlnet import ControlNetModel
from ref_encoder.adapter import *
from ref_encoder.reference_unet import ref_unet
from utils.pipeline import StableHairPipeline
from utils.pipeline_cn import StableDiffusionControlNetPipeline

def concatenate_images(image_files, output_file, type="pil"):
    if type == "np":
        image_files = [Image.fromarray(img) for img in image_files]
    images = image_files  # list
    max_height = max(img.height for img in images)
    images = [img.resize((img.width, max_height)) for img in images]
    total_width = sum(img.width for img in images)
    combined = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    combined.save(output_file)

class StableHair:
    def __init__(self, config="stable_hair/configs/hair_transfer.yaml", device="cuda", weight_dtype=torch.float16) -> None:
        print("Initializing Stable Hair Pipeline...")
        self.config = OmegaConf.load(config)
        self.device = device

        ### Load controlnet
        unet = UNet2DConditionModel.from_pretrained(self.config.pretrained_model_path, subfolder="unet").to(device)
        controlnet = ControlNetModel.from_unet(unet).to(device)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.controlnet_path))
        controlnet.load_state_dict(_state_dict, strict=False)
        controlnet.to(weight_dtype)

        ### >>> create pipeline >>> ###
        self.pipeline = StableHairPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=weight_dtype,
        ).to(device)
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)

        ### load Hair encoder/adapter
        self.hair_encoder = ref_unet.from_pretrained(self.config.pretrained_model_path, subfolder="unet").to(device)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.encoder_path))
        self.hair_encoder.load_state_dict(_state_dict, strict=False)
        self.hair_adapter = adapter_injection(self.pipeline.unet, device=self.device, dtype=torch.float16, use_resampler=False)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.adapter_path))
        self.hair_adapter.load_state_dict(_state_dict, strict=False)

        ### load bald converter
        bald_converter = ControlNetModel.from_unet(unet).to(device)
        _state_dict = torch.load(self.config.bald_converter_path)
        bald_converter.load_state_dict(_state_dict, strict=False)
        bald_converter.to(dtype=weight_dtype)
        del unet

        ### create pipeline for hair removal
        self.remove_hair_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=bald_converter,
            safety_checker=None,
            torch_dtype=weight_dtype,
        )
        self.remove_hair_pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self.remove_hair_pipeline.scheduler.config)
        self.remove_hair_pipeline = self.remove_hair_pipeline.to(device)

        ### move to fp16
        self.hair_encoder.to(weight_dtype)
        self.hair_adapter.to(weight_dtype)

        print("Initialization Done!")

    def Hair_Transfer(self, source_image, reference_image, random_seed, step, guidance_scale, scale, controlnet_conditioning_scale, size=512):
        prompt = ""
        n_prompt = ""
        random_seed = int(random_seed)
        step = int(step)
        guidance_scale = float(guidance_scale)
        scale = float(scale)

        # load imgs
        source_image = Image.open(source_image).convert("RGB").resize((size, size))
        id = np.array(source_image)
        reference_image = np.array(Image.open(reference_image).convert("RGB").resize((size, size)))
        source_image_bald = np.array(self.get_bald(source_image, scale=0.9))
        H, W, C = source_image_bald.shape

        # generate images
        set_scale(self.pipeline.unet, scale)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(random_seed)
        sample = self.pipeline(
            prompt,
            negative_prompt=n_prompt,
            num_inference_steps=step,
            guidance_scale=guidance_scale,
            width=W,
            height=H,
            controlnet_condition=source_image_bald,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            reference_encoder=self.hair_encoder,
            ref_image=reference_image,
        ).samples
        return id, sample, source_image_bald, reference_image

    def get_bald(self, id_image, scale):
        H, W = id_image.size
        scale = float(scale)
        image = self.remove_hair_pipeline(
            prompt="",
            negative_prompt="",
            num_inference_steps=30,
            guidance_scale=1.5,
            width=W,
            height=H,
            image=id_image,
            controlnet_conditioning_scale=scale,
            generator=None,
        ).images[0]

        return image


def main(args):
    """Main function for batch processing hair transfer."""
    # Initialize model
    model = StableHair(config=args.config, weight_dtype=torch.float16)
    
    # Load pairs from CSV file
    data_dir = Path(args.data_dir)
    pairs_csv_path = data_dir / 'pairs.csv'
    if not pairs_csv_path.exists():
        raise FileNotFoundError(f"pairs.csv not found in {data_dir}")
    
    df = pd.read_csv(pairs_csv_path)
    
    # Validate required columns
    if 'source_id' not in df.columns or 'target_id' not in df.columns:
        raise ValueError("pairs.csv must contain 'source_id' and 'target_id' columns")
    
    # Shuffle the pairs for consistent ordering with other methods
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Setup paths
    image_dir = data_dir / 'aligned_image'
    if not image_dir.exists():
        image_dir = data_dir / 'image'
        if not image_dir.exists():
            raise FileNotFoundError(f"Neither 'aligned_image' nor 'image' folder found in {data_dir}")
    
    # Inference parameters
    random_seed = args.seed
    step = args.steps
    guidance_scale = args.guidance_scale
    scale = args.scale
    controlnet_conditioning_scale = args.controlnet_scale
    size = args.size
    
    # Process each pair
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing pairs"):
        source_id = str(row['source_id'])
        target_id = str(row['target_id'])
        
        source_path = None
        target_path = None
        
        for ext in ['.png', '.jpeg', '.jpg', '.webp']:
            if source_path is None:
                candidate = image_dir / f'{source_id}{ext}'
                if candidate.exists():
                    source_path = candidate
            
            if target_path is None:
                candidate = image_dir / f'{target_id}{ext}'
                if candidate.exists():
                    target_path = candidate
            
            if source_path and target_path:
                break

        # Set to default if not found (will be caught by exists check below)
        if source_path is None:
            source_path = image_dir / f'{source_id}.png'
        if target_path is None:
            target_path = image_dir / f'{target_id}.png'
        
        # Check if images exist
        if not source_path.exists():
            print(f"Warning: Source image not found: {source_path}")
            continue
        if not target_path.exists():
            print(f"Warning: Target image not found: {target_path}")
            continue
        
        # Output directory: {data_dir}/baselines/stablehair/{target_id}_to_{source_id}/
        sample_id = f'{target_id}_to_{source_id}'
        output_dir = data_dir / 'baselines' / 'stablehair' / sample_id
        
        # Output paths
        transferred_path = output_dir / 'transferred.png'
        bald_path = output_dir / 'source_bald.png'
        
        # Skip if already processed
        if transferred_path.exists() and bald_path.exists():
            continue
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load images
            source_image = Image.open(source_path).convert("RGB").resize((size, size))
            reference_image = np.array(Image.open(target_path).convert("RGB").resize((size, size)))
            
            # Get bald version of source
            source_image_bald = np.array(model.get_bald(source_image, scale=0.9))
            H, W, C = source_image_bald.shape
            
            # Generate transferred image
            set_scale(model.pipeline.unet, scale)
            generator = torch.Generator(device="cuda")
            generator.manual_seed(random_seed)
            
            sample = model.pipeline(
                "",  # prompt
                negative_prompt="",
                num_inference_steps=step,
                guidance_scale=guidance_scale,
                width=W,
                height=H,
                controlnet_condition=source_image_bald,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
                reference_encoder=model.hair_encoder,
                ref_image=reference_image,
            ).samples
            
            # Save outputs
            # Save bald image
            Image.fromarray(source_image_bald).save(bald_path)
            
            # Save transferred image (sample is numpy array in 0-1 range)
            transferred_img = Image.fromarray((sample * 255.).astype(np.uint8))
            transferred_img.save(transferred_path)
            
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            continue
    
    print("Processing complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable-Hair batch inference')
    
    # Data directory argument
    parser.add_argument('--data_dir', type=str, default="/workspace/celeba_subset",
                        help='Directory containing pairs.csv and image/ folder')
    parser.add_argument('--config', type=str, default="./configs/hair_transfer.yaml",
                        help='Path to config file')
    
    # Inference parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--steps', type=int, default=30, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=2.0, help='Guidance scale')
    parser.add_argument('--scale', type=float, default=1.0, help='Adapter scale')
    parser.add_argument('--controlnet_scale', type=float, default=1.0, help='ControlNet conditioning scale')
    parser.add_argument('--size', type=int, default=512, help='Image size')
    
    args = parser.parse_args()
    main(args)
