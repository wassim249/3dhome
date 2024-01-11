from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch
import argparse


prj_path = "/content/drive/MyDrive/3Dhome/3dhome"
model = "stabilityai/stable-diffusion-xl-base-1.0"

if pipe is None:   
    pipe = DiffusionPipeline.from_pretrained(
        model,
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    pipe.load_lora_weights(prj_path, weight_name="pytorch_lora_weights.safetensors")
    
parser = argparse.ArgumentParser(description='Generate image from text.')
parser.add_argument('text', type=str, help='Text prompt for image generation')

args = parser.parse_args()

image = pipe(prompt=text).images[0]
image.save("generated.png")
    

    
