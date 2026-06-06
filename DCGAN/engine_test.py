import torch
import time
# --- Transformers FP8 Bug Fix Patch ---
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)
# --------------------------------------

from diffusers import AutoPipelineForText2Image

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print(f"Booting Engine on: {device.upper()}")
print("Downloading/Loading SDXL Turbo... (This will take a few minutes the first time)")

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=dtype,
    variant="fp16" if device != "cpu" else None
)
pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=True)

generator = torch.Generator(device=device).manual_seed(42)
fixed_latents = torch.randn(
    (1,4,64,64),
    generator=generator,
    device=device,
    dtype=dtype
)

# Let's simulate a progressive, additive thought process:
prompts = [
    "masterpiece, thick impasto oil painting, a serene sunny beach with bright white sand and clear blue ocean",
    "masterpiece, thick impasto oil painting, a serene sunny beach with bright white sand and clear blue ocean, a tall green coconut tree leaning on the left side",
    "masterpiece, thick impasto oil painting, a serene sunny beach with bright white sand and clear blue ocean, a tall green coconut tree leaning on the left side, a tall wooden watchtower on the right side"
]

print("\nStarting Progressive Morph Sequence...")

for i, prompt in enumerate(prompts):
    # Extract the additive part of the prompt for the print statement
    theme = prompt.split(',')[-1].strip()
    print(f"Frame {i+1}: Adding [{theme}]...")
    
    start_time = time.time()
    
    image = pipe(
        prompt=prompt, 
        num_inference_steps=1, 
        guidance_scale=0.0,
        latents=fixed_latents # Still using the exact same canvas!
    ).images[0]
    
    end_time = time.time()
    
    generation_time_ms = (end_time - start_time) * 1000
    print(f"  -> Finished in {generation_time_ms:.0f} ms")
    
    image.save(f"progressive_frame_{i+1}.png")

print("\nDone! Open progressive_frame_1.png, 2, and 3.")