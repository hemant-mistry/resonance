import io
import base64
import time
import torch

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)
# --------------------------------------

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from diffusers import AutoPipelineForText2Image, AutoencoderTiny
from transformers import pipeline

app = FastAPI(title="Resonance Neural Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Hardware Setup ---
is_gpu = torch.cuda.is_available()
device = "cuda" if is_gpu else "cpu"
dtype = torch.float16 if is_gpu else torch.float32

if is_gpu:
    print(f"Booting Server on: {torch.cuda.get_device_name(0)}")
else:
    print("Booting Server on: CPU")

nlp_agent = pipeline(
    "text-classification", 
    model="SamLowe/roberta-base-go_emotions", 
    device=0 if is_gpu else -1
)

print("Loading Tiny VAE (TAESD)...")
vae = AutoencoderTiny.from_pretrained(
    "madebyollin/taesdxl", 
    torch_dtype=dtype
)

print("Assembling Pipeline (FP16)...")
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    vae=vae,
    torch_dtype=dtype,
    variant="fp16" if is_gpu else None
)

pipe.set_progress_bar_config(disable=True)

if is_gpu:
    pipe.enable_model_cpu_offload() 
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xFormers Attention enabled.")
    except Exception as e:
        pipe.enable_attention_slicing()
        print("Attention Slicing enabled.")

print("Models Loaded Successfully!")

def generate_image_base64(prompt: str, latents: torch.Tensor) -> str:
    image = pipe(
        prompt=prompt, 
        num_inference_steps=1, 
        guidance_scale=0.0,
        latents=latents
    ).images[0]
    
    buffered = io.BytesIO()
    image.save(buffered, format="WEBP", quality=80)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return f"data:image/webp;base64,{img_str}"

if is_gpu:
    print("Warming up the GPU pipeline...")
    # Because of offloading, we need to pass the target device for latents
    warmup_latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=dtype)
    generate_image_base64("warmup", warmup_latents)
    print("GPU is hot and ready!")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("\n[+] NEURAL LINK ESTABLISHED WITH FRONTEND")
    
    generator = torch.Generator(device="cuda" if is_gpu else "cpu").manual_seed(42)
    session_latents = torch.randn(
        (1, 4, 64, 64),
        generator=generator,
        device="cuda" if is_gpu else "cpu",
        dtype=dtype
    )
    
    try:
        # --- THE EMOTION STYLE DICTIONARY ---
        EMOTION_STYLES = {
            "SADNESS": "dull muted colors, grayscale and blue tones, melancholic, dark and somber, lonely abstract human figure silhouette",
            "JOY": "vibrant bright colors, luminous, uplifting, warm golden lighting, energetic abstract human figure",
            "ANGER": "harsh red and black tones, aggressive sharp brushstrokes, chaotic storm, tense abstract human figure",
            "FEAR": "eerie shadows, cold blue lighting, unsettling atmosphere, isolated abstract human figure",
            "CONFUSION": "surreal overlapping shapes, foggy hazy atmosphere, wandering abstract human figure",
            "NEUTRAL": "balanced lighting, calm abstract human figure, soft impressionist tones"
        }

        while True:
            user_text = await websocket.receive_text()
            print(f"\n--- New Frame ---")
            
            t0 = time.time()
            emotion_result = nlp_agent(user_text)[0]
            emotion = emotion_result['label'].upper()
            t1 = time.time()
            print(f"NLP Extracted [{emotion}] in {(t1-t0)*1000:.0f} ms")
            style_modifiers = EMOTION_STYLES.get(emotion, EMOTION_STYLES["NEUTRAL"])
            
            full_prompt = f"masterpiece, thick impasto oil painting, {style_modifiers}. The scene represents: {user_text}. clean canvas, no signatures, no text"
            
            image_str = generate_image_base64(full_prompt, session_latents)
            t2 = time.time()
            print(f"SDXL Generated Image in {(t2-t1)*1000:.0f} ms")
            
            await websocket.send_json({
                "emotion": emotion,
                "confidence": f"{emotion_result['score']*100:.0f}%",
                "image_data": image_str
            })

    except WebSocketDisconnect:
        print("\n[-] NEURAL LINK SEVERED")