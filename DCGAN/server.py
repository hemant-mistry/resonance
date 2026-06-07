import io
import base64
import json
import time
import hashlib
import torch
from google import genai
from google.genai import types
from dotenv import load_dotenv

if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from diffusers import AutoPipelineForText2Image, AutoencoderTiny

app = FastAPI(title="Resonance Neural Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

gemini_client = genai.Client()

# --- Hardware Setup ---
is_gpu = torch.cuda.is_available()
device = "cuda" if is_gpu else "cpu"
dtype = torch.float16 if is_gpu else torch.float32

if is_gpu:
    print(f"Booting Server on: {torch.cuda.get_device_name(0)}")
else:
    print("Booting Server on: CPU")

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
    except Exception:
        pipe.enable_attention_slicing()
        print("Attention Slicing enabled.")

print("Models Loaded Successfully!")


def text_to_seed(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)


def make_latents(seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        (1, 4, 64, 64),
        generator=generator,
        device=device,
        dtype=dtype
    )


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
    warmup_latents = make_latents(0)
    generate_image_base64("warmup", warmup_latents)
    print("GPU is hot and ready!")


SYSTEM_INSTRUCTION = """
You are an expert art director. Analyze the user's journal entry and return a JSON object.

STRICT RULES:
1. Return ONLY a raw JSON object. No markdown, no code fences, no explanation, nothing else.
2. The JSON must have exactly two keys: "emotion" and "prompt".
3. "emotion": ONE uppercase word (e.g. "JOY", "GRIEF", "RAGE", "ANXIETY", "ECSTASY").
4. "prompt": a Stable Diffusion prompt that physically depicts the scene and matches the emotion.
   Keep it under 60 words. Format: "masterpiece, thick impasto oil painting, [SCENE], [COLORS], [LIGHTING]"

Example:
Input: "I feel so alone watching the rain"
Output: {"emotion":"LONELINESS","prompt":"masterpiece, thick impasto oil painting, solitary figure at rain-streaked window, dim room, muted grays and cold blues, soft diffused rainy light, melancholic atmosphere"}
"""


def extract_json_from_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract valid JSON from: {repr(text[:200])}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("\n[+] NEURAL LINK ESTABLISHED WITH FRONTEND")

    try:
        while True:
            user_text = await websocket.receive_text()
            print(f"\n--- New Frame ---")
            print(f"User text: {user_text[:80]}...")

            t0 = time.time()
            detected_emotion = "UNKNOWN"
            dynamic_sdxl_prompt = None

            try:
                response = await gemini_client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_text,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.8,
                        response_mime_type="application/json",
                        max_output_tokens=2048,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    )
                )

                finish_reason = ""
                if response.candidates:
                    finish_reason = str(response.candidates[0].finish_reason)
                    print(f"Finish reason: {finish_reason}")

                if "SAFETY" in finish_reason or "BLOCK" in finish_reason:
                    raise ValueError(f"Safety blocked: {finish_reason}")
                if "MAX_TOKENS" in finish_reason:
                    raise ValueError("Response truncated by MAX_TOKENS — increase max_output_tokens")

                raw_text = response.text
                print(f"--- RAW GEMINI OUTPUT ---\n{raw_text}\n---")

                ai_data = extract_json_from_response(raw_text)
                detected_emotion = ai_data.get("emotion", "UNKNOWN").upper()
                dynamic_sdxl_prompt = ai_data.get("prompt", "").strip()

                if not dynamic_sdxl_prompt:
                    raise ValueError("Empty prompt returned")

            except Exception as e:
                print(f"Gemini error: {e}. Using fallback.")
                detected_emotion = "UNKNOWN"
                dynamic_sdxl_prompt = None

            t1 = time.time()
            print(f"Emotion: [{detected_emotion}]")
            print(f"Prompt:  {dynamic_sdxl_prompt}")
            print(f"Gemini time: {(t1 - t0) * 1000:.0f} ms")

            if dynamic_sdxl_prompt:
                seed = text_to_seed(dynamic_sdxl_prompt)
            else:
                seed = text_to_seed(user_text + str(int(time.time())))
                dynamic_sdxl_prompt = (
                    f"masterpiece, thick impasto oil painting, "
                    f"surreal emotional dreamscape representing: {user_text[:80]}, "
                    f"vivid colors, dramatic lighting, textured brushstrokes"
                )

            latents = make_latents(seed)
            image_str = generate_image_base64(dynamic_sdxl_prompt, latents)

            t2 = time.time()
            print(f"SDXL time: {(t2 - t1) * 1000:.0f} ms")

            await websocket.send_json({
                "emotion": detected_emotion,
                "confidence": "JSON-SYNC",
                "image_data": image_str
            })

    except WebSocketDisconnect:
        print("\n[-] NEURAL LINK SEVERED")