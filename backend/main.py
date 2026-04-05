import modal
import os
import uuid
import base64
from pydantic import BaseModel
import requests

app = modal.App("music-generator")

image = (
    modal.Image.debian_slim()
    .apt_install("git", "ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    .run_commands([
        "git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step",
        "cd /tmp/ACE-Step && pip install ."
    ])
    .env({"HF_HOME": "/.cache/huggingface"})
    .add_local_python_source("prompts")
)

model_volume = modal.Volume.from_name("ace-step-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("qw-hf-cache", create_if_missing=True)
music_gen_secrets = modal.Secret.from_name("music-gen-secrets")

class AudioGenerateBase(BaseModel):
    audio_duration: float = 120.0
    seed: int = -1
    guidance_scale: float = 15.0
    infer_step: int = 60
    instrumental: bool = False

class GenerateFromDescriptionRequest(AudioGenerateBase):
    full_described_song: str

class GenerateWithCustomLyricsRequest(AudioGenerateBase):
    prompt: str
    lyrics: str

class GenerateFromDescribedLyricsRequest(AudioGenerateBase):
    prompt: str
    described_lyrics: str


class GenerateMusicResponse(BaseModel):
    audio_data: str

@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/models": model_volume, "/.cache/huggingface": hf_volume},
    secrets=[music_gen_secrets],
    scaledown_window=15
)
class MusicGenServer:
    @modal.enter()
    def load_model(self):
        import torch
        import soundfile as sf
        import torchaudio
        torchaudio.save = lambda path, src, sample_rate, **kwargs: sf.write(
            path, 
            src.squeeze().cpu().float().numpy().T if src.dim() > 1 else src.squeeze().cpu().float().numpy(), 
            sample_rate
        )

        from acestep.pipeline_ace_step import ACEStepPipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from diffusers import StableDiffusionXLPipeline

        self.music_model = ACEStepPipeline(
            checkpoint_dir="/models",
            dtype="bfloat16",
            torch_compile=False,
            cpu_offload=False,
            overlapped_decode=False
        )

        model_id = "Qwen/Qwen2-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir="/.cache/huggingface"
        )

        self.image_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir="/.cache/huggingface"
        ).to("cuda")

    @modal.fastapi_endpoint(method="POST")
    def generate(self) -> GenerateMusicResponse:
        output_dir = "/tmp/outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")

        self.music_model(
            prompt="electronic rap",
            lyrics="[verse]\nWaves on the bass, pulsing in the speakers,\nTurn the dial up, we chasing six-figure features,\nGrinding on the beats, codes in the creases,\nDigital hustler, midnight in sneakers.\n\n[chorus]\nElectro vibes, hearts beat with the hum,\nUrban legends ride, we ain't ever numb,\nCircuits sparking live, tapping on the drum,\nLiving on the edge, never succumb.",
            audio_duration=180,
            infer_step=60,
            guidance_scale=15,
            save_path=output_path
        )

        with open(output_path, "rb") as f:
            audio_bytes = f.read()

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        os.remove(output_path)

        return GenerateMusicResponse(audio_data=audio_b64)

    
    # Endpoint to generate from description
    
    @modal.fastapi_endpoint(method="POST")
    def generate_from_description(self) -> GenerateMusicResponse:
        
    

    # Endpoint to generate with lyrics 

    @modal.fastapi_endpoint(method="POST")
    def generate_with_lyrics(self) -> GenerateMusicResponse:

    
    # Endpoint to generate from described lyrics by the user

    @modal.fastapi_endpoint(method="POST")
    def generate__with_described_lyrics(self) -> GenerateMusicResponse:

    


@app.local_entrypoint()
def main():
    server = MusicGenServer()
    endpoint_url = server.generate.get_web_url()

    response = requests.post(endpoint_url)
    response.raise_for_status()
    result = GenerateMusicResponse(**response.json())

    audio_bytes = base64.b64decode(result.audio_data)
    with open("generated.wav", "wb") as f:
        f.write(audio_bytes)