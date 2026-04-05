from typing import List
import modal
import os
import uuid
import base64
from pydantic import BaseModel
import requests
import boto3

from backend.prompts import LYRICS_GENERATOR_PROMPT, PROMPT_GENERATOR_PROMPT

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

model_volume = modal.Volume.from_name(
    "ace-step-models", create_if_missing=True)
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


class GenerateMusicResponseS3(BaseModel):
    s3_key: str
    cover_image_s3_key: str
    categories: List[str]


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
            src.squeeze().cpu().float().numpy().T if src.dim(
            ) > 1 else src.squeeze().cpu().float().numpy(),
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

    def prompt_qwen(self, question: str):
        messages = [
            {"role": "user", "content": question}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer(
            [text], return_tensors="pt").to(self.llm_model.device)

        generated_ids = self.llm_model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        return response

    def generate_prompt(self, description: str):
        # Insert into a template
        full_prompt = PROMPT_GENERATOR_PROMPT.format(user_prompt=description)

        # run LLM inference and return it
        return self.prompt_qwen(full_prompt)

    def generate_lyrics(self, description: str):
        # Insert into a template
        full_prompt = LYRICS_GENERATOR_PROMPT.format(description=description)

        # run LLM inference and return it
        return self.prompt_qwen(full_prompt)
    
    def generateCategories(self, description:str)->List[str]:
        prompt = f"Based on the following music description, list 3-5 relevant genres or categories as a comma-separated list. For example: Pop, Electronic, Sad, 80s. Description: '{description}'"

        response_text = self.prompt_qwen(prompt)
        categories = [cat.strip() 
                      for cat in response_text.split(",") if cat.strip()]
        
        return categories

    def generate_and_upload_to_S3(
        self,
        prompt: str,
        lyrics: str,
        instrumental: bool,
        audio_duration: float,
        infer_step: int,
        guidance_scale: float,
        seed: int,
        description_for_categorization: str
    ) -> GenerateMusicResponseS3:
        final_lyrics = "[instrumental]" if instrumental else lyrics
        print(f"Generated lyrics: \n{final_lyrics}")
        print(f"Prompt: \n{prompt}")

        #-----AWS-----

        #S3Bucket: Songs Generation

        s3_client = boto3.client("s3")
        bucket_name = os.environ("S3_BUCKET_NAME")

        output_dir = "/tmp/outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")

        self.music_model(
            prompt=prompt,
            lyrics=final_lyrics,
            audio_duration=audio_duration,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            save_path=output_path
            manual_seed=str(seed)
        )

        audio_s3_key = f"{uuid.uuid4()}.wav"
        s3_client.upload_file(output_path, bucket_name, audio_s3_key)
        os.remove(output_path)

        #S3Bucket: Thumbnail Generation

        thumbnail_prompt = f"{prompt}, album cover art"
        image = self.image_pipe(prompt=thumbnail_prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
        image_output_path = os.path.join(output_dir, f"{uuid.uuid4()}.png")
        image.save(image_output_path)

        image_s3_key = f"{uuid.uuid4()}.png"
        s3_client.upload_file(image_output_path, bucket_name, image_s3_key)
        os.remove(image_output_path)


        #S3Bucket: Categories Generation (eg: hiphop, melody etc)
        categories = self.generateCategories(description_for_categorization)


        return GenerateMusicResponseS3(
            s3_key=audio_s3_key,
            cover_image_s3_key=image_s3_key,
            categories=categories
        )


   

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
    def generate_from_description(self, request: GenerateFromDescriptionRequest) -> GenerateMusicResponseS3:

        # Generating a prompt
        prompt = self.generate_prompt(request.full_described_song)

        # Generate the lyrics
        lyrics = ""
        if not request.instrumental:
            lyrics = self.generate_lyrics(request.full_described_song)

    # Endpoint to generate with lyrics

    @modal.fastapi_endpoint(method="POST")
    def generate_with_lyrics(self, request: GenerateWithCustomLyricsRequest) -> GenerateMusicResponseS3:

        # Endpoint to generate from described lyrics by the user

    @modal.fastapi_endpoint(method="POST")
    def generate__with_described_lyrics(self, request: GenerateFromDescribedLyricsRequest) -> GenerateMusicResponseS3:

        # Generating Lyrics


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
