<div align="center">

# MuseAI — AI Music Studio

</div>

![MuseAI](/frontend/public/header-img.png)

---

## Overview

An AI-powered music creation platform that generates songs, lyrics, rhythms, covers, and artwork using multiple specialized models running on serverless GPUs via Modal. Infrastructure is secured with AWS IAM, Security Groups, and uses AWS S3 for scalable storage.

## Core Models

### ACE-Step
Primary music generation model used for composing melodies, beats, instrumentals, rhythms, and full song arrangements.

### Qwen2-7B-Instruct
Language model used for lyrics generation, prompt understanding, song structure creation, multilingual text generation, and user interaction.

### Hugging Face Diffusers
Used to generate custom cover art / album images for each song based on prompts, mood, genre, or theme.

## Infrastructure

### Modal Serverless GPU
Runs all AI models on-demand with scalable serverless GPU inference.

### AWS IAM + Security Groups
Provides secure access control, permissions management, and protected networking.

### AWS S3
Stores generated songs, cover images, model assets, and user uploads securely.

---
<div align="center">

# Coming Soon

</div>