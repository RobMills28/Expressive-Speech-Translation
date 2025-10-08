# 🗣️ Expressive Speech Translation

**A proof-of-concept for a multimodal, identity-preserved speech translation pipeline.** This system translates spoken content from video or audio files into a new language whilst preserving the original speaker's unique vocal characteristics, emotional tone, and natural timing.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18+-61DAFB?style=flat&logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![AI](https://img.shields.io/badge/AI-Speech%20Translation-FF6B6B?style=flat)

## ✨ Core Features

🎭 **Vocal Identity Preservation** — Utilises zero-shot voice cloning to ensure the translated speech sounds like the original speaker

😊 **Emotional Congruence** — Maintains the emotional prosody (tone, pitch, rhythm) of the original speech in the translated output

🎬 **Audio-Visual Coherence** — Generates accurate lip-synchronisation for video content, matching the new translated audio track

🌍 **Multilingual Support** — Built on a powerful NMT model capable of translating between hundreds of languages

🏗️ **Modular Architecture** — Designed as a flexible and scalable system of independent microservices

## 🏛️ Technical Architecture

This project is built using a **Modern Cascaded Framework (MCF)**, a modular microservice architecture designed for robustness and scalability. A central **Orchestrator Service** (built with Flask) manages the workflow and communicates with a set of independent, containerised AI services.

### The pipeline consists of four primary AI models:

#### 🎤 1. Automatic Speech Recognition (ASR)
- **Model:** OpenAI Whisper
- **Purpose:** Transcribes the source audio into text with high accuracy

#### 🔄 2. Neural Machine Translation (NMT)
- **Model:** Meta NLLB (No Language Left Behind)
- **Purpose:** Translates the source text into the target language

#### 🗣️ 3. Expressive Text-to-Speech (TTS)
- **Model:** CosyVoice
- **Purpose:** Synthesises the translated text into speech. It uses the original source audio as a voice prompt to perform zero-shot voice cloning, preserving the speaker's unique vocal timbre

#### 👄 4. Lip-Synchronisation
- **Model:** MuseTalk
- **Purpose:** Takes the original video (without audio) and the newly generated translated audio, and produces a final video with perfectly synchronised lip movements

## 🔄 How It Works (High-Level Process Flow)

1. 📤 A user uploads a source video or audio file via the React frontend
2. 🎛️ The Flask **Orchestrator** receives the file and extracts the audio
3. 🎤 The audio is sent to the **ASR Service (Whisper)** to be transcribed
4. 🔄 The resulting source text is sent to the **NMT Service (NLLB)** to be translated
5. 🗣️ The translated text, along with the original source audio (as a voice reference), is sent to the **Expressive TTS Service (CosyVoice)**. This service generates the final translated audio in the original speaker's voice
6. 🎬 For video, the original video frames and the new translated audio are sent to the **Lip-Sync Service (MuseTalk)**, which generates the final, perfectly synchronised video

## 🛠️ Tech Stack

**Backend:**
- 🐍 Python
- ⚡ Flask
- 🐳 Docker
- 🦄 Gunicorn

**Frontend:**
- ⚛️ React
- 📝 JavaScript
- 🎨 Tailwind CSS
- 🎭 Shadcn/UI
- 🌊 Wavesurfer.js

**AI Models:**
- 🔥 PyTorch
- 🤗 Transformers
- 🎤 Whisper
- 🌍 NLLB
- 🗣️ CosyVoice
- 👄 MuseTalk

**DevOps/Deployment:**
- 🐳 Docker Compose for local orchestration
- ☁️ Designed for cloud infrastructure (AWS) or local HPC cluster with Slurm

## 🚀 Running the Project

This project is composed of a frontend application and several backend microservices.

### 🔧 Backend Setup

1. Navigate to the Backend directory
2. Create a Python virtual environment (e.g., with Conda) and install the dependencies from `requirements.txt`
3. The TTS and Lip-Sync models are run as separate Docker containers. Navigate to the Docker directory and use docker-compose to build and run the services:

```bash
docker-compose up -d cosyvoice musetalk
```

4. Start the main Flask application:

```bash
python app.py
```

### 🎨 Frontend Setup

1. Navigate to the Frontend directory
2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm start
```

4. Open `http://localhost:3000` in your web browser

## 📁 Project Structure

```
expressive-speech-translation/
├── Backend/               # Flask orchestrator and API endpoints
├── Frontend/              # React user interface
├── Docker/                # Container configurations for AI services
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── docker-compose.yml    # Service orchestration
```

## 🤝 Contributing

This is a research prototype and proof-of-concept. For academic or research collaboration enquiries, please open an issue or contact the maintainer.

## 📄 Licence

This project is currently under development. Please contact the author for usage permissions and licensing enquiries.

## 👨‍💻 Author

**Robert Mills** - [RobMills28](https://github.com/RobMills28)

---

*An advanced speech translation system demonstrating the potential for preserving human expressiveness in cross-lingual communication.*
