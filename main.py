from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_API_KEY")
if not HF_TOKEN:
    print("WARNING: HF_API_KEY not found in .env")

hf_client = InferenceClient(api_key=HF_TOKEN) if HF_TOKEN else None

app = FastAPI(title="Voice Assistant Cloud Backend")

MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen2.5-72B-Instruct",
]

STT_MODEL = "openai/whisper-large-v3-turbo"


class ChatRequest(BaseModel):
    message: str


def transcribe_audio(audio_bytes: bytes) -> str:
    """Speech-to-Text using HF Whisper model."""
    result = hf_client.automatic_speech_recognition(
        audio=audio_bytes,
        model=STT_MODEL,
    )
    return result.text


def generate_response(text: str) -> str:
    """Send text to LLM and get a reply."""
    messages = [
        {"role": "system", "content": "You are a helpful voice assistant. Keep your response concise, natural, and conversational. Do not use markdown."},
        {"role": "user", "content": text},
    ]

    for model in MODELS:
        try:
            response = hf_client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=256,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                continue
            elif "503" in error_str or "loading" in error_str.lower():
                continue
            else:
                raise

    raise Exception("All models are currently unavailable.")


@app.post("/voice")
async def voice(audio: UploadFile = File(...)):
    """Full pipeline: Audio in → STT → LLM → Text out"""
    if not hf_client:
        return {"error": "HF_API_KEY not configured"}

    try:
        # Step 1: Speech-to-Text
        audio_bytes = await audio.read()
        transcript = transcribe_audio(audio_bytes)
        print(f"STT: {transcript}")

        # Step 2: LLM response
        reply = generate_response(transcript)
        print(f"LLM: {reply}")

        return {"transcript": transcript, "response": reply}
    except Exception as e:
        print(f"Error in /voice: {e}")
        return {"error": str(e)}


@app.post("/chat")
def chat(req: ChatRequest):
    """Text in → LLM → Text out"""
    if not hf_client:
        return {"error": "HF_API_KEY not configured"}

    try:
        reply = generate_response(req.message)
        return {"response": reply}
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
