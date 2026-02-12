from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import edge_tts
import io
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
]

STT_MODEL = "openai/whisper-large-v3-turbo"
TTS_VOICE = "en-US-ChristopherNeural"


class ChatRequest(BaseModel):
    message: str


def transcribe_audio(audio_bytes: bytes) -> str:
    """Speech-to-Text using HF Whisper model."""
    result = hf_client.automatic_speech_recognition(
        audio=audio_bytes,
        model=STT_MODEL,
    )
    return getattr(result, "text", "")


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


async def text_to_speech(text: str) -> bytes:
    """Convert text to MP3 audio using Edge TTS."""
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    audio_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])
    audio_buffer.seek(0)
    return audio_buffer.getvalue()


@app.post("/voice")
async def voice(audio: UploadFile = File(...)):
    """Full pipeline: Audio in → STT → LLM → TTS → Audio out"""
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

        # Step 3: Text-to-Speech
        audio_response = await text_to_speech(reply)

        return StreamingResponse(
            io.BytesIO(audio_response),
            media_type="audio/mpeg",
            headers={
                "X-Transcript": transcript,
                "X-Reply": reply[:200],
            }
        )
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


@app.post("/tts")
async def tts(req: ChatRequest):
    """Text in → Audio out"""
    try:
        audio = await text_to_speech(req.message)
        return StreamingResponse(
            io.BytesIO(audio),
            media_type="audio/mpeg",
        )
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
