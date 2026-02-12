import asyncio
import os
import subprocess
import speech_recognition as sr
from huggingface_hub import InferenceClient
import edge_tts
import time

# Configure Hugging Face API
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    print("WARNING: HF_API_KEY not found in environment variables.")
    print("Set it with: export HF_API_KEY='your_key_here'")

hf_client = None
if HF_API_KEY:
    hf_client = InferenceClient(api_key=HF_API_KEY)

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.voice = "en-US-ChristopherNeural" # High quality male voice
        # Alternative voices: en-US-AriaNeural, en-GB-SoniaNeural, etc.

    def listen(self):
        """Listens to the microphone and returns the recognized text."""
        with self.mic as source:
            print("\nListening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print("Processing audio...")
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
            except sr.WaitTimeoutError:
                print("No speech detected.")
                return None
            except sr.UnknownValueError:
                print("Could not understand audio.")
                return None
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return None

    async def think(self, text):
        """Sends the text to Hugging Face and returns the response."""
        if not hf_client:
            return "Please configure the Hugging Face API key."
        
        print("Thinking...")
        models_to_try = [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "Qwen/Qwen2.5-72B-Instruct",
        ]
        messages = [
            {"role": "system", "content": "You are a helpful voice assistant. Keep your response concise, natural, and conversational. Do not use markdown like * asterisks."},
            {"role": "user", "content": text}
        ]

        for model_name in models_to_try:
            try:
                response = hf_client.chat_completion(
                    model=model_name,
                    messages=messages,
                    max_tokens=256,
                )
                reply = response.choices[0].message.content
                print(f"AI ({model_name}): {reply}")
                return reply
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower():
                    print(f"Rate limited on {model_name}, trying next model...")
                    continue
                elif "503" in error_str or "loading" in error_str.lower():
                    print(f"{model_name} is loading, trying next model...")
                    continue
                else:
                    print(f"Error with {model_name}: {e}")
                    continue
        
        return "Sorry, no models are available right now. Please try again later."

    async def speak(self, text):
        """Converts text to speech and plays it."""
        print(f"Speaking: {text}")
        output_file = "response.mp3"
        
        # Determine voice (Edge TTS)
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_file)

        # Play audio using macOS afplay (no pygame dependency needed)
        try:
            subprocess.run(["afplay", output_file], check=True)
        except FileNotFoundError:
            print("Error: 'afplay' not found. Install ffmpeg and use 'ffplay' as alternative.")
        except Exception as e:
            print(f"Error playing audio: {e}")

    async def run(self):
        """Main interaction loop."""
        print("Voice Assistant Started. Press Ctrl+C to exit.")
        if not hf_client:
             print("CRITICAL: HF_API_KEY is not set. The assistant will not be able to respond intelligently.")
        
        while True:
            user_input = self.listen()
            if user_input:
                if "exit" in user_input.lower() or "quit" in user_input.lower():
                    print("Goodbye!")
                    await self.speak("Goodbye!")
                    break
                
                response_text = await self.think(user_input)
                await self.speak(response_text)
            
            # Small delay to prevent tight loop if recognition fails repeatedly
            await asyncio.sleep(0.5)

if __name__ == "__main__":
    assistant = VoiceAssistant()
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        print("\nAssistant stopped.")
