from pathlib import Path
import openai
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Choose output file
speech_file_path = Path("speech.mp3")

# Create speech from text
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",   # TTS model
    voice="cedar",             # available voices: marin, cedar
    input="Hello! This is a test of OpenAI text to speech in Python."
) as response:
    response.stream_to_file(speech_file_path)

print(f"Saved speech to {speech_file_path}")