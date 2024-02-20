import os
import assemblyai as aai
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
load_dotenv()

# apikeys are saved in .env file
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

# file can also be saved locally
audio_file = "https://imxze2im7tagxmrw.public.blob.vercel-storage.com/SH_121-O3Uyv8WUlNlfVA36YPUIuzzESKKgDq.m4a"
config = aai.TranscriptionConfig(language_code="hi")
transcriber = aai.Transcriber(config=config)
transcript = transcriber.transcribe(audio_file, config=config)

# this will store the transcribed text
x = transcript.text
print(x)

# this is for calling the gemini model
llm = GoogleGenerativeAI(model="gemini-pro")
print(llm.invoke(x))
