import assemblyai as aai
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())

aai.settings.api_key = os.environ['ASSEMBLYAI_API_KEY']

config = aai.TranscriptionConfig(
    speaker_labels=True,
)

transcriber = aai.Transcriber()

def transcribe(file_path):
    transcript = transcriber.transcribe(file_path, config=config)

    result = []
    for utterance in transcript.utterances:
        result.append({"speaker": utterance.speaker, "text": utterance.text, "timestamp": utterance.start/1000})

    return result, transcript.text   
