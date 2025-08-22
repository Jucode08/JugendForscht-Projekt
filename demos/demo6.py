# change: faster_whisper + ffmpeg + looped

import numpy as np
from faster_whisper import WhisperModel  # Speech-to-Text
import asyncio
import edge_tts  # Text-to-Speech
from deep_translator import GoogleTranslator
import os
import ffmpeg

# imports for testing
import sounddevice as sd
import soundfile as sf
from playsound import playsound
import sys


recordDuration = 3
samplerate = 16000  # 16 kHz

known_languages = ["de", "en"]
model = WhisperModel("tiny", device="cpu", compute_type="int8") 

translation_support = ['ar', 'de', 'en', 'es', 'fa', 'fr', 'hi', 'id', 'it', 'ja', 'kn', 'ko', 'mr', 'pl', 'pt', 'ru', 'sw', 'ta', 'th', 'tr', 'uk', 'ur', 'vi', 'zh-CN', 'zh-TW']

selected_language = "fr"

default_voices = {
    "ar": {"male": "ar-AE-HamdanNeural", "female": "ar-AE-FatimaNeural"},
    "de": {"male": "de-AT-JonasNeural", "female": "de-AT-IngridNeural"},
    "en": {"male": "en-AU-WilliamMultilingualNeural", "female": "en-AU-NatashaNeural"},
    "es": {"male": "es-AR-TomasNeural", "female": "es-AR-ElenaNeural"},
    "fa": {"male": "fa-IR-FaridNeural", "female": "fa-IR-DilaraNeural"},
    "fr": {"male": "fr-BE-GerardNeural", "female": "fr-BE-CharlineNeural"},
    "hi": {"male": "hi-IN-MadhurNeural", "female": "hi-IN-SwaraNeural"},
    "id": {"male": "id-ID-ArdiNeural", "female": "id-ID-GadisNeural"},
    "it": {"male": "it-IT-DiegoNeural", "female": "it-IT-ElsaNeural"},
    "ja": {"male": "ja-JP-KeitaNeural", "female": "ja-JP-NanamiNeural"},
    "kn": {"male": "kn-IN-GaganNeural", "female": "kn-IN-SapnaNeural"},
    "ko": {"male": "ko-KR-InJoonNeural", "female": {}},
    "mr": {"male": "mr-IN-ManoharNeural", "female": "mr-IN-AarohiNeural"},
    "pl": {"male": "pl-PL-MarekNeural", "female": "pl-PL-ZofiaNeural"},
    "pt": {"male": "pt-BR-AntonioNeural", "female": "pt-BR-FranciscaNeural"},
    "ru": {"male": "ru-RU-DmitryNeural", "female": "ru-RU-SvetlanaNeural"},
    "sw": {"male": "sw-KE-RafikiNeural", "female": "sw-KE-ZuriNeural"},
    "ta": {"male": "ta-IN-ValluvarNeural", "female": "ta-IN-PallaviNeural"},
    "th": {"male": "th-TH-NiwatNeural", "female": "th-TH-PremwadeeNeural"},
    "tr": {"male": "tr-TR-AhmetNeural", "female": "tr-TR-EmelNeural"},
    "uk": {"male": "uk-UA-OstapNeural", "female": "uk-UA-PolinaNeural"},
    "ur": {"male": "ur-IN-SalmanNeural", "female": "ur-IN-GulNeural"},
    "vi": {"male": "vi-VN-NamMinhNeural", "female": {}},
    "zh": {"male": "zh-CN-YunjianNeural", "female": "zh-CN-XiaoxiaoNeural"}
}

selected_gender = "male"
selected_voice = default_voices[known_languages[0]][selected_gender]

def record():
    print("recording...")
    recording = sd.rec(
        int(recordDuration * samplerate), samplerate=samplerate, channels=1
    )  # channels=1 => mono , channels=2 => stereo
    sd.wait()

    process_audio(recording)
    # threading.Thread(target=process_audio, args=(recording,), daemon=True).start()

def preprocess_ffmpeg(recording):
    sf.write("temp.wav", recording, samplerate)
    out, _ = (
        ffmpeg
        .input("temp.wav")
        .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar="16000")
        .run(capture_stdout=True, quiet=True)
    )
    audio = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
    return audio

def process_audio(recording):


    # Preprocessing
    
    # Wandelt (N, 1) => (N,) um und castet auf float32
    recording = recording.flatten().astype(np.float32)
    
    recording = preprocess_ffmpeg(recording)
    os.remove("temp.wav")


    # Transkription
    segments, info = model.transcribe(recording, language=selected_language, beam_size=1)
    text = " ".join([s.text for s in segments]).strip()

    print(f"erkannter Text: {text}")
    if text == "..." or text.lower() == "you":
        return

    
    language = info.language
    # info.language_probability
    
    if language not in known_languages:
        translated = GoogleTranslator(
            source=language, target=known_languages[0]
        ).translate(text)

        textToSpeech(translated)
        print(f"übersetzter Text: {translated}")

        play_and_delete("output.mp3")

    # threading.Thread(target=play_and_delete, args=("output.mp3",), daemon=True).start()


def textToSpeech(text, voice=selected_voice, rate="+0%"):
    async def inner():
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save("output.mp3")

    asyncio.run(inner())


def play_and_delete(path):
    playsound(path)
    os.remove(path)


import time as t

inp = input("loop?(y/n): ").lower()
if inp == "y":
    for i in range(5):
        record()
        t.sleep(0.1)
elif inp == "quit":
    print("exiting...")
else:
    record()
