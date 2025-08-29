# change: TTS Queue mit worker thread

from faster_whisper import WhisperModel  # Speech-to-Text
import asyncio
import edge_tts  # Text-to-Speech
from deep_translator import GoogleTranslator
import os
import threading
import uuid
import queue


from utils import preprocess_ffmpeg

# imports for testing
import sounddevice as sd
from playsound import playsound
import time as t

# global variables
model = WhisperModel("tiny", device="cpu", compute_type="int8") 

recordDuration = 2.5
samplerate = 16000  # 16 kHz

known_languages = ["en"]
translation_support = ['ar', 'de', 'en', 'es', 'fa', 'fr', 'hi', 'id', 'it', 'ja', 'kn', 'ko', 'mr', 'pl', 'pt', 'ru', 'sw', 'ta', 'th', 'tr', 'uk', 'ur', 'vi', 'zh-CN', 'zh-TW']
selected_language = "de"  # None := Sprache erkennen

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

tts_queue = queue.Queue()


def record():
    print("recording...")
    recording = sd.rec(
        int(recordDuration * samplerate), samplerate=samplerate, channels=1
    )  # channels=1 => mono , channels=2 => stereo
    sd.wait()

    return recording


def translate(language, text):
    if language not in known_languages:
        translated = GoogleTranslator(
            source=language, target=known_languages[0]
        ).translate(text)
        return translated
    else:
        return None


tts_queue = queue.Queue()

def tts_worker(voice=selected_voice, rate="+10%"):
    while True:
        text, filename = tts_queue.get()
        async def inner():
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(filename)
        asyncio.run(inner())
        playsound(filename)
        os.remove(filename)
        tts_queue.task_done()


def process_audio(recording):

    # Preprocess  16 kHz Mono float32
    audio = preprocess_ffmpeg(recording, samplerate)


    # Transkription
    segments, info = model.transcribe(audio, language=selected_language, beam_size=1)
    text = " ".join([s.text for s in segments]).strip()
    
    if text is None: return
    else: print(f"erkannter Text: {text}")
    
    language = info.language
    
    translated = translate(language, text)
    if not translated:
        return
    print(f"übersetzter Text: {translated}")
    
    filename = uuid.uuid4().hex + ".mp3"
    
    tts_queue.put((translated, filename))
    
    
    
threading.Thread(target=tts_worker, daemon=True).start()

while True:
    recording = record()
    
    try:
        threading.Thread(target=process_audio, args=(recording,), daemon=True).start()
    except Exception as e:
        print("Error: ", e)
        import sys
        sys.exit()
        
