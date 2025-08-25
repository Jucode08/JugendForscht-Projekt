# change: VAD

from faster_whisper import WhisperModel  # Speech-to-Text
import asyncio
import edge_tts  # Text-to-Speech
from deep_translator import GoogleTranslator
import os
import threading

from utils import preprocess_ffmpeg
from vad import split_into_speech_segments

# imports for testing
import sounddevice as sd
from playsound import playsound
import time as t


recordDuration = 2.5
samplerate = 16000  # 16 kHz

known_languages = ["de", "en"]
model = WhisperModel("tiny", device="cpu", compute_type="int8") 


translation_support = ['ar', 'de', 'en', 'es', 'fa', 'fr', 'hi', 'id', 'it', 'ja', 'kn', 'ko', 'mr', 'pl', 'pt', 'ru', 'sw', 'ta', 'th', 'tr', 'uk', 'ur', 'vi', 'zh-CN', 'zh-TW']

selected_language = "fr"  # None := Sprache erkennen

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

    return recording


def translate(language, text):
    if language not in known_languages:
        translated = GoogleTranslator(
            source=language, target=known_languages[0]
        ).translate(text)
        return translated
    else:
        return None


def textToSpeech(text, voice=selected_voice, rate="+10%"):
    async def inner():
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save("output.mp3")

    asyncio.run(inner())


def play_and_delete(path):
    playsound(path)
    os.remove(path)


def process_audio(recording):

    # 1  Preprocess  16 kHz Mono float32
    audio = preprocess_ffmpeg(recording, samplerate)

    # 2  WebRTC-VAD  nur echte Sprachsegmente durchlassen
    voiced_segments = split_into_speech_segments(
        audio,
        sample_rate=16000,
        frame_ms=30,       # 10/20/30 erlaubt  30 ms = weniger Overhead
        padding_ms=300,    # Hangover  verhindert zu hartes Abschneiden
        aggressiveness=2   # 0..3  2 ist guter Start
    )

    if not voiced_segments:
        return
    
    text = ""

    for seg in voiced_segments:
        segments, info = model.transcribe(seg, language=selected_language, beam_size=1)
        text += " ".join(s.text for s in segments).strip()

    # Transkription
    # segments, info = model.transcribe(audio, language=selected_language, beam_size=1)
    # text = " ".join([s.text for s in segments]).strip()

    print(f"erkannter Text: {text}")
    
    language = info.language
    
    translated = translate(language, text)
    if not translated:
        return
    print(f"übersetzter Text: {translated}")
    
    textToSpeech(translated)
    
    threading.Thread(target=play_and_delete, args=("output.mp3",), daemon=True).start()

while True:
    recording = record()
    
    threading.Thread(target=process_audio, args=(recording,), daemon=True).start()

    t.sleep(4)