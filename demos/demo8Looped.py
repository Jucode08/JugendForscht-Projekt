# change: VAD start

import numpy as np
from faster_whisper import WhisperModel  # Speech-to-Text
import asyncio
import edge_tts  # Text-to-Speech
from deep_translator import GoogleTranslator
import os
import ffmpeg
import io
import threading
import webrtcvad

from utils import preprocess_ffmpeg

# imports for testing
import sounddevice as sd
import soundfile as sf
from playsound import playsound
import time as t


recordDuration = 2.5
samplerate = 16000  # 16 kHz

known_languages = ["de", "en"]
model = WhisperModel("tiny", device="cpu", compute_type="int8") 

vad = webrtcvad.Vad(2)

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


def float32_to_pcm16(audio_float32: np.ndarray) -> bytes:
    #Clipping, falls Werte leicht außerhalb [-1, 1] sind (numerische Fehler)
    audio_float32 = np.clip(audio_float32, -1.0, 1.0)

    #Skalieren auf int16-Bereich
    audio_int16 = (audio_float32 * 32767).astype(np.int16)

    #In Bytes umwandeln
    return audio_int16.tobytes()

def frame_generator(frame_duration_ms, audio_bytes, samplerate):
    frames = []
    """
    Schneidet die PCM-Bytes in gleichlange Frames.
    frame_size_in_bytes = samplerate * frame_ms/1000 * 2 (weil 16-bit = 2 Bytes)
    """
    frame_size = int(samplerate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    # nur volle Frames an VAD geben
    while offset + frame_size <= len(audio_bytes):
        frames.append(audio_bytes[offset:offset + frame_size])
        offset += frame_size
    return frames


def process_audio(recording):

    # Preprocessing 
    recording = preprocess_ffmpeg(recording, samplerate)
    
    # float32 -> PCM16 Bytes
    audio_bytes = float32_to_pcm16(recording)

    # Frames erzeugen
    frame_ms=30
    frames = frame_generator(frame_ms, audio_bytes, samplerate)
    # print(frames)
    
    for frame in frames:
        print(vad.is_speech(frame, samplerate))
    

    # Transkription
    segments, info = model.transcribe(recording, language=selected_language, beam_size=1)
    text = " ".join([s.text for s in segments]).strip()

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

    t.sleep(2)