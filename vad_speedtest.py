import timeit
import sounddevice as sd
from faster_whisper import WhisperModel  # Speech-to-Text
import asyncio
import edge_tts  # Text-to-Speech
from deep_translator import GoogleTranslator
import os
import sounddevice as sd
from playsound import playsound

from utils import preprocess_ffmpeg
from vad import split_into_speech_segments


recordDuration = 10
samplerate = 16000  # 16 kHz
model = WhisperModel("tiny", device="cpu", compute_type="int8") 
selected_language = "fr"  # None := Sprache erkennen


def record():
    print("recording...")
    recording = sd.rec(
        int(recordDuration * samplerate), samplerate=samplerate, channels=1
    )  # channels=1 => mono , channels=2 => stereo
    sd.wait()

    return recording




def with_vad(audio):
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
    


def without_vad(audio):
    # Transkription
    segments, info = model.transcribe(audio, language=selected_language, beam_size=1)
    text = " ".join([s.text for s in segments]).strip()

    print(f"erkannter Text: {text}")



recording = record()
audio = preprocess_ffmpeg(recording, samplerate)
if input("Continue?: "):
    
    
    print("with vad:")
    t1 = timeit.default_timer()
    for i in range(10):
        with_vad(audio)
    t2 = timeit.default_timer()  
    print(t2-t1)
    
    print("\nwithout vad:")
    t3 = timeit.default_timer()
    for i in range(10):
        without_vad(audio)
    t4 = timeit.default_timer()  
    print(t4-t3)
