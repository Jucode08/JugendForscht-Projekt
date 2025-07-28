import numpy as np
import whisper  # Speech-to-Text
from langdetect import detect  # Sprache erkennen
from gtts import gTTS  # Text-to-Speech
from deep_translator import GoogleTranslator
import os


#imports for testing
import sounddevice as sd
from playsound import playsound 
import threading

def play_and_delete(path):
    playsound(path)
    os.remove(path) 

recordDuration = 3
samplerate = 16000  # 16 kHz

# knownLanguages = ["de", "en"]
knownLanguages = ["de"]
model = whisper.load_model("tiny")

while True:

    print("recording...")
    recording = sd.rec(int(recordDuration * samplerate), samplerate=samplerate, channels=1) #channels=1 => mono , channels=2 => stereo
    sd.wait()

    # Wandelt (N, 1) => (N,) um und castet auf float32
    recording = recording.flatten().astype(np.float32)

    # Whisper-Preprocessing
    audio = whisper.pad_or_trim(recording)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # logging
    _, probabilities = model.detect_language(mel)
    language = max(probabilities, key=probabilities.get)
    print("Sprache erkannt als:", language)
    print("Wahrscheinlichkeiten: ", "de: " , probabilities["de"], " en: " , probabilities["en"])

    # Transkription
    # options = whisper.DecodingOptions(fp16=False)  # wenn man keine GPU nutzt
    result = whisper.decode(model, mel)

    print(result.text)

    if language not in knownLanguages:
        translated = GoogleTranslator(source=language, target=knownLanguages[0]).translate(result.text) 
        tts = gTTS(text=translated, lang=knownLanguages[0])
        tts.save("output.mp3")
        print(translated)
        
    threading.Thread(target=play_and_delete, args=("output.mp3",), daemon=True).start()

