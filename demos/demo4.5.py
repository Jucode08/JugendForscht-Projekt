# more readable

import numpy as np
import whisper  # Speech-to-Text
from langdetect import detect  # Sprache erkennen
from gtts import gTTS  # Text-to-Speech
from deep_translator import GoogleTranslator
import os


#imports for testing
import sounddevice as sd
from playsound import playsound 
import sys

recordDuration = 3
samplerate = 16000  # 16 kHz

knownLanguages = ["de"]
model = whisper.load_model("tiny")

translationSupport = ['ar', 'zh-TW', 'en', 'fr', 'de', 'hi', 'id', 'it', 'ja', 'kn', 'ko', 'mr', 'fa', 'pl', 'pt', 'ru', 'es', 'sw', 'ta', 'th', 'tr', 'uk', 'ur', 'vi']

selected_language = "fr"


def record():
    print("recording...")
    recording = sd.rec(int(recordDuration * samplerate), samplerate=samplerate, channels=1) #channels=1 => mono , channels=2 => stereo
    sd.wait()
    
    process_audio(recording)
    # threading.Thread(target=process_audio, args=(recording,), daemon=True).start()
    

def process_audio(recording):
    
    # Wandelt (N, 1) => (N,) um und castet auf float32
    recording = recording.flatten().astype(np.float32)

    # Whisper-Preprocessing
    audio = whisper.pad_or_trim(recording)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    confidence = whisper.decode(model, mel).avg_logprob
    print(f"confidence: {confidence}")
    # if confidence < -.97:
    #     print("Transkription zu unsicher – überspringe")
    #     sys.exit()

    _, probabilities = model.detect_language(mel)
    language = max(probabilities, key=probabilities.get)
    if language not in translationSupport:
        print("language not supported")
        sys.exit()
    print("Sprache erkannt als:", language)

    # Transkription
    options = whisper.DecodingOptions(language=selected_language, fp16=False)
    result = whisper.decode(model, mel, options)

    print(result.text)
    
    if language not in knownLanguages:
        translated = GoogleTranslator(source=language, target=knownLanguages[0]).translate(result.text) 
        textToSpeach(language, translated)
        print(translated)
    
    play_and_delete("output.mp3")
    # threading.Thread(target=play_and_delete, args=("output.mp3",), daemon=True).start()

    
def textToSpeach(language, text):
    tts = gTTS(text=text, lang=knownLanguages[0])
    tts.save("output.mp3")

def play_and_delete(path): 
    playsound(path)
    os.remove(path) 





record()










