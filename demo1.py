import numpy as np
import whisper  # Speech-to-Text
from langdetect import detect  # Sprache erkennen
from gtts import gTTS  # Text-to-Speech
from deep_translator import GoogleTranslator

#imports for testing
import sounddevice as sd

# for i in range(10):
recordDuration = 4
samplerate = 16000  # 16 kHz

print("recording...")
recording = sd.rec(int(recordDuration * samplerate), samplerate=samplerate, channels=1) #channels=1 => mono , channels=2 => stereo
sd.wait()

# Wandelt (N, 1) => (N,) um und castet auf float32
recording = recording.flatten().astype(np.float32)

# print("playing...")
# sd.play(recording, samplerate=samplerate)
# sd.wait()

knownLanguages = ["de", "en"]
model = whisper.load_model("tiny")

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

