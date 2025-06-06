import os
import wave  # für Audio-Dateien
import whisper  # Speech-to-Text
from langdetect import detect  # Sprache erkennen
from gtts import gTTS  # Text-to-Speech
from pydub import AudioSegment  # Audio konvertieren/abspielen

model = whisper.load_model("tiny")  # Whisper-Modell laden