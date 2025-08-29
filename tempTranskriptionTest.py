# import pyaudio
# import numpy as np
# import speech_recognition as sr

# # ----- Setup -----
# RATE = 16000
# CHUNK = 1600*11  # 1100ms bei 16kHz

# r = sr.Recognizer()


# # ----- Mikrofon-Stream -----
# p = pyaudio.PyAudio()
# stream = p.open(format=pyaudio.paInt16,
#                 channels=1,
#                 rate=RATE,
#                 input=True,
#                 frames_per_buffer=CHUNK)

# print("Start speaking...")

# buffer_text = ""

# while True:
#     data = stream.read(CHUNK, exception_on_overflow=False)
#     audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

#     # Transkription
#     text = r.recognize_google(audio_chunk, language="de")

#     if text:
#         buffer_text += text  # aktualisiere die aktuelle Hypothese
#         print("\r" + buffer_text, end=" ")  # live update, überschreibt vorherigen Text

#!/usr/bin/env python3

# NOTE: this example requires PyAudio because it uses the Microphone class

import os

import speech_recognition as sr

# obtain audio from the microphone
r = sr.Recognizer()


with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)

# recognize speech using Google Speech Recognition
try:
    print(r.recognize_google(audio, language="de"))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))


