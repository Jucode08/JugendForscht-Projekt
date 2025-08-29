import pyaudio
import numpy as np
from faster_whisper import WhisperModel

# ----- Setup -----
RATE = 16000
CHUNK = 1600*11  # 1100ms bei 16kHz

model = WhisperModel("tiny", device="cpu", compute_type="int8")  


# ----- Mikrofon-Stream -----
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Start speaking...")

buffer_text = ""

while True:
    data = stream.read(CHUNK, exception_on_overflow=False)
    audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    # Transkription
    segments, info = model.transcribe(audio_chunk, language="en") 
    text = " ".join([s.text for s in segments]).strip()

    if text:
        buffer_text += text  # aktualisiere die aktuelle Hypothese
        print("\r" + buffer_text, end=" ")  # live update, überschreibt vorherigen Text
