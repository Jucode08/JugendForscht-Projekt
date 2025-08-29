import pyaudio
import numpy as np
from faster_whisper import WhisperModel

# ----- Setup -----
RATE = 16000
CHUNK = 1600*11  # 1100ms bei 16kHz

model = WhisperModel("tiny", device="cpu")  


def transcribe_with_overlap(audio_data, window_size=2.0, overlap=0.5, sample_rate=16000):
    step_size = window_size - overlap
    num_samples = len(audio_data)
    transcriptions = []

    for start in range(0, num_samples, int(step_size * sample_rate)):
        end = min(start + int(window_size * sample_rate), num_samples)
        segment = audio_data[start:end]
        segments, _ = model.transcribe(segment, beam_size=5, language="de")
        transcription = " ".join([seg.text for seg in segments])
        transcriptions.append(transcription)

    return " ".join(transcriptions)

# ----- Mikrofon-Stream -----
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Start speaking...")


while True:
    data = stream.read(CHUNK, exception_on_overflow=False)
    audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    # Transkription
    t = transcribe_with_overlap(audio_chunk)
    if t:
        print(transcribe_with_overlap(audio_chunk))
