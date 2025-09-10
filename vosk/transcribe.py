import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer

lang = "fr"
# 1. Modell laden (z. B. Englisch, Deutsch, Französisch gibt es alles bei vosk)
model = Model("vosk/models/vosk-model-small-" + lang)  # Ordner mit entpacktem Modell

# 2. Rekognizer initialisieren (16kHz Mono)
rec = KaldiRecognizer(model, 16000)

# 3. Queue für Audio-Daten
audio_data_queue = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    audio_data_queue.put(bytes(indata))

# 4. Aufnahme starten
with sd.RawInputStream(samplerate=16000, blocksize=12000, dtype="int16",
                       channels=1, callback=callback):
    print("Sag etwas...")

    while True:
        data = audio_data_queue.get()
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            print("Final:", result["text"])
        else:
            partial = json.loads(rec.PartialResult())
            print("Partial:", partial["partial"])
        
