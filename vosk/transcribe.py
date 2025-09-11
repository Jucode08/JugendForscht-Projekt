import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer

import argparse
parser = argparse.ArgumentParser(description='Start Speech-to-Text test with configuration options.')

parser.add_argument('-l', '--lang', '--language', type=str, default="en",
                help='Language code for the STT model to transcribe in a specific language.')

args = parser.parse_args()

LANG = args.lang
SAMPLE_RATE = 16000
BLOCK_SIZE = 3200


# 1. Modell laden (z. B. Englisch, Deutsch, Französisch gibt es alles bei vosk)
model = Model("vosk/models/vosk-model-small-" + LANG)  # Ordner mit entpacktem Modell

# 2. Rekognizer initialisieren (16kHz Mono)
rec = KaldiRecognizer(model, SAMPLE_RATE)

# 3. Queue für Audio-Daten
audio_data_queue = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    audio_data_queue.put(bytes(indata))

# 4. Aufnahme starten
with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, dtype="int16",
                       channels=1, callback=callback):
    print("System ready. Waiting for speech...")

    while True:
        data = audio_data_queue.get()
        if not rec.AcceptWaveform(data):
            partial = json.loads(rec.PartialResult())
            if partial["partial"] != "":
                print(f"\r{partial["partial"]}", end=" ")
        else:
            result = json.loads(rec.Result())
            print("\n> ", result["text"])
        
