import sounddevice as sd
import queue
import collections
import json
from vosk import Model, KaldiRecognizer
import webrtcvad
import sys

import argparse
parser = argparse.ArgumentParser(description='Start Speech-to-Text test with configuration options.')
parser.add_argument('-l', '--lang', '--language', type=str, default="en",
                help='Language code for the STT model to transcribe in a specific language.')
parser.add_argument('--log', action='store_true', default=False)
args = parser.parse_args()

LOGGING = args.log
LANG = args.lang

SAMPLE_RATE = 16000
BLOCK_SIZE = 3200        # ca. 0.3s bei 16kHz
VAD_FRAME_MS = 30        # ms
PREBUFFER_DURATION = 0.75 # Sekunden

STATE = "WAITING"

model = Model("vosk/models/vosk-model-small-" + LANG) 
rec = KaldiRecognizer(model, 16000)
audio_data_queue = queue.Queue()
vad = webrtcvad.Vad(2)

max_prebuffer_frames = int(PREBUFFER_DURATION * SAMPLE_RATE / (VAD_FRAME_MS/1000))
prebuffer = collections.deque(maxlen=max_prebuffer_frames)

def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    audio_data_queue.put(bytes(indata))

#  Aufnahme starten
with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, dtype="int16",
                       channels=1, callback=callback):
    print("System ready. Waiting for speech...")

    while True:
        try: 
        
            data = audio_data_queue.get()
            
            if STATE == "WAITING":
                frame_len = int(SAMPLE_RATE * (VAD_FRAME_MS / 1000.0)) * 2  # 2 Byte pro Sample
                for i in range(0, len(data), frame_len):
                    frame = data[i:i+frame_len]
                    if len(frame) == frame_len and vad.is_speech(frame, SAMPLE_RATE):
                        if LOGGING: print(">>> Speech detected. Switching to TRANSCRIBING...")
                        STATE = "TRANSCRIBING"
                        
                        for pb in prebuffer:
                            rec.AcceptWaveform(pb)
                        break
            
            elif STATE == "TRANSCRIBING":
                if not rec.AcceptWaveform(data):
                    partial = json.loads(rec.PartialResult())
                    if partial["partial"] != "":
                        print(f"\r{partial["partial"]}", end=" ")
                else:
                    result = json.loads(rec.Result())
                    print("\n> ", result["text"])
                    
                silence_frames = 0
                frame_len = int(SAMPLE_RATE * (VAD_FRAME_MS / 1000.0)) * 2
                for i in range(0, len(data), frame_len):
                    frame = data[i:i+frame_len]
                    if len(frame) == frame_len and not vad.is_speech(frame, SAMPLE_RATE):
                        silence_frames += 1

                if silence_frames > 5:  #TODO: change to timer 
                    if LOGGING : print("<<< Silence detected. Switching to WAITING...")
                    STATE = "WAITING"
                    prebuffer.clear()
        except KeyboardInterrupt:
            print("Exiting...")
            sys.exit(0)
        
