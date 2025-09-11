import sounddevice as sd
import queue
import collections
import json
from vosk import Model, KaldiRecognizer
import webrtcvad
import sys
from time import time

import argparse
parser = argparse.ArgumentParser(description='Start Speech-to-Text test with configuration options.')
parser.add_argument('-l', '--lang', '--language', type=str, default="en",
                help='Language code for the STT model to transcribe in a specific language.')
parser.add_argument('--log', action='store_true', default=False)
args = parser.parse_args()

LOGGING = args.log
LANG = args.lang

SAMPLE_RATE = 16000
BLOCK_SIZE = 3200        # ca. 0.1s bei 16kHz
VAD_FRAME_MS = 30        # ms
PREBUFFER_DURATION = 1 # Sekunden
POST_SPEECH_SILENCE = 0.8 # Sekunden
last_speech_time = None

STATE = "WAITING"

model = Model("vosk/models/vosk-model-small-" + LANG) 
rec = KaldiRecognizer(model, 16000)
audio_data_queue = queue.Queue()
vad = webrtcvad.Vad(3)

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
    if LOGGING: print(">>> Current State: ", STATE)

    while True:
        try: 
        
            data = audio_data_queue.get()
            
            if STATE == "WAITING":
                prebuffer.append(data)
                
                frame_len = int(SAMPLE_RATE * (VAD_FRAME_MS / 1000.0)) * 2  # 2 Byte pro Sample
                for i in range(0, len(data), frame_len):
                    frame = data[i:i+frame_len]
                    if len(frame) == frame_len and vad.is_speech(frame, SAMPLE_RATE):
                        if LOGGING: print("\n>>> Speech detected. Switching to TRANSCRIBING...")
                        STATE = "TRANSCRIBING"
                        
                        if len(prebuffer) > 0:
                            pb_bytes = b"".join(prebuffer)
                            rec.AcceptWaveform(pb_bytes)
                        break
            
            elif STATE == "TRANSCRIBING":
                if not rec.AcceptWaveform(data):
                    partial = json.loads(rec.PartialResult())
                    if partial["partial"] != "":
                        print(f"\r{partial["partial"]}", end=" ")
                else:
                    result = json.loads(rec.Result())
                    print("\n> ", result["text"])
                    
                speech_detected_in_block = False
                frame_len = int(SAMPLE_RATE * (VAD_FRAME_MS / 1000.0)) * 2
                for i in range(0, len(data), frame_len):
                    frame = data[i:i+frame_len]
                    if len(frame) == frame_len and vad.is_speech(frame, SAMPLE_RATE):
                        speech_detected_in_block = True
                        
                if speech_detected_in_block:
                    last_speech_time = time()

                if last_speech_time and (time() - last_speech_time) > POST_SPEECH_SILENCE:
                    if LOGGING : print("\n<<< Silence detected. Switching to WAITING...")
                    STATE = "WAITING"
                    prebuffer.clear()
                    last_speech_time = None
        except KeyboardInterrupt:
            print("Exiting...")
            sys.exit(0)
        
