import numpy as np
import ffmpeg
import io
import soundfile as sf


def preprocess_ffmpeg(recording, samplerate):
    # Schreibe das NumPy-Array in einen Bytes-Stream (wie eine Datei im RAM)
    buf = io.BytesIO()
    sf.write(buf, recording, samplerate, format="WAV")
    buf.seek(0)  # zurück an den Anfang
    
    # Füttere ffmpeg direkt mit Bytes (pipe:0 = stdin, pipe:1 = stdout)
    out, _ = (
        ffmpeg
        .input("pipe:0")
        .output("pipe:1", format="s16le", acodec="pcm_s16le", ac=1, ar="16000")
        .run(input=buf.read(), capture_stdout=True, quiet=True)
    )
    audio = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
    return audio