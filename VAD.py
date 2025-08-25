import webrtcvad
import collections
import numpy as np


vad = webrtcvad.Vad(2)
# Aggressivität: 0 = tolerant, 3 = sehr strikt 


def float32_to_pcm16(audio_float32: np.ndarray) -> bytes:
    #Clipping, falls Werte leicht außerhalb [-1, 1] sind (numerische Fehler)
    audio_float32 = np.clip(audio_float32, -1.0, 1.0)

    #Skalieren auf int16-Bereich
    audio_int16 = (audio_float32 * 32767).astype(np.int16)

    #In Bytes umwandeln
    return audio_int16.tobytes()

def frame_generator(frame_duration_ms, audio_bytes, sample_rate):
    frames = []
    """
    Schneidet die PCM-Bytes in gleichlange Frames.
    frame_size_in_bytes = sample_rate * frame_ms/1000 * 2 (weil 16-bit = 2 Bytes)
    """
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    # nur volle Frames an VAD geben
    while offset + frame_size <= len(audio_bytes):
        frames.append(audio_bytes[offset:offset + frame_size])
        offset += frame_size
    return frames


def collect_speech_blocks(frames, vad, sample_rate, frame_ms=30, padding_ms=300):
    """
    frames: Generator von PCM16 Bytes
    vad: webrtcvad.Vad Objekt
    sample_rate: 16000
    frame_ms: Framegröße in ms
    padding_ms: wie lange wir warten, bevor Segment endet
    """
    voiced_segments = []
    buffer_frames = collections.deque(maxlen=int(padding_ms / frame_ms))
    triggered = False
    current_segment = []

    for frame in frames:
        is_speech = vad.is_speech(frame, sample_rate)
        buffer_frames.append((frame, is_speech))

        if not triggered:
            # Trigger starten, wenn genug Sprache im Puffer
            num_voiced = sum(1 for f, s in buffer_frames if s)
            if num_voiced > 0.9 * buffer_frames.maxlen:
                triggered = True
                # Alle Frames aus dem Puffer zum Segment hinzufügen
                current_segment.extend(f for f, s in buffer_frames)
                buffer_frames.clear()
        else:
            # Trigger aktiv → Segment fortsetzen
            current_segment.append(frame)
            num_unvoiced = sum(1 for f, s in buffer_frames if not s)
            if num_unvoiced > 0.9 * buffer_frames.maxlen:
                # Segment beenden
                triggered = False
                voiced_segments.append(b"".join(current_segment))
                current_segment = []
                buffer_frames.clear()

    # Rest flushen
    if current_segment:
        voiced_segments.append(b"".join(current_segment))

    return voiced_segments


def split_into_speech_segments(audio_f32, sample_rate=16000, frame_ms=30, padding_ms=300, aggressiveness=2):
    """
    Komfort-Wrapper: nimmt float32-Audio, gibt eine Liste float32-Segmente mit Sprache zurück.
    """
    vad = webrtcvad.Vad(aggressiveness)

    # float32 -> PCM16 Bytes
    audio_bytes = float32_to_pcm16(audio_f32)

    # Frames erzeugen
    frames = frame_generator(frame_ms, audio_bytes, sample_rate)

    # Sprachblöcke sammeln
    segments_bytes = list(collect_speech_blocks(frames, vad, sample_rate, frame_ms, padding_ms))

    # Bytes -> float32 fürs ASR
    segments_f32 = [
        np.frombuffer(seg, dtype=np.int16).astype(np.float32) / 32768.0
        for seg in segments_bytes
    ]

    # Optional: ultrakurze Schnipsel aussortieren < 200 ms
    # min_len = int(0.2 * sample_rate)
    # segments_f32 = [s for s in segments_f32 if len(s) >= min_len]

    return segments_f32
