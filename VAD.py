import webrtcvad
import collections
import contextlib
import wave
import os

vad = webrtcvad.Vad(2)
# Aggressivität: 0 = tolerant, 3 = sehr strikt (nur sehr klare Sprache)

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)  # 16-bit PCM = 2 Bytes
    offset = 0
    while offset + n < len(audio):
        yield audio[offset:offset + n]
        offset += n

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)

    triggered = False
    voiced_frames = []

    for frame in frames:
        is_speech = vad.is_speech(frame, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b"".join(voiced_frames)
                ring_buffer.clear()
                voiced_frames = []

    if voiced_frames:
        yield b"".join(voiced_frames)

