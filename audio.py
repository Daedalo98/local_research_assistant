# audio.py
import pyaudio
import numpy as np
import threading
import queue
import asyncio
import os
import tempfile
import edge_tts
from faster_whisper import WhisperModel

class Transcriber:
    def __init__(self, model_size="base", silence_threshold=0.01):
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_thread = None
        self.silence_threshold = silence_threshold

    def _calculate_rms(self, audio_data):
        """Calculates Root Mean Square energy of the audio chunk."""
        return np.sqrt(np.mean(np.square(audio_data)))

    def _record_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        
        while self.is_recording:
            try:
                data = stream.read(8000, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # VAD: Only put audio in queue if it exceeds silence threshold
                if self._calculate_rms(audio_data) > self.silence_threshold:
                    self.audio_queue.put(audio_data)
            except Exception as e:
                print(f"Audio stream error: {e}")
                break
                
        stream.stop_stream()
        stream.close()
        p.terminate()

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            with self.audio_queue.mutex:
                self.audio_queue.queue.clear()
            self.audio_thread = threading.Thread(target=self._record_audio, daemon=True)
            self.audio_thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self.audio_thread:
            self.audio_thread.join()

    def transcribe_chunk(self):
        """Consumes a bounded amount of audio to prevent lag."""
        frames = []
        # Pull up to 5 chunks at a time to keep UI responsive
        for _ in range(5): 
            if not self.audio_queue.empty():
                frames.append(self.audio_queue.get())
            else:
                break
        
        if not frames:
            return "", []

        audio_chunk = np.concatenate(frames)
        waveform_data = np.clip(np.abs(audio_chunk[::100]), 0, 1) # Simplified waveform

        segments, _ = self.model.transcribe(audio_chunk, beam_size=5, language="en")
        text = " ".join([segment.text for segment in segments])
        
        return text.strip(), waveform_data

def generate_tts(text, voice="en-US-AriaNeural"):
    """Generates TTS using Microsoft Edge Neural voices and saves to a temp file."""
    if not text.strip():
        return None
        
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file.close() # Close so edge-tts can write to it
    
    # edge_tts is async, we need a sync wrapper
    async def _amain():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(temp_file.name)
        
    asyncio.run(_amain())
    return temp_file.name