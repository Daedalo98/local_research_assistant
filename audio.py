# audio.py
import pyaudio
import numpy as np
import threading
import queue
from faster_whisper import WhisperModel
import time

class Transcriber:
    def __init__(self, model_size="base"):
        """Initializes Faster-Whisper. Will download the model on first run."""
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_thread = None

    def _record_audio(self):
        """Background thread function to capture audio smoothly."""
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        
        while self.is_recording:
            try:
                data = stream.read(8000, exception_on_overflow=False)
                # Convert raw bytes to numpy array (Faster-Whisper expects float32 between -1 and 1)
                audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
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
            # Clear previous queue
            while not self.audio_queue.empty():
                self.audio_queue.get()
            self.audio_thread = threading.Thread(target=self._record_audio, daemon=True)
            self.audio_thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self.audio_thread:
            self.audio_thread.join()

    def transcribe_chunk(self):
        """Pulls accumulated audio from the queue and transcribes it."""
        if self.audio_queue.empty():
            return "", []

        # Gather all audio currently in the queue
        frames = []
        while not self.audio_queue.empty():
            frames.append(self.audio_queue.get())
        
        if not frames:
            return "", []

        audio_chunk = np.concatenate(frames)
        
        # Calculate visual waveform data
        chunk_length = max(1, len(audio_chunk) // 100)
        compressed = np.array([np.max(np.abs(audio_chunk[i:i+chunk_length])) for i in range(0, len(audio_chunk), chunk_length)])
        waveform_data = np.clip(compressed, 0, 1)

        # Transcribe using Faster-Whisper
        segments, _ = self.model.transcribe(audio_chunk, beam_size=5, language="en", condition_on_previous_text=False)
        
        text = " ".join([segment.text for segment in segments])
        return text.strip(), waveform_data