import logging
import os
import threading
import wave
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Optional

import numpy as np
import subprocess
import whisper
from datetime import datetime

logger = logging.getLogger('dndscribe')

@dataclass
class TranscriptionTask:
    audio_data: bytes
    user_id: int
    timestamp: str

class TranscriptionPool:
    def __init__(self, num_workers: int = 2, model: Optional[whisper.Whisper] = None, debug_audio: bool = True):
        self.task_queue = Queue()
        self.active = True
        self.model = model or whisper.load_model("base", download_root="models")
        self.debug_audio = debug_audio
        
        # Create worker threads
        self.workers = [
            threading.Thread(target=self._worker_loop, name=f"transcriber-{i}")
            for i in range(num_workers)
        ]
        
        for worker in self.workers:
            worker.daemon = True
            worker.start()
        
        logger.info(f"Started {num_workers} transcription workers")

    def _process_audio(self, audio_bytes: bytes) -> np.ndarray:
        """
        Process raw audio bytes into a format suitable for Whisper.
        Uses ffmpeg for reliable conversion from Discord's format to Whisper's expected format.
        """
        try:
            # Create a temporary file for ffmpeg processing
            temp_input = f"temp_input_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.raw"
            temp_output = f"temp_output_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.raw"
            
            try:
                # Write input data to temporary file
                with open(temp_input, 'wb') as f:
                    f.write(audio_bytes)
                
                # Use ffmpeg for audio processing
                ffmpeg_command = [
                    "ffmpeg",
                    "-f", "s16le",         # Input format: signed 16-bit little-endian
                    "-ar", "48000",        # Input sample rate: 48000 Hz
                    "-ac", "2",            # Input channels: 2 (Stereo)
                    "-i", temp_input,      # Input file
                    "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=resampler=soxr",  # Normalize audio loudness with high quality resampling
                    "-f", "f32le",         # Output format: 32-bit float
                    "-ac", "1",            # Output channels: 1 (Mono)
                    "-ar", "16000",        # Output sample rate: 16000 Hz (Whisper's expected rate)
                    temp_output            # Output file
                ]

                # Run ffmpeg process
                process = subprocess.run(
                    ffmpeg_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
                
                # Read processed output
                with open(temp_output, 'rb') as f:
                    processed_data = f.read()
                
                # Convert to float32 array (already in float32 format from ffmpeg)
                return np.frombuffer(processed_data, dtype=np.float32)
                
            finally:
                # Clean up temporary files
                if os.path.exists(temp_input):
                    os.remove(temp_input)
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                    
        except subprocess.CalledProcessError as e:
            error_output = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "Unknown ffmpeg error"
            logger.error(f"FFmpeg error: {error_output}")
            return self._fallback_processing(audio_bytes)
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            return self._fallback_processing(audio_bytes)
    
    def _fallback_processing(self, audio_bytes: bytes) -> np.ndarray:
        """Fallback method if ffmpeg processing fails"""
        logger.warning("Using fallback audio processing method")
        try:
            # Convert bytes to numpy array (assuming 16-bit stereo at 48kHz)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Check if we have enough data
            if len(audio_np) < 100:
                logger.warning(f"Audio too short: {len(audio_np)} samples")
                return np.zeros(1600, dtype=np.float32)
            
            # Reshape to stereo (2 channels) if applicable
            if len(audio_np) % 2 == 0:  # Ensure even number of samples
                audio_np = audio_np.reshape(-1, 2)
                # Convert to mono by averaging channels
                audio_mono = audio_np.mean(axis=1).astype(np.int16)
            else:
                # If odd number of samples, just use as is (assuming mono)
                logger.warning(f"Odd number of samples in audio: {len(audio_np)}")
                audio_mono = audio_np
            
            # Resample from 48kHz to 16kHz (or whatever the original rate was to 16kHz)
            original_rate = 48000
            target_rate = 16000
            
            if len(audio_mono) > 0:
                target_length = int(len(audio_mono) * target_rate / original_rate)
                if target_length > 0:
                    indices = np.linspace(0, len(audio_mono) - 1, target_length)
                    resampled = np.interp(indices, np.arange(len(audio_mono)), audio_mono)
                else:
                    resampled = np.zeros(1600, dtype=np.float32)  # 100ms of silence
            else:
                resampled = np.zeros(1600, dtype=np.float32)
            
            # Normalize to float32 in [-1, 1] range
            return resampled.astype(np.float32) / 32768.0
            
        except Exception as e:
            logger.error(f"Fallback processing failed: {str(e)}", exc_info=True)
            # Return empty array rather than crashing
            return np.zeros(1600, dtype=np.float32)  # 100ms of silence at 16kHz

    def _worker_loop(self):
        """Main worker thread loop"""
        while self.active:
            try:
                task = self.task_queue.get(timeout=1)
                self._process_task(task)
                self.task_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in transcription worker: {e}", exc_info=True)

    def _save_debug_audio(self, audio_bytes: bytes, user_id: int, timestamp: str):
        """Save raw audio to wav file for debugging"""
        try:
            user_folder = f"recordings/{user_id}/debug"
            os.makedirs(user_folder, exist_ok=True)
            
            # Save as WAV
            wav_filename = f"{user_folder}/{timestamp}.wav"
            with wave.open(wav_filename, 'wb') as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(48000)
                wav_file.writeframes(audio_bytes)
            logger.info(f"Saved debug WAV file: {wav_filename}")
            
        except Exception as e:
            logger.error(f"Error saving debug audio: {e}")

    def _save_debug_resampled_audio(self, audio: np.ndarray, user_id: int, timestamp: str):
        """Save resampled audio to wav file for debugging"""
        try:
            user_folder = f"recordings/{user_id}/debug"
            os.makedirs(user_folder, exist_ok=True)
            filename = f"{user_folder}/{timestamp}-resampled.wav"
            
            # Convert float32 back to int16
            audio_int16 = (audio * 32768.0).astype(np.int16)
            
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(audio_int16.tobytes())
            
            logger.info(f"Saved resampled debug audio file: {filename}")
        except Exception as e:
            logger.error(f"Error saving resampled debug audio: {e}")

    def _check_audio_quality(self, audio: np.ndarray) -> bool:
        """Check if audio quality is sufficient for transcription"""
        # Check for silence
        if np.abs(audio).max() < 0.01:
            logger.warning("Audio is nearly silent")
            return False
            
        # Check for extremely short audio
        if len(audio) < 8000:  # Less than 0.5 seconds at 16kHz
            logger.warning(f"Audio too short: {len(audio)} samples, {len(audio)/16000:.2f}s")
            return False
            
        return True

    def _process_task(self, task: TranscriptionTask):
        """Process a transcription task"""
        try:
            # Save original audio if debug is enabled
            if self.debug_audio:
                self._save_debug_audio(task.audio_data, task.user_id, task.timestamp)
            
            # Check if there's enough audio data
            if len(task.audio_data) < 1000:
                logger.warning(f"Audio data too short for user {task.user_id}: {len(task.audio_data)} bytes")
                return
            
            # Process audio data
            audio = self._process_audio(task.audio_data)
            
            # Save resampled audio if debug is enabled
            if self.debug_audio:
                self._save_debug_resampled_audio(audio, task.user_id, task.timestamp)
            
            # Check audio quality before transcription
            if not self._check_audio_quality(audio):
                logger.warning(f"Audio quality check failed for user {task.user_id}")
                return
            
            # Transcribe the audio
            logger.info(f"Starting transcription for user {task.user_id}, audio length: {len(audio)/16000:.2f}s")
            result = self.model.transcribe(audio, fp16=False, language="en")
            
            # Check if transcription is empty
            if not result["text"].strip():
                logger.warning(f"Empty transcription for user {task.user_id}")
                return
            
            # Write transcription to file
            user_folder = f"recordings/{task.user_id}"
            os.makedirs(user_folder, exist_ok=True)
            transcription_file = f"{user_folder}/transcription.txt"
            
            with open(transcription_file, "a") as f:
                f.write(f"[{task.timestamp}] ")
                f.write(result["text"].strip())
                f.write("\n")
            
            logger.info(f"Added transcription for user {task.user_id}: {result['text'][:50]}...")
            
        except Exception as e:
            logger.error(f"Error transcribing audio for user {task.user_id}: {e}", exc_info=True)

    def add_task(self, audio_data: bytes, user_id: int, timestamp: str):
        """Add a new transcription task to the queue"""
        task = TranscriptionTask(audio_data, user_id, timestamp)
        self.task_queue.put(task)
        logger.debug(f"Added transcription task for user {user_id}, data size: {len(audio_data)} bytes")

    def stop(self):
        """Stop the worker threads and wait for pending tasks"""
        self.active = False
        logger.info("Waiting for transcription tasks to complete...")
        self.task_queue.join()
        logger.info("All transcription tasks completed")