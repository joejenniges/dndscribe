import logging
import os
import threading
import wave
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Optional, Dict
import signal
import sys
import time
import numpy as np
import subprocess
import whisper
from whisper.decoding import DecodingOptions
from datetime import datetime
import io
import soundfile as sf
import torch

logger = logging.getLogger('dndscribe')

@dataclass
class TranscriptionTask:
    audio_data: bytes
    user_id: int
    username: str
    timestamp: datetime

@dataclass
class WriterTask:
    text: str
    username: str
    timestamp: datetime

class TranscriptionPool:
    def __init__(self, num_workers: int = 2, model: Optional[whisper.Whisper] = None, debug_audio: bool = False):
        self.task_queue = Queue()
        self.writer_queue = Queue()
        self.active = True
        self.model = model or whisper.load_model("base", download_root="models")
        self.debug_audio = debug_audio
        
        # Create worker threads
        self.workers = [
            threading.Thread(target=self._worker_loop, name=f"transcriber-{i}")
            for i in range(num_workers)
        ]

        self.writer_worker = threading.Thread(target=self._writer_worker_loop, name="writer-worker")
        self.writer_worker.start()

        self.new_audio_process = False
        
        for worker in self.workers:
            worker.start()
        
        logger.info(f"Started {num_workers} transcription workers")


    def _trim_silence(self, audio: np.ndarray, threshold: float = 0.01, min_silence_duration: float = 1.0) -> np.ndarray:
        """
        Trim silence from audio if it lasts longer than min_silence_duration seconds.
        
        Args:
            audio: Input audio array
            threshold: Amplitude threshold for silence detection (default: 0.01)
            min_silence_duration: Minimum duration of silence to trim in seconds (default: 1.0)
            
        Returns:
            Trimmed audio array
        """
        # Calculate samples per silence duration
        samples_per_silence = int(min_silence_duration * 16000)  # 16000 is our target sample rate
        
        # Find non-silent regions
        is_silent = np.abs(audio) < threshold
        
        # Find transitions between silent and non-silent regions
        silent_starts = np.where(np.diff(is_silent.astype(int)) == 1)[0]
        silent_ends = np.where(np.diff(is_silent.astype(int)) == -1)[0]
        
        # Handle edge cases
        if len(silent_starts) == 0 and len(silent_ends) == 0:
            return audio
            
        if len(silent_starts) == 0:
            silent_starts = np.array([0])
        if len(silent_ends) == 0:
            silent_ends = np.array([len(audio)])
            
        # Ensure we have matching start/end pairs
        if silent_starts[0] > silent_ends[0]:
            silent_starts = np.insert(silent_starts, 0, 0)
        if silent_ends[-1] < silent_starts[-1]:
            silent_ends = np.append(silent_ends, len(audio))
            
        # Find long silent regions
        silent_durations = silent_ends - silent_starts
        long_silence_mask = silent_durations >= samples_per_silence
        
        if not np.any(long_silence_mask):
            return audio
            
        # Create mask for regions to keep
        keep_mask = np.ones(len(audio), dtype=bool)
        for start, end in zip(silent_starts[long_silence_mask], silent_ends[long_silence_mask]):
            keep_mask[start:end] = False
            
        # Apply mask and return trimmed audio
        return audio[keep_mask]

    def _process_audio(self, audio_bytes: bytes) -> np.ndarray:
        """
        Process raw audio bytes into a format suitable for Whisper.
        Uses ffmpeg for reliable conversion from Discord's format to Whisper's expected format.
        """
        try:
            # Use ffmpeg for audio processing
            ffmpeg_command = [
                "ffmpeg",
                "-f", "s16le",         
                "-ar", "48000",        
                "-ac", "2",            
                "-i", "pipe:0",        
                "-f", "s16le",         
                "-ac", "1",            
                "-ar", "16000",        
                "pipe:1"               
            ]

            # Run ffmpeg process
            process = subprocess.Popen(
                ffmpeg_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            try:
                out, err = process.communicate(input=audio_bytes, timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                logger.error("FFmpeg timed out")
                return self._fallback_processing(audio_bytes)

            if process.returncode != 0:
                logger.error(f"FFmpeg process returned non-zero exit code: {process.returncode}")
                logger.error(f"Error output: {err.decode('utf-8', errors='ignore')}")
            if len(out) == 0:
                logger.error("FFmpeg output is empty")
                raise ValueError("FFmpeg output is empty")
            
            # Convert to float32 array (already in float32 format from ffmpeg)
            audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
            
            # Trim silence from the audio
            audio = self._trim_silence(audio)

            if audio.size == 0:
                raise ValueError("Audio was empty after silence trimming.")
            
            return audio
                    
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
            audio = resampled.astype(np.float32) / 32768.0

            audio = self._trim_silence(audio)

            if audio.size == 0:
                raise ValueError("Audio was empty after silence trimming.")
            
            return audio
            
        except Exception as e:
            logger.error(f"Fallback processing failed: {str(e)}", exc_info=True)
            # Return empty array rather than crashing
            return np.zeros(1600, dtype=np.float32)  # 100ms of silence at 16kHz

    def _worker_loop(self):
        """Main worker thread loop"""
        while self.active:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:
                    break
                self._transcribe_audio_task(task)
                self.task_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in transcription worker: {e}", exc_info=True)

    def _writer_worker_loop(self):
        """Writer worker thread loop"""
        while self.active:
            try:
                task = self.writer_queue.get(timeout=1)
                if task is None:
                    break
                self._write_transcription_task(task)
                self.writer_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in writer worker: {e}", exc_info=True)

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
            
            return wav_filename
            
        except Exception as e:
            logger.error(f"Error saving debug audio: {e}")
            return None

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
            
            return filename
        except Exception as e:
            logger.error(f"Error saving resampled debug audio: {e}")
            return None

    def _check_audio_quality(self, audio: np.ndarray) -> bool:
        """Check if audio quality is sufficient for transcription"""
        # Check for silence
        if np.abs(audio).max() < 0.01:
            return False
            
        # Check for extremely short audio
        if len(audio) < 8000:  # Less than 0.5 seconds at 16kHz
            return False
            
        return True

    def _transcribe_audio_task(self, task: TranscriptionTask):
        """Process a transcription task"""
        result = None  # Initialize result to None
        filename = None
        try:
            # Save original audio if debug is enabled
            if self.debug_audio:
                self._save_debug_audio(task.audio_data, task.user_id, task.timestamp.strftime("%Y-%m-%d %H-%M-%S.%f"))
            
            # Check if there's enough audio data
            if len(task.audio_data) < 1000:
                logger.warning(f"Audio data too short for user {task.user_id}: {len(task.audio_data)} bytes")
                return
            
            audio = self._process_audio(task.audio_data)

            
            # Save resampled audio if debug is enabled
            if self.debug_audio:
                self._save_debug_resampled_audio(audio, task.user_id, task.timestamp.strftime("%Y-%m-%d %H-%M-%S.%f"))
            
            # Check audio quality before transcription
            if not self._check_audio_quality(audio):
                return

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if hasattr(self.model, 'decoder'):
                self.model.decoder.kv_cache = {}

            try:            
                # Transcribe the audio
                result = self.model.transcribe(audio, fp16=False, language="en")
            except Exception as e:
                logger.error(f"Error during transcription for user {task.username}: {e}", exc_info=True)
                logger.info("Trying fallback processing from audio file")
                filename = self._save_debug_resampled_audio(audio, task.user_id, task.timestamp.strftime("%Y-%m-%d %H-%M-%S.%f"))
                # Sleep briefly to allow system resources to stabilize
                time.sleep(0.05)
                result = self.model.transcribe(filename, fp16=False, language="en")
                return
                
            # Check if transcription is empty
            if not result or not result.get("text", "").strip():
                return
            
            # Remove file if we're able to successfully transcribe
            if filename:
                os.remove(filename)

            self.add_writer_task(result["text"].strip(), task.username, task.timestamp)
            logger.debug(f"Created transcription for user {task.username}: {result['text'][:50]}...")
            
        except Exception as e:
            logger.error(f"Error transcribing audio for user {task.username}, audio length: {len(audio)/16000:.2f}s: {e}", exc_info=True)

    def _write_transcription_task(self, task: WriterTask):
        """Write transcription to file"""
        try:
            transcription_file = "recordings/transcription.txt"
            ascii_text = ''.join(char for char in task.text if ord(char) < 128)
            ascii_username = ''.join(char for char in task.username if ord(char) < 128)
            with open(transcription_file, "a", encoding="utf-8") as f:
                # Strip non-ASCII characters from text before writing
                f.write(f"[{task.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')}]: {ascii_username}: {ascii_text}\n")

        except Exception as e:
            logger.error(f"Error writing transcription for user {task.username}: {e}", exc_info=True)

    def add_transcription_task(self, audio_data: bytes, user_id: int, username: str, timestamp: str):
        """Add a new transcription task to the queue"""
        task = TranscriptionTask(audio_data, user_id, username, timestamp)
        self.task_queue.put(task)
        logger.debug(f"Added transcription task for user {user_id}, data size: {len(audio_data)} bytes")

    def add_writer_task(self, text: str, username: str, timestamp: datetime):
        """Add a new writer task to the queue"""
        task = WriterTask(text, username, timestamp)
        self.writer_queue.put(task)
        logger.debug(f"Added writer task for user {username}, text length: {len(text)}")

    def cleanup_transcription_file(self):
        """Cleanup the transcription file"""
        transcription_file = "recordings/transcription.txt"
        lines: Dict[datetime, str] = {}
        first_timestamp = None
        with open(transcription_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    timestamp, username, text = line.strip().split(": ", 2)
                except ValueError:
                    continue
                if first_timestamp is None:  # Strip the square brackets and milliseconds
                    first_timestamp = timestamp[1:-8]
                timestamp = datetime.strptime(timestamp, "[%Y-%m-%d %H:%M:%S.%f]")
                lines[timestamp] = f"{username}: {text.strip()}"

        out_file = f"recordings/transcription-{first_timestamp}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            for timestamp, text in sorted(lines.items()):
                f.write(f"{text}\n")

        with open(transcription_file, "w", encoding="utf-8") as f:
            f.write("")       
        logger.info(f"Saved transcription file: {out_file}")


    def stop(self):
        """Stop all workers and clean up resources"""
        logger.info("Stopping transcription pool...")
        self.active = False
        
        # Send None to queues to signal workers to stop
        for _ in self.workers:
            self.task_queue.put(None)
        self.writer_queue.put(None)
        
        # Wait for all workers to finish with timeout
        for worker in self.workers:
            worker.join(timeout=10)  # Reduced timeout to 2 seconds
            if worker.is_alive():
                logger.warning(f"Worker {worker.name} did not terminate gracefully")
        
        # Wait for writer worker with timeout
        self.writer_worker.join(timeout=10)
        if self.writer_worker.is_alive():
            logger.warning("Writer worker did not terminate gracefully")
        
        # Clear queues
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except Empty:
                break
                
        while not self.writer_queue.empty():
            try:
                self.writer_queue.get_nowait()
            except Empty:
                break
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear the model's internal cache
        if hasattr(self.model, 'decoder'):
            self.model.decoder.kv_cache = {}
        
        logger.info("Transcription pool stopped")