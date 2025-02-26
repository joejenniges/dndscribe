import discord
from discord import app_commands, opus
from discord.ext import commands, voice_recv
from discord.opus import Decoder as OpusDecoder
import os
import platform
import wave
import numpy as np
import whisper
from datetime import datetime
import threading
from typing import Optional
import logging
import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import asyncio
from queue import Queue, Empty
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import subprocess
import torch 
from scipy.signal import resample
import ffmpeg

logger = logging.getLogger('dndscribe')
options = whisper.DecodingOptions(language="en")

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
        self.workers = [
            threading.Thread(target=self._worker_loop, name=f"transcriber-{i}")
            for i in range(num_workers)
        ]
        for worker in self.workers:
            worker.daemon = True
            worker.start()
        logger.info(f"Started {num_workers} transcription workers")

    def _process_audio(self, file_bytes: bytes, sr: int = 16_000) -> np.ndarray:
        ffmpeg_command = [
            "ffmpeg",
            "-f", "s16le",         # Input format: signed 16-bit little-endian
            "-ar", "48000",        # Input sample rate: 48000 Hz
            "-ac", "2",            # Input channels: 2 (Stereo)
            "-i", "pipe:0",        # Input from stdin (pipe)
            "-f", "s16le",         # Output format: signed 16-bit little-endian
            "-ac", "1",            # Output channels: 1 (Mono)
            "-ar", "16000",        # Output sample rate: 16000 Hz
            "pipe:1"               # Output to stdout (pipe)
        ]

        process = subprocess.Popen(
            ffmpeg_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        out, err = process.communicate(input=file_bytes)

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    def _worker_loop(self):
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
        """Save raw audio to wav file and binary file for debugging"""
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

    def _process_task(self, task: TranscriptionTask):
        try:
            # Save original audio if debug is enabled
            if self.debug_audio:
                self._save_debug_audio(task.audio_data, task.user_id, task.timestamp)
            
            # Convert audio bytes to numpy array in whisper format
            audio = self._process_audio(task.audio_data)
            
            # Save resampled audio if debug is enabled
            if self.debug_audio:
                self._save_debug_resampled_audio(audio, task.user_id, task.timestamp)

            
            return
            
            # Transcribe the audio
            logger.info(f"Starting transcription for user {task.user_id}")
            result = self.model.transcribe(audio, fp16=False, language="en")
            
            # Write transcription to file
            user_folder = f"recordings/{task.user_id}"
            os.makedirs(user_folder, exist_ok=True)
            transcription_file = f"{user_folder}/transcription.txt"
            
            with open(transcription_file, "a") as f:
                f.write(f"[{task.timestamp}] ")
                f.write(result["text"].strip())
                f.write("\n")
            
            logger.info(f"Added transcription for user {task.user_id}")
            
        except Exception as e:
            logger.error(f"Error transcribing audio for user {task.user_id}: {e}", exc_info=True)

    def add_task(self, audio_data: bytes, user_id: int, timestamp: str):
        """Add a new transcription task to the queue"""
        task = TranscriptionTask(audio_data, user_id, timestamp)
        self.task_queue.put(task)
        logger.debug(f"Added transcription task for user {user_id}")

    def stop(self):
        """Stop the worker threads and wait for pending tasks"""
        self.active = False
        logger.info("Waiting for transcription tasks to complete...")
        self.task_queue.join()
        logger.info("All transcription tasks completed")