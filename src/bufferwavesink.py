import discord
from discord import app_commands, opus
from discord.ext import commands, voice_recv
from discord.opus import Decoder as OpusDecoder
from discord.ext.voice_recv import VoiceData, AudioSink
import os
import platform
import wave
import numpy as np
import whisper
from datetime import datetime
import threading
from typing import Optional, List
import logging
import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import asyncio
from queue import Queue, Empty
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from transcriptionpool import TranscriptionPool
logger = logging.getLogger('dndscribe')

@dataclass
class AudioPacket:
    pcm: bytes
    timestamp: int

class BufferedWaveSink(AudioSink):
    def __init__(self, transcription_pool: TranscriptionPool, cleanup_wav: bool = False):
        super().__init__()
        self.buffers: dict[int, List[AudioPacket]] = {}
        self.active = True
        self.last_packet_time: dict[int, float] = {}
        self.SILENCE_DURATION = 3.0
        self.transcription_pool = transcription_pool
        self.lock = threading.Lock()  # Add lock for thread safety
    
    def wants_opus(self) -> bool:
        return False
    
    def cleanup(self) -> None:
        """Called by discord when the sink is being cleaned up"""
        if self.active:  # Only stop if we haven't already
            self.stop()

    def _start_transcription(self, user_id: int, packets: List[AudioPacket]) -> None:
        """Queue sorted audio buffer for transcription"""
        # Sort packets by timestamp
        sorted_packets = sorted(packets, key=lambda p: p.timestamp)

        # Combine PCM data in order with accurate silence padding
        combined_buffer = bytearray()
        previous_timestamp = None
        packet_duration_ms = 20  # Discord sends packets every 20 ms
        sample_rate = 48000
        channels = 2  # Stereo

        for packet in sorted_packets:
            # Skip duplicate timestamps
            if packet.timestamp == previous_timestamp:
                logger.debug(f"Duplicate timestamp detected: {packet.timestamp} for user {user_id}, skipping packet")
                continue
            
            # Calculate time difference between current and previous packet
            if previous_timestamp is not None:
                time_diff_ms = packet.timestamp - previous_timestamp
                
                # Only add silence if the gap is reasonable (e.g., 20 ms < gap < 100 ms)
                if packet_duration_ms < time_diff_ms < 100:
                    gap_duration_ms = time_diff_ms - packet_duration_ms
                    num_silence_samples = int(gap_duration_ms * (sample_rate / 1000) * channels)
                    
                    # Create silence buffer
                    silence_buffer = bytearray([0] * num_silence_samples * 2)  # 2 bytes per sample
                    combined_buffer.extend(silence_buffer)
                    
                    logger.info(f"Timestamp: {packet.timestamp}, Time Diff: {time_diff_ms} ms, Silence Samples: {num_silence_samples} for user {user_id}")
                else:
                    logger.info(f"Skipping abnormal time diff: {time_diff_ms} ms at timestamp {packet.timestamp} for user {user_id}")

            # Append the current packet
            combined_buffer.extend(packet.pcm)
            previous_timestamp = packet.timestamp

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.transcription_pool.add_task(bytes(combined_buffer), user_id, timestamp)

        
    def _check_silence(self, user_id: int) -> bool:
        """Check if there's been a significant gap since the last packet"""
        current_time = datetime.now().timestamp()
        
        if user_id not in self.last_packet_time:
            self.last_packet_time[user_id] = current_time
            return False

        time_since_last_packet = current_time - self.last_packet_time[user_id]
        self.last_packet_time[user_id] = current_time
        
        return time_since_last_packet > self.SILENCE_DURATION

    def write(self, user_id: int, data: VoiceData) -> None:
        if not self.active:
            return

        with self.lock:  # Protect dictionary access
            # Initialize buffer if needed
            if user_id not in self.buffers:
                self.buffers[user_id] = []
            
            # Check for silence before writing
            if self._check_silence(user_id):
                logger.info(f"Detected silence for user {user_id}, processing buffer")
                if len(self.buffers[user_id]) > 0:
                    packets_to_process = self.buffers[user_id]
                    self.buffers[user_id] = []  # Clear buffer
                    self._start_transcription(user_id, packets_to_process)
            
            # Create and store packet with timestamp
            packet = AudioPacket(
                pcm=data.pcm,
                timestamp=data.packet.timestamp
            )
            self.buffers[user_id].append(packet)

    def stop(self) -> None:
        """Stop the sink and process remaining buffers"""
        self.active = False
        logger.info("Processing remaining buffers...")
        
        # Make a copy of the buffers to process
        with self.lock:
            buffers_to_process = {
                user_id: packets.copy() 
                for user_id, packets in self.buffers.items()
                if packets
            }
            self.buffers.clear()
        
        # Process the copied buffers
        for user_id, packets in buffers_to_process.items():
            if packets:
                self._start_transcription(user_id, packets)
        
        logger.info("Finished processing buffers")