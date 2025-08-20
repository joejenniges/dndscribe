import logging
from discord import Bot
from discord.sinks import Sink, Filters, default_filters
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import wave
import os
import asyncio
from transcriptionpool import TranscriptionPool
logger = logging.getLogger('dndscribe')

@dataclass
class AudioPacket:
    pcm: bytes
    timestamp: int
    sequence: int

class BufferedWaveSink(Sink):
    def __init__(self, bot: Bot, transcription_pool: TranscriptionPool, cleanup_wav: bool = False, ignored_users: set = None, filters=None, max_buffer_duration: float = 10.0):
        self.buffers: Dict[int, bytearray] = {}
        self.timestamps: Dict[int, datetime] = {}
        self.active = True
        self.last_packet_time: Dict[int, float] = {}
        self.last_timestamp: Dict[int, int] = {}
        self.last_sequence: Dict[int, int] = {}
        self.expected_packet_duration = 20  # Discord sends packets every 20ms
        self.SILENCE_DURATION = 1.0
        self.MAX_BUFFER_DURATION = max_buffer_duration  # Maximum duration in seconds before forcing transcription
        self.transcription_pool = transcription_pool
        self.lock = threading.Lock()
        self.ignored_users = ignored_users
        self.bot = bot
        self.user_names: Dict[int, str] = {}
        
        # Audio format constants
        self.SAMPLE_RATE = 48000
        self.CHANNELS = 2
        self.BYTES_PER_SAMPLE = 2  # 16-bit audio = 2 bytes per sample
        
        # PCM frame size calculations
        self.SAMPLES_PER_MS = self.SAMPLE_RATE / 1000
        self.SAMPLES_PER_PACKET = int(self.SAMPLES_PER_MS * self.expected_packet_duration)
        self.BYTES_PER_PACKET = self.SAMPLES_PER_PACKET * self.CHANNELS * self.BYTES_PER_SAMPLE

        if filters is None:
            filters = default_filters
        self.filters = filters
        Filters.__init__(self, **self.filters)
        
        self.encoding = "pcm"
        self.vc = None
        self.audio_data = {}
    
    def wants_opus(self) -> bool:
        return False
    
    def cleanup(self) -> None:
        """Called by discord when the sink is being cleaned up"""
        if not self.finished:
           self.finished = True
           self.stop()

    
    def get_user_name(self, user_id: int) -> str:
        if user_id in self.user_names:
            return self.user_names[user_id]
        for member in self.bot.get_all_members():
            if member.id == user_id:
                self.user_names[user_id] = member.name
                return member.name
        return "Unknown User"

    def _start_transcription(self, user_id: int, packets: List[bytes], timestamp: datetime) -> None:
        """Queue audio buffer for transcription with improved timestamp handling"""
        if not packets:
            logger.warning(f"No packets to process for user {user_id}")
            return
        
        username = self.get_user_name(user_id)
        self.transcription_pool.add_transcription_task(packets, user_id, username, timestamp)
    
    def _check_silence(self, user_id: int) -> bool:
        """Check if there's been a significant gap since the last packet"""
        current_time = datetime.now().timestamp()
        
        if user_id not in self.last_packet_time:
            self.last_packet_time[user_id] = current_time
            return False

        time_since_last_packet = current_time - self.last_packet_time[user_id]
        return time_since_last_packet > self.SILENCE_DURATION
    
    def _update_last_packet_time(self, user_id: int) -> None:
        """Update the last packet time for a user"""
        self.last_packet_time[user_id] = datetime.now().timestamp()

    def format_audio(self, audio):
        return

    def _calculate_buffer_duration(self, user_id: int) -> float:
        """Calculate the current duration of the buffer in seconds"""
        if user_id not in self.timestamps:
            return 0.0
        
        current_time = datetime.now()
        buffer_start = self.timestamps[user_id]
        duration = (current_time - buffer_start).total_seconds()
        return duration

    def _should_process_buffer(self, user_id: int) -> bool:
        """Check if the buffer should be processed based on silence or duration"""
        if user_id not in self.buffers or not self.buffers[user_id]:
            return False
            
        # Check for silence
        if self._check_silence(user_id):
            return True
            
        # Check for maximum duration
        if self._calculate_buffer_duration(user_id) >= self.MAX_BUFFER_DURATION:
            return True
            
        return False

    def write(self, data: bytes, user_id) -> None:
        if user_id in self.ignored_users:
            return
        
        if self.finished:
            return

        if user_id not in self.buffers:
            self.buffers[user_id] = bytearray()

        if user_id not in self.timestamps or self.timestamps[user_id] is None:
            self.timestamps[user_id] = datetime.now()

        # Update the last packet time when we receive new data
        self._update_last_packet_time(user_id)
        self.buffers[user_id].extend(data)

        # Check all users for silence
        for user in list(self.buffers.keys()):  # Create a copy of keys to avoid modification during iteration
            if self._should_process_buffer(user):
                packets_to_process = self.buffers[user].copy()
                timestamp = self.timestamps[user]
                self.buffers[user] = bytearray()
                self.timestamps[user] = None
                self._start_transcription(user, packets_to_process, timestamp)


    def stop(self) -> None:
        """Stop the sink and process remaining buffers"""
        logger.info("Processing remaining buffers...")
        
        # Process any remaining buffers
        buffers_to_process = {
            user_id: packets.copy() 
            for user_id, packets in self.buffers.items()
            if packets
        }
        
        # Clear buffers before processing to prevent new data
        self.buffers.clear()
        
        # Process the copied buffers
        for user_id, packets in buffers_to_process.items():
            if packets:
                self._start_transcription(user_id, packets, self.timestamps[user_id])
        
        # Stop the transcription pool
        if self.transcription_pool:
            self.transcription_pool.stop()
        
        logger.info("Finished processing buffers")