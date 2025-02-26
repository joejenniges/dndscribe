import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from discord.ext.voice_recv import AudioSink, VoiceData
import wave
import os

logger = logging.getLogger('dndscribe')

@dataclass
class AudioPacket:
    pcm: bytes
    timestamp: int
    sequence: int

class BufferedWaveSink(AudioSink):
    def __init__(self, transcription_pool, cleanup_wav: bool = False):
        super().__init__()
        self.buffers: Dict[int, List[AudioPacket]] = {}
        self.active = True
        self.last_packet_time: Dict[int, float] = {}
        self.last_timestamp: Dict[int, int] = {}
        self.last_sequence: Dict[int, int] = {}
        self.expected_packet_duration = 20  # Discord sends packets every 20ms
        self.SILENCE_DURATION = 3.0
        self.transcription_pool = transcription_pool
        self.lock = threading.Lock()
        
        # Audio format constants
        self.SAMPLE_RATE = 48000
        self.CHANNELS = 2
        self.BYTES_PER_SAMPLE = 2  # 16-bit audio = 2 bytes per sample
        
        # PCM frame size calculations
        self.SAMPLES_PER_MS = self.SAMPLE_RATE / 1000
        self.SAMPLES_PER_PACKET = int(self.SAMPLES_PER_MS * self.expected_packet_duration)
        self.BYTES_PER_PACKET = self.SAMPLES_PER_PACKET * self.CHANNELS * self.BYTES_PER_SAMPLE
        
        # For debugging
        self.debug_dir = "debug_packets"
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def wants_opus(self) -> bool:
        return False
    
    def cleanup(self) -> None:
        """Called by discord when the sink is being cleaned up"""
        if self.active:
            self.stop()

    def _normalize_timestamps(self, packets: List[AudioPacket]) -> List[Tuple[AudioPacket, int]]:
        """
        Convert real timestamps to normalized time positions based on sequence numbers.
        Returns packets with their expected positions in the audio stream.
        """
        if not packets:
            return []
        
        # Sort by sequence first
        sorted_packets = sorted(packets, key=lambda p: p.sequence)
        normalized_packets = []
        
        # Calculate base position from the first packet
        base_sequence = sorted_packets[0].sequence
        
        for packet in sorted_packets:
            # Position based on sequence number (each packet should be 20ms)
            position_ms = (packet.sequence - base_sequence) * self.expected_packet_duration
            normalized_packets.append((packet, position_ms))
            
        return normalized_packets

    def _start_transcription(self, user_id: int, packets: List[AudioPacket]) -> None:
        """Queue audio buffer for transcription with improved timestamp handling"""
        if not packets:
            logger.warning(f"No packets to process for user {user_id}")
            return
            
        try:
            # Normalize timestamps to create a consistent timeline
            normalized_packets = self._normalize_timestamps(packets)
            
            # Calculate total expected duration and create an empty buffer
            total_duration_ms = normalized_packets[-1][1] + self.expected_packet_duration
            total_samples = int(self.SAMPLES_PER_MS * total_duration_ms) * self.CHANNELS
            output_buffer = bytearray(total_samples * self.BYTES_PER_SAMPLE)
            
            # Fill the buffer with each packet at its appropriate position
            for packet, position_ms in normalized_packets:
                # Calculate position in bytes
                position_samples = int(self.SAMPLES_PER_MS * position_ms) * self.CHANNELS
                position_bytes = position_samples * self.BYTES_PER_SAMPLE
                
                # Ensure position is valid
                if position_bytes >= len(output_buffer):
                    logger.warning(f"Packet position {position_bytes} exceeds buffer size {len(output_buffer)}")
                    continue
                
                # Ensure the packet isn't too large
                packet_size = min(len(packet.pcm), self.BYTES_PER_PACKET)
                
                if packet_size + position_bytes > len(output_buffer):
                    # Trim packet to fit
                    packet_data = packet.pcm[:len(output_buffer) - position_bytes]
                else:
                    packet_data = packet.pcm[:packet_size]
                
                # Copy packet data into the buffer at the appropriate position
                output_buffer[position_bytes:position_bytes + len(packet_data)] = packet_data
            
            # Generate a timestamp for the file
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            logger.info(f"Sending {len(output_buffer)} bytes of audio for user {user_id}, {total_duration_ms}ms")
            
            # Save a debug WAV if needed
            if logger.level <= logging.DEBUG:
                self._save_debug_buffer(user_id, output_buffer, timestamp)
            
            # Send to transcription
            self.transcription_pool.add_task(bytes(output_buffer), user_id, timestamp)
            
        except Exception as e:
            logger.error(f"Error processing audio for user {user_id}: {e}", exc_info=True)
    
    def _save_debug_buffer(self, user_id: int, buffer: bytearray, timestamp: str):
        """Save audio buffer to a WAV file for debugging"""
        try:
            debug_file = f"{self.debug_dir}/user_{user_id}_{timestamp}.wav"
            with wave.open(debug_file, 'wb') as wav_file:
                wav_file.setnchannels(self.CHANNELS)
                wav_file.setsampwidth(self.BYTES_PER_SAMPLE)
                wav_file.setframerate(self.SAMPLE_RATE)
                wav_file.writeframes(buffer)
            logger.debug(f"Saved debug audio to {debug_file}")
        except Exception as e:
            logger.error(f"Error saving debug audio: {e}")
    
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

        with self.lock:
            # Initialize buffer and tracking data if needed
            if user_id not in self.buffers:
                self.buffers[user_id] = []
                self.last_sequence[user_id] = -1
                self.last_timestamp[user_id] = 0
            
            # Check for silence before writing
            if self._check_silence(user_id):
                logger.info(f"Detected silence for user {user_id}, processing buffer")
                if len(self.buffers[user_id]) > 0:
                    packets_to_process = self.buffers[user_id].copy()
                    self.buffers[user_id] = []
                    self._start_transcription(user_id, packets_to_process)
            
            # Calculate the next sequence number
            next_sequence = self.last_sequence[user_id] + 1
            
            # Compare with previous timestamp to detect jumps
            if self.last_timestamp[user_id] > 0:
                timestamp_diff = data.packet.timestamp - self.last_timestamp[user_id]
                
                # A normal packet is about 20ms (960 samples @ 48kHz)
                # If the difference is significantly more, log a jump but don't bail
                if abs(timestamp_diff) > 1000:  # More than 1000 timestamp units
                    logger.warning(f"Large timestamp jump for user {user_id}: {self.last_timestamp[user_id]} -> {data.packet.timestamp}")
                    
                    # Adjust sequence number based on timestamp jump
                    # Each packet is ~20ms, so calculate how many we missed
                    if timestamp_diff > 0:  # Forward jump
                        missed_packets = int(timestamp_diff / 960) - 1  # 960 samples is ~20ms at 48kHz
                        next_sequence += missed_packets
            
            # Update tracking info
            self.last_timestamp[user_id] = data.packet.timestamp
            self.last_sequence[user_id] = next_sequence
            
            # Create and store packet with timestamp and sequence
            packet = AudioPacket(
                pcm=data.pcm,
                timestamp=data.packet.timestamp,
                sequence=next_sequence
            )
            self.buffers[user_id].append(packet)
            
            # If the buffer gets too large, process older packets
            max_buffer_size = 300  # ~6 seconds at 20ms packets
            if len(self.buffers[user_id]) > max_buffer_size:
                logger.info(f"Buffer size exceeded for user {user_id}, processing oldest {max_buffer_size//2} packets")
                packets_to_process = self.buffers[user_id][:max_buffer_size//2]
                self.buffers[user_id] = self.buffers[user_id][max_buffer_size//2:]
                self._start_transcription(user_id, packets_to_process)

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