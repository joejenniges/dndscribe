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

dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)
logger = logging.getLogger('dndscribe')

TOKEN = os.getenv('TOKEN')
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
bot = discord.Client(intents=intents)

if platform.system() == 'Darwin':  # Check if on macOS
    opus.load_opus("/opt/homebrew/lib/libopus.dylib")
else:
    opus._load_default()

tree = app_commands.CommandTree(bot)
GUILD_ID = 774020122649821204

voice_clients: dict[int, voice_recv.VoiceRecvClient] = {}
recordings = {}
transcription_threads = {}

model = whisper.load_model(name="base", download_root="models")

class BufferedWaveSink:
    def __init__(self, path: str = "recordings"):
        self.path = path
        self.buffers: dict[int, list[bytes]] = {}
        self.current_files: dict[int, tuple[wave.Wave_write, str]] = {}
        self.transcription_threads: dict[int, list[threading.Thread]] = {}
        self.active = True  # Flag to track if sink is active
        os.makedirs(path, exist_ok=True)
        
    def _create_new_file(self, user_id: int) -> tuple[wave.Wave_write, str]:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        user_folder = f"{self.path}/{user_id}"
        os.makedirs(user_folder, exist_ok=True)
        filename = f"{user_folder}/{timestamp}.wav"
        
        wav_file = wave.open(filename, 'wb')
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(48000)
        logger.info(f"Created new wav file: {filename}")
        return wav_file, filename

    def _start_transcription(self, filename: str, user_id: int) -> None:
        """Start transcription of a file in a new thread"""
        # Clean up completed threads first
        if user_id in self.transcription_threads:
            self.transcription_threads[user_id] = [
                t for t in self.transcription_threads[user_id] 
                if t.is_alive()
            ]
        
        thread = threading.Thread(
            target=self._transcribe_file,
            args=(filename, user_id),
            name=f"transcribe-{os.path.basename(filename)}"
        )
        thread.daemon = True  # Make thread daemon so it doesn't prevent program exit
        thread.start()
        
        if user_id not in self.transcription_threads:
            self.transcription_threads[user_id] = []
        self.transcription_threads[user_id].append(thread)
        logger.info(f"Started transcription thread for {filename}")

    def _transcribe_file(self, audio_path: str, user_id: int) -> None:
        try:
            logger.info(f"Starting transcription of {audio_path}")
            result = model.transcribe(audio_path)
            
            user_folder = f"{self.path}/{user_id}"
            transcription_file = f"{user_folder}/transcription.txt"
            
            timestamp = os.path.basename(audio_path).replace('.wav', '')
            with open(transcription_file, "a") as f:
                f.write(f"\n[{timestamp}]\n")
                f.write(result["text"].strip())
                f.write("\n")
            
            logger.info(f"Added transcription to {transcription_file}")
            
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}", exc_info=True)

    def _close_file(self, user_id: int) -> Optional[str]:
        """Close the current file for a user and return its filename"""
        if user_id in self.current_files:
            wav_file, filename = self.current_files[user_id]
            try:
                wav_file.close()
                logger.info(f"Closed file: {filename}")
                del self.current_files[user_id]
                return filename
            except Exception as e:
                logger.error(f"Error closing file: {e}", exc_info=True)
        return None
            
    def write(self, user_id: int, pcm_data: bytes) -> None:
        if user_id not in self.current_files:
            self.current_files[user_id] = self._create_new_file(user_id)
        
        wav_file, filename = self.current_files[user_id]
        try:
            wav_file.writeframes(pcm_data)
        except Exception as e:
            logger.error(f"Error writing to {filename}: {e}", exc_info=True)
            self.rotate_file(user_id)
            
    def rotate_file(self, user_id: int) -> None:
        """Close current file and create a new one"""
        filename = self._close_file(user_id)
        if filename:
            self.current_files[user_id] = self._create_new_file(user_id)
            self._start_transcription(filename, user_id)

    def stop(self) -> None:
        """Stop the sink and cleanup"""
        self.active = False
        self.cleanup()

    def cleanup(self) -> None:
        """Close all open files and wait for transcriptions"""
        logger.info("Starting cleanup...")
        
        # Close all files and start their transcriptions
        for user_id in list(self.current_files.keys()):
            filename = self._close_file(user_id)
            if filename:
                self._start_transcription(filename, user_id)
        
        if self.active:  # Only wait for threads if we're doing a normal cleanup
            # Wait for all transcription threads to complete
            logger.info("Waiting for transcription threads to complete...")
            for user_id, user_threads in self.transcription_threads.items():
                for thread in user_threads:
                    if thread.is_alive():
                        thread.join(timeout=30)  # Add timeout to prevent hanging
                    if thread.is_alive():
                        logger.warning(f"Thread {thread.name} did not complete in time")
                logger.info(f"Completed all transcriptions for user {user_id}")
        
        self.transcription_threads.clear()
        logger.info("Cleanup complete")

@bot.event
async def on_ready():
    logger.info(f'{bot.user.name} has connected to Discord!')


@tree.command(name="join", description="Join your voice channel")
async def join(interaction: discord.Interaction):
    if interaction.user.voice:
        channel = interaction.user.voice.channel
        vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
        voice_clients[interaction.guild_id] = vc
        await interaction.response.send_message(f"Joined {channel}", ephemeral=True)
    else:
        await interaction.response.send_message("You are not in a voice channel", ephemeral=True)

@tree.command(name="leave", description="Leave the voice channel")
async def leave(interaction: discord.Interaction):
    if interaction.guild_id in voice_clients:
        vc = voice_clients[interaction.guild_id]
        if hasattr(vc, 'wav_sink'):
            vc.stop_listening()  # Stop recording first
            vc.wav_sink.stop()   # Stop and cleanup the sink
        await vc.disconnect()
        del voice_clients[interaction.guild_id]
        await interaction.response.send_message("Left the voice channel", ephemeral=True)
    else:
        await interaction.response.send_message("I'm not in a voice channel", ephemeral=True)

@tree.command(name="transcribe", description="Start recording and transcribing")
async def record(interaction: discord.Interaction):
    if interaction.user.voice:
        channel = interaction.user.voice.channel
        vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
        voice_clients[interaction.guild_id] = vc
    else:
        await interaction.response.send_message("You are not in a voice channel", ephemeral=True)
        return

    if interaction.guild_id in voice_clients:
        vc = voice_clients[interaction.guild_id]
        await interaction.response.send_message("Recording...", ephemeral=True)

        wav_sink = BufferedWaveSink()
        
        def callback(user: Optional[discord.User], data: voice_recv.VoiceData):
            if user:
                wav_sink.write(user.id, data.pcm)
                # Rotate file every 30 seconds (roughly)
                if data.packet.timestamp % (48000 * 30) < 960:  # 48kHz * 30sec, check within one packet
                    wav_sink.rotate_file(user.id)

        vc.listen(voice_recv.BasicSink(callback))
        voice_clients[interaction.guild_id].wav_sink = wav_sink  # Store reference
        await interaction.followup.send("Recording started", ephemeral=True)
    else:
        await interaction.response.send_message("I am not in a voice channel", ephemeral=True)

@tree.command(name="stop", description="Stop recording")
async def stop(interaction: discord.Interaction):
    if interaction.guild_id in voice_clients:
        vc = voice_clients[interaction.guild_id]
        if hasattr(vc, 'wav_sink'):
            vc.stop_listening()
            vc.wav_sink.stop()  # Use stop instead of cleanup
        await interaction.response.send_message("Recording stopped", ephemeral=True)
    else:
        await interaction.response.send_message("I'm not recording", ephemeral=True)

bot.run(TOKEN)