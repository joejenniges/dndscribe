import discord
from discord import app_commands, opus
from discord.ext import commands, voice_recv
from discord.opus import Decoder as OpusDecoder
from discord.ext.voice_recv import WaveSink
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
from transcriptionpoolnew import TranscriptionPool
from bufferwavesinknew import BufferedWaveSink
from subprocess import run
import subprocess
import ffmpeg
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
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

IGNORED_USERS_FILE = "ignored.txt"
ignored_users = set() # Connect to Ollama container

def load_ignored_users():
    """Load ignored users from file into memory"""
    global ignored_users
    try:
        if os.path.exists(IGNORED_USERS_FILE):
            with open(IGNORED_USERS_FILE, 'r') as f:
                ignored_users = set(map(int, f.read().splitlines()))
            logger.info(f"Loaded {len(ignored_users)} ignored users")
    except Exception as e:
        logger.error(f"Error loading ignored users: {e}")
        ignored_users = set()

def save_ignored_users():
    """Save ignored users from memory to file"""
    try:
        with open(IGNORED_USERS_FILE, 'w') as f:
            for user_id in ignored_users:
                f.write(f"{user_id}\n")
        logger.info(f"Saved {len(ignored_users)} ignored users")
    except Exception as e:
        logger.error(f"Error saving ignored users: {e}")

# Load ignored users at startup
load_ignored_users()
# Create global transcription pool
transcription_pool = TranscriptionPool(num_workers=2)


@bot.event
async def on_ready():
    await tree.sync(guild=discord.Object(id=GUILD_ID))
    logger.info(f'{bot.user.name} has connected to Discord!')


@tree.command(name="sync", description="Sync bot commands globally", guild=discord.Object(id=GUILD_ID))
async def sync(interaction: discord.Interaction):
    await tree.sync()
    await interaction.response.send_message("Synced", ephemeral=True)

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

        # wav_sink = BufferedWaveSink(cleanup_wav=False, transcription_pool=transcription_pool)
        wav_sink = WaveSink("recordings/output.wav")

        vc.listen(wav_sink)
        voice_clients[interaction.guild_id].wav_sink = wav_sink
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

@tree.command(name="ignore", description="Ignore a user's audio")
@app_commands.describe(user="The user to ignore")
async def ignore(interaction: discord.Interaction, user: discord.Member):
    global ignored_users
    if user.id in ignored_users:
        await interaction.response.send_message(f"Already ignoring {user.display_name}", ephemeral=True)
        return
    
    ignored_users.add(user.id)
    save_ignored_users()
    await interaction.response.send_message(f"Now ignoring {user.display_name}", ephemeral=True)

@tree.command(name="unignore", description="Stop ignoring a user's audio")
@app_commands.describe(user="The user to stop ignoring")
async def unignore(interaction: discord.Interaction, user: discord.Member):
    global ignored_users
    if user.id not in ignored_users:
        await interaction.response.send_message(f"Wasn't ignoring {user.display_name}", ephemeral=True)
        return
    
    ignored_users.remove(user.id)
    save_ignored_users()
    await interaction.response.send_message(f"No longer ignoring {user.display_name}", ephemeral=True)

@tree.command(name="list_ignored", description="List all ignored users")
async def list_ignored(interaction: discord.Interaction):
    if not ignored_users:
        await interaction.response.send_message("No users are being ignored", ephemeral=True)
        return
    
    ignored_members = []
    for user_id in ignored_users:
        member = interaction.guild.get_member(user_id)
        if member:
            ignored_members.append(f"• {member.display_name} ({user_id})")
        else:
            ignored_members.append(f"• Unknown User ({user_id})")
    
    message = "Currently ignored users:\n" + "\n".join(ignored_members)
    await interaction.response.send_message(message, ephemeral=True)

@bot.event
async def on_close():
    transcription_pool.stop()

bot.run(TOKEN)

# template = ChatPromptTemplate([
#                 ("system", "You are an AI assistant who is an expert in Dungeons and Dragons. You are able to recognize the names of the characters in the game and the names of the places they are in. You are also able to recognize the names of the monsters they are fighting and the names of the items they are using."),
# ])

# model = OllamaLLM(model="llama2-uncensored", base_url="http://localhost:11434")
# chain = template | model
# print(chain.invoke({"question": "What is the capital of France?"}))

# model = whisper.load_model("base", download_root="models")
# result = model.transcribe("recordings/215686800348413953/debug/2025-02-25-21-52-51.wav")
# print(result["text"])

def load_wave_file(filename: str) -> bytes:
    try:
        with wave.open(filename, 'rb') as wav_file:
            return wav_file.readframes(wav_file.getnframes())
    except Exception as e:
        logger.error(f"Error loading wave file {filename}: {e}")
        return None

# wav_bytes = load_wave_file("recordings/215686800348413953/debug/2025-02-25-21-52-51.wav")

# print(f"Size of wav_bytes: {len(wav_bytes)}")


# def load_audio2(file_bytes: bytes, sr: int = 16_000) -> np.ndarray:
#     ffmpeg_command = [
#         "ffmpeg",
#         "-f", "s16le",         # Input format: signed 16-bit little-endian
#         "-ar", "48000",        # Input sample rate: 48000 Hz
#         "-ac", "2",            # Input channels: 2 (Stereo)
#         "-i", "pipe:0",        # Input from stdin (pipe)
#         "-f", "s16le",         # Output format: signed 16-bit little-endian
#         "-ac", "1",            # Output channels: 1 (Mono)
#         "-ar", "16000",        # Output sample rate: 16000 Hz
#         "pipe:1"               # Output to stdout (pipe)
#     ]

#     process = subprocess.Popen(
#         ffmpeg_command,
#         stdin=subprocess.PIPE,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE
#     )

#     out, err = process.communicate(input=file_bytes)

#     return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    

# wave_bytes, metadata = TranscriptionPool.load_raw_debug_audio("recordings/215686800348413953/debug/2025-02-26-13-55-50.raw")
# out = load_audio2(wave_bytes)
# print(len(wave_bytes))
# print(len(out))

# # out = np.frombuffer(wav_bytes, np.int16).flatten().astype(np.float32) / 32768.0 

# model = whisper.load_model("base", download_root="models")
# result = model.transcribe(out, language="en", fp16=False)
# print(result["text"])
