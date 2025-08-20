import discord
from discord import opus, ApplicationContext, VoiceState, VoiceChannel, VoiceClient
from discord.sinks import AudioData
import os
import platform
import wave
import numpy as np
import whisper
from datetime import datetime
import threading
import time
from typing import Optional
import logging
import dotenv
import asyncio
from queue import Queue, Empty
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from transcriptionpool import TranscriptionPool
from bufferwavesink import BufferedWaveSink
# from summarizer import Summarizer
from subprocess import run
import psutil
import traceback
import sys
from memory_profiler import profile
import gc
import atexit
import signal

dotenv.load_dotenv()


# Configure logging with more detailed format

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler(f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")  # Output to file
    ]
)
logger = logging.getLogger('dndscribe')

atexit.register(lambda: logger.info("Python process is exiting..."))

def log_memory_usage():
    """Log current memory usage of the process"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    active_threads = threading.active_count()
    logger.info(f"Active Threads: {active_threads}")

def setup_exception_handlers():
    """Setup handlers for uncaught exceptions"""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Handle Ctrl+C gracefully
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        else:
            logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
            log_memory_usage()
    
    sys.excepthook = handle_exception
    
    # Handle asyncio exceptions
    def handle_async_exception(loop, context):
        logger.error(f"Async exception: {context}")
        log_memory_usage()
    
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(handle_async_exception)

# Setup exception handlers
setup_exception_handlers()

TOKEN = os.getenv('TOKEN')
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.members = True
bot = discord.Bot(intents=intents)

# Add periodic memory usage logging
async def log_memory_periodically():
    while True:
        log_memory_usage()
        await asyncio.sleep(300)  # Log every 5 minutes

@bot.event
async def on_error(event, *args, **kwargs):
    logger.error(f"Error in {event}:", exc_info=True)
    log_memory_usage()

if platform.system() == 'Darwin':  # Check if on macOS
    opus.load_opus("/opt/homebrew/lib/libopus.dylib")
else:
    opus._load_default()

GUILD_ID = int(os.getenv('GUILD_ID'))

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

def save_ignored_users(guild_id: int = None):
    """Save ignored users from memory to file"""
    # Only update the sink if guild_id is provided and exists in connections
    if guild_id is not None and guild_id in connections:
        vc = connections[guild_id]
        if vc:
            vc.sink.ignored_users = ignored_users
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

# Create global transcription pool
transcription_pool = TranscriptionPool(num_workers=1)

connections: dict[int, VoiceClient] = {}


@bot.event
async def on_ready():
    logger.info(f'{bot.user.name} has connected to Discord!')
    
    # Start periodic memory logging
    bot.loop.create_task(log_memory_periodically())
    log_memory_usage()

@bot.slash_command(name="sync", description="Sync global commands")
async def sync(ctx: ApplicationContext):
    await bot.sync_commands()
    logger.info("Synced commands")
    await ctx.respond("Synced", ephemeral=True)

@bot.slash_command(name="leave", description="Leave the voice channel")
async def leave(ctx: ApplicationContext):
    if ctx.guild.id in connections:
        vc = connections[ctx.guild.id]
        vc.stop_recording()
        del connections[ctx.guild.id]
        await ctx.respond("Left the voice channel", ephemeral=True)
    else:
        await ctx.respond("I'm not in a voice channel", ephemeral=True)

@bot.slash_command(name="transcribe", description="Start recording and transcribing")
async def record(ctx: ApplicationContext):
    voice: VoiceState = None
    voice = ctx.user.voice
    if voice is None:
        await ctx.respond("You are not in a voice channel", ephemeral=True)
        return
    
    vc: VoiceClient = await voice.channel.connect();
    connections.update({ctx.guild_id: vc})
    vc.start_recording(
        BufferedWaveSink(bot=bot, transcription_pool=transcription_pool, cleanup_wav=True, ignored_users=ignored_users),
        done_callback,
        ctx.channel
    )

    await ctx.respond("Recording...", ephemeral=True)

async def done_callback(sink: discord.sinks, channel: discord.TextChannel, *args): 
    await sink.vc.disconnect()
    # await channel.send(f"Finished recording")

@bot.slash_command(name="ignore", description="Ignore a user's audio")
async def ignore(ctx: ApplicationContext, user: discord.Member):
    global ignored_users
    if user.id in ignored_users:
        await ctx.respond(f"Already ignoring {user.display_name}", ephemeral=True)
        return
    
    ignored_users.add(user.id)
    save_ignored_users(ctx.guild_id)
    await ctx.respond(f"Now ignoring {user.display_name}", ephemeral=True)

@bot.slash_command(name="unignore", description="Stop ignoring a user's audio")
async def unignore(ctx: ApplicationContext, user: discord.Member):
    global ignored_users
    if user.id not in ignored_users:
        await ctx.respond(f"Wasn't ignoring {user.display_name}", ephemeral=True)
        return
    
    ignored_users.remove(user.id)
    save_ignored_users(ctx.guild_id)
    await ctx.respond(f"No longer ignoring {user.display_name}", ephemeral=True)

@bot.slash_command(name="list_ignored", description="List all ignored users")
async def list_ignored(ctx: ApplicationContext):
    if not ignored_users:
        await ctx.response.send_message("No users are being ignored", ephemeral=True)
        return
    
    ignored_members = []
    for user_id in ignored_users:
        member = ctx.guild.get_member(user_id)
        if member:
            ignored_members.append(f"• {member.display_name} ({user_id})")
        else:
            ignored_members.append(f"• Unknown User ({user_id})")
    
    message = "Currently ignored users:\n" + "\n".join(ignored_members)
    await ctx.respond(message, ephemeral=True)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}")
    if bot.is_closed():
        logger.info("Bot is already closed, forcing exit")
        sys.exit(0)
    else:
        logger.info("Closing bot...")
        # Cancel all running tasks
        for task in asyncio.all_tasks():
            task.cancel()
        # Run the close in the event loop
        asyncio.create_task(bot.close())

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@bot.event
async def on_close():
    logger.info("Bot is shutting down...")
    # Stop all voice connections
    for guild_id, vc in connections.items():
        if vc:
            vc.stop_recording()
            await vc.disconnect()
    
    # Stop transcription pool
    if transcription_pool:
        transcription_pool.stop()
    
    # Cancel all running tasks
    for task in asyncio.all_tasks():
        task.cancel()
    
    # Wait for tasks to complete with timeout
    pending = asyncio.all_tasks()
    if pending:
        try:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=2)
        except asyncio.TimeoutError:
            logger.warning("Some tasks did not complete in time")
    
    log_memory_usage()
    logger.info("Shutdown complete")

# Add memory profiling decorator to key functions
@profile
def run_bot():
    load_ignored_users()
    try:
        bot.run(TOKEN)
    except Exception as e:
        logger.error("Fatal error in bot:", exc_info=True)
        log_memory_usage()
        raise
    finally:
        # Ensure cleanup happens even if there's an error
        if not bot.is_closed():
            asyncio.run(bot.close())
        # Force exit after cleanup
        os._exit(0)  # Use os._exit to force immediate termination

if __name__ == "__main__":
    run_bot()