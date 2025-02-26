import io
import time
import discord
from discord.ext.voice_recv import AudioSink, VoiceData, WaveSink
from discord.ext.voice_recv.silence import SilenceGenerator
from typing import Dict, Optional
import numpy as np
import wave # mandatory for those who wonder


# def add_silence_to_wav(input_data: bytes, silence_duration: float) -> bytes:
#     audio = AudioSegment.from_wav(io.BytesIO(input_data))
#     silence = AudioSegment.silent(duration=int(silence_duration * 1000))  # pydub uses milliseconds
#     final_audio = silence + audio
#     output_buffer = io.BytesIO()
#     final_audio.export(output_buffer, format="wav")
#     return output_buffer.getvalue()

class MultiAudioImprovedWithSilenceSink(AudioSink):
    def __init__(self):
        super().__init__()
        self.user_sinks: Dict[int, WaveSink] = {}
        self.user_buffers: Dict[int, io.BytesIO] = {}
        self.silence_generators: Dict[int, SilenceGenerator] = {}
        self.start_time = time.perf_counter_ns()
        self.first_packet_time: Dict[int, int] = {}

    def _get_or_create_sink(self, user_id: int) -> WaveSink:
        if user_id not in self.user_sinks:
            buffer = io.BytesIO()
            sink = WaveSink(buffer)
            self.user_sinks[user_id] = sink
            self.user_buffers[user_id] = buffer
            # self.silence_generators[user_id] = SilenceGenerator(sink.write)
            # self.silence_generators[user_id].start()
        return self.user_sinks[user_id]

    def wants_opus(self) -> bool:
        return False

    def write(self, user: Optional[discord.User], data: VoiceData) -> None:
        if user is None:
            return

        sink = self._get_or_create_sink(user.id)
        silence_gen = self.silence_generators[user.id]
        
        if user.id not in self.first_packet_time:
            self.first_packet_time[user.id] = time.perf_counter_ns()

        silence_gen.push(user, data.packet)
        sink.write(user, data)

    def cleanup(self) -> None:
        for silence_gen in self.silence_generators.values():
            silence_gen.stop()
        self.user_sinks.clear()
        self.user_buffers.clear()
        self.silence_generators.clear()

    def get_user_audio(self, user_id: int) -> Optional[bytes]:
        if user_id in self.user_buffers:
            buffer = self.user_buffers[user_id]
            buffer.seek(0)
            audio_data = buffer.read()
            return audio_data
        return None

    def get_initial_silence_duration(self, user_id: int) -> float:
        if user_id in self.first_packet_time:
            return (self.first_packet_time[user_id] - self.start_time) / 1e9  # nano to sec
        return 0.0

    def mix_audio(self, audio_data_dict: Dict[int, bytes]) -> Optional[bytes]:
        audio_arrays = []
        sample_rate = 0
        num_channels = 0
        sample_width = 0

        for audio_data in audio_data_dict.values():
            if len(audio_data) <= 44:
                continue
            
            with wave.open(io.BytesIO(audio_data), 'rb') as wav_file:
                params = wav_file.getparams()
                sample_rate = params.framerate
                num_channels = params.nchannels
                sample_width = params.sampwidth

                frames = wav_file.readframes(params.nframes)
                audio_array = np.frombuffer(frames, dtype=np.int16)
                audio_arrays.append(audio_array)

        if not audio_arrays:
            return None

        max_length = max(len(arr) for arr in audio_arrays)
        padded_audio_arrays = [np.pad(arr, (0, max_length - len(arr)), 'constant') for arr in audio_arrays]
        mixed_audio = np.mean(padded_audio_arrays, axis=0).astype(np.int16)

        output_buffer = io.BytesIO()
        with wave.open(output_buffer, 'wb') as output_wav:
            output_wav.setnchannels(num_channels)
            output_wav.setsampwidth(sample_width)
            output_wav.setframerate(sample_rate)
            output_wav.writeframes(mixed_audio.tobytes())
        
        output_buffer.seek(0)
        return output_buffer.read()