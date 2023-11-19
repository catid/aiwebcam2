# Service that uses an ASR model to recognize the language and transcribe the audio

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence
import samplerate
from multiprocessing import Process, Queue
import asyncio

# Model
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

# Logging
from utils import logger

class ASRService:
    def __init__(self, command_queue: Queue, response_queue: Queue):
        self.command_queue = command_queue
        self.response_queue = response_queue

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model = model.to_bettertransformer()
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )

        # ASR Engine
        self.pipe = pipe
        self.whisper_sample_rate = 16000
        self.fast_resampler = samplerate.Resampler('sinc_fastest', channels=1)
        self.best_resampler = samplerate.Resampler('sinc_best', channels=1)

    def normalize_audio(self, pcm_data_array, channels, input_sample_rate):
        normalized_array = []

        for pcm_data in pcm_data_array:
            # Squeeze out extra dimensions:
            if pcm_data.ndim > 1 and (pcm_data.shape[0] == 1 or pcm_data.shape[1] == 1):
                pcm_data = np.squeeze(pcm_data)

            # Check if audio is stereo (2 channels). If so, convert it to mono.
            if pcm_data.ndim > 1 and pcm_data.shape[1] == 2:
                pcm_data = np.mean(pcm_data, axis=1)
            elif pcm_data.ndim == 1 and channels == 2:
                # Left channel
                pcm_data = pcm_data[::2]

            if np.issubdtype(pcm_data.dtype, np.floating):
                # Convert to int16, PyDub doesn't work with float
                pcm_data = (pcm_data * np.iinfo(np.int16).max).astype(np.int16)

            normalized_array.append(pcm_data)

        pcm_data = np.concatenate(normalized_array)

        # Create an audio segment from the data
        sound = AudioSegment(
            pcm_data.tobytes(), 
            frame_rate=input_sample_rate,
            sample_width=pcm_data.dtype.itemsize, 
            channels=1
        )

        # Remove silence from audio clip
        chunks = split_on_silence(sound, 
            min_silence_len=500, # milliseconds
            silence_thresh=-30, # dB
            keep_silence = 300, # milliseconds
            seek_step=20 # Speed up by seeking this many ms at a time
        )

        if len(chunks) <= 0:
            return None

        non_silent_audio = sum(chunks)

        byte_string = non_silent_audio.raw_data
        int_audio = np.frombuffer(byte_string, dtype=np.int16)
        float_audio = int_audio.astype(np.float32) / np.iinfo(np.int16).max

        # Use a faster resampler for longer audio
        if len(float_audio) > 100000:
            resampler = self.fast_resampler
        else:
            resampler = self.best_resampler

        # Resample audio to the Whisper model's sample rate
        pcm_data = resampler.process(
            np.array(float_audio),
            self.whisper_sample_rate / input_sample_rate)

        return pcm_data

    def transcribe(self, pcm_data_array, channels, sample_rate):
        pcm_data = self.normalize_audio(pcm_data_array, channels, sample_rate)
        if pcm_data is None:
            logger.info("remove_silence returned no data")
            return None

        result = self.pipe(pcm_data)

        return result["text"]

    def run(self):
        while True:
            command, *args = self.command_queue.get()
            if command == 'stop':
                # Exit!
                break
            elif command == 'transcribe':
                result = self.transcribe(*args)
                self.response_queue.put(result)

# This is run from a background process
def run_loop(command_queue: Queue, response_queue: Queue):
    service = ASRService(command_queue, response_queue)
    service.run()

# Runner for the service
# Also provides an API wrapper around the queues
# There should just be one instance of this class in the main process

class ASRServiceRunner:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.command_queue = Queue()
        self.response_queue = Queue()
        self.service_process = Process(
            target=run_loop,
            args=(self.command_queue, self.response_queue))
        self.service_process.start()

    def close(self):
        self.command_queue.put(('stop',))
        self.service_process.join()
        self.command_queue.close()
        self.response_queue.close()

    async def Transcribe(self, pcm_data_array, channels, sample_rate):
        async with self.lock:
            self.command_queue.put(('transcribe', pcm_data_array, channels, sample_rate))
            return await asyncio.get_running_loop().run_in_executor(None, self.response_queue.get)
