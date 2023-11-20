# This runs TTS processing in a background thread to avoid blocking the main Python thread

import multiprocessing, asyncio

# Logging
from utils import logger

# OGG/Opus Parser

import struct

class OggPageHeader:
    def __init__(self, buffer, offset = 0):
        self.buffer = buffer
        self.offset = offset
        self.is_id_page = False
        self.is_comment_page = False
        self.is_audio_page = False
        self.page_segments = struct.unpack_from('B', buffer, offset + 26)[0]
        self.granule_position = struct.unpack_from('<Q', buffer, offset + 6)[0]
        self.header_size = 27 + self.page_segments

        # Calculate total page size
        self.page_sizes = struct.unpack_from('B' * self.page_segments, buffer, offset + 27)
        self.page_size = sum(self.page_sizes)

        self.version = struct.unpack_from('B', buffer, offset + 4)[0]
        self.type = {
            "continued_page": bool(struct.unpack_from('B', buffer, offset + 5)[0] & 1),
            "first_page":     bool(struct.unpack_from('B', buffer, offset + 5)[0] & 2),
            "last_page":      bool(struct.unpack_from('B', buffer, offset + 5)[0] & 4),
        }
        self.serial = format(struct.unpack_from('<I', buffer, offset + 14)[0], 'x')
        self.checksum = format(struct.unpack_from('<I', buffer, offset + 22)[0], 'x')

    @property
    def page_sequence(self):
        return struct.unpack_from('<I', self.buffer, self.offset + 18)[0]

    @property
    def is_first_page(self):
        return self.type['first_page']

    @property
    def is_last_page(self):
        return self.type['last_page']

def read_opus_id_page(buffer, offset):
    # Unpack the OpusHead structure
    opus_head_format = '<8sBBHIhB'  # Little-endian byte order
    opus_head_size = struct.calcsize(opus_head_format)

    (magic_signature, version, channel_count, pre_skip, input_sample_rate, output_gain, channel_mapping) = struct.unpack_from(opus_head_format, buffer, offset)

    # FIXME: Seeing 24 kHz here, but this plays back much too slow in the browser
    input_sample_rate *= 2

    # Convert bytes to a proper string for the magic signature
    magic_signature = magic_signature.decode('utf-8')

    # Additional channel mapping information if channel_mapping == 1
    if channel_mapping == 1:
        # Read additional fields
        stream_count, coupled_stream_count = struct.unpack_from('<BB', buffer, offset + opus_head_size)
        channel_mapping = struct.unpack_from(f'<{channel_count}B', buffer, offset + opus_head_size + 2)
    else:
        stream_count = coupled_stream_count = None

    return {
        'magic_signature': magic_signature,
        'version': version,
        'channel_count': channel_count,
        'pre_skip': pre_skip,
        'input_sample_rate': input_sample_rate,
        'output_gain': output_gain,
        'channel_mapping': channel_mapping,
        'stream_count': stream_count,
        'coupled_stream_count': coupled_stream_count
    }

class OpusFileSplitter:
    def __init__(self, buffer):
        self.buffer = buffer
        self.audio_page_boundaries = []
        self.header_bytes = None
        self.info = None
        self.parse_file(buffer)

    def scan_pages(self, buffer, callback):
        page_marker = struct.unpack('>I', b'OggS')[0]
        opus_id_header_marker = struct.unpack('>Q', b'OpusHead')[0]
        opus_comment_header_marker = struct.unpack('>Q', b'OpusTags')[0]

        id_page_found = False
        comment_page_found = False

        i = 0
        while i < len(buffer) - 4:
            if page_marker != struct.unpack_from('>I', buffer, i)[0]:
                i += 1
                continue

            if len(buffer) < i + 28:
                break

            page_header = OggPageHeader(buffer, i)

            # FIXME: Technically you can get multiple streams tagged by serial, but in practice we only receive one.

            if not id_page_found:
                if opus_id_header_marker == struct.unpack_from('>Q', buffer, i + page_header.header_size)[0]:
                    page_header.is_id_page = True
                    id_page_found = True
            elif not comment_page_found:
                if opus_comment_header_marker == struct.unpack_from('>Q', buffer, i + page_header.header_size)[0]:
                    page_header.is_comment_page = True
                    comment_page_found = True
            else:
                page_header.is_audio_page = True

            if page_header.page_size:
                i += page_header.page_size
            else:
                i += 1

            callback(page_header)

    def parse_file(self, buffer):
        audio_pages = []

        def on_page(page_header):
            nonlocal audio_pages
            if page_header.is_audio_page:
                audio_pages.append(page_header.offset)
            if page_header.is_id_page:
                self.info = read_opus_id_page(buffer, page_header.offset + page_header.header_size)

        try:
            self.scan_pages(buffer, on_page)
        except:
            pass # Handle any out of bounds reads by stopping the parsing early

        if not audio_pages:
            raise ValueError('Invalid Ogg Opus file. No audio pages found')

        self.header_bytes = buffer[:audio_pages[0]]
        self.audio_page_boundaries = audio_pages

    # Returns None if no more data is available
    def get_page(self, index, buffer_complete):
        boundaries = self.audio_page_boundaries
        if buffer_complete:
            if index >= len(boundaries):
                return None
        else:
            # Still streaming the buffer in so we need to avoid touching the last one in the file for now
            if index + 1 >= len(boundaries):
                return None

        bytes_start = boundaries[index]
        end = index + 1
        if end >= len(boundaries):
            bytes_end = len(self.buffer)
        else:
            bytes_end = boundaries[end]

        page_buffer = self.buffer[bytes_start:bytes_end]

        page_header = OggPageHeader(page_buffer)

        chunks = []

        offset = page_header.header_size
        for size in page_header.page_sizes:
            end = offset + size

            if end > len(page_buffer):
                break

            chunk = page_buffer[offset:end]
            offset += size

            chunks.append(chunk)

        if len(chunks) <= 0:
            return None

        return chunks

# TTS

import api_key
import requests

# FIXME: Other values sound bad =(  I don't think it's a bug in my code?
tts_speed = 1.0

def streamed_audio(input_text, callback, voice='alloy', model='tts-1', speed=tts_speed):
    #t0 = time.time()

    # OpenAI API endpoint and parameters
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key.api_key}",
    }

    data = {
        "model": model,
        "input": input_text,
        "speed": speed,
        "voice": voice,
        "response_format": "opus",
    }

    with requests.post(url, headers=headers, json=data, stream=True) as response:
        if response.status_code != 200:
            return False

        buffer = b''
        next_page_index = 0

        def queue_results(callback, page_index, parsed, buffer_complete):
            if not parsed.info:
                return False

            chunks = parsed.get_page(page_index, buffer_complete)

            if not chunks:
                return False

            channel_count = parsed.info["channel_count"]
            sample_rate = parsed.info["input_sample_rate"]

            for chunk in chunks:
                callback(channel_count, sample_rate, chunk)

            return True

        for chunk in response.iter_content(chunk_size=16384):
            buffer += chunk

            parsed = OpusFileSplitter(buffer)
            while queue_results(callback, next_page_index, parsed, buffer_complete=False):
                next_page_index += 1

        parsed = OpusFileSplitter(buffer)
        while queue_results(callback, next_page_index, parsed, buffer_complete=True):
            next_page_index += 1

    return True

# Runs TTS on a background thread

import av, fractions

# RTP timebase needs to be 48kHz: https://datatracker.ietf.org/doc/rfc7587/
time_base = 48000
time_base_fraction = fractions.Fraction(1, time_base)

class TTSService:
    def __init__(self, command_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue):
        self.command_queue = command_queue
        self.response_queue = response_queue

        self.codec = None
        self.channels = 0
        self.sample_rate = 0

    def init_codec(self, channels, sample_rate):
        self.codec = av.codec.CodecContext.create('opus', 'r')

        self.codec.sample_rate = sample_rate
        self.codec.channels = channels

        self.sample_rate = sample_rate
        self.channels = channels

    def speak(self, text):
        def on_chunk(channels, sample_rate, chunk):
            if self.sample_rate != sample_rate or self.channels != channels:
                self.init_codec(channels, sample_rate)

            packet = av.packet.Packet(chunk)
            packet.pts = 0
            packet.time_base = time_base_fraction

            sample_count = 0
            for frame in self.codec.decode(av.packet.Packet(chunk)):
                sample_count += frame.samples

            duration = sample_count / self.sample_rate

            pts_count = round(duration * time_base)

            self.response_queue.put((duration, pts_count, chunk))

        streamed_audio(text, on_chunk)

    def run(self):
        while True:
            command, *args = self.command_queue.get()
            if command == 'stop':
                # Exit!
                break
            elif command == 'speak':
                self.speak(*args)

# This is run from a background process
def run_loop(command_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue):
    service = TTSService(command_queue, response_queue)
    service.run()
    logger.info("TTS background run loop process exiting")

# Runner for the service
# Also provides an API wrapper around the queues
# There should just be one instance of this class in the main process

class TTSServiceRunner:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.command_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()
        self.service_process = multiprocessing.Process(
            target=run_loop,
            args=(self.command_queue, self.response_queue))
        self.service_process.start()

        self.silence_duration = 0.02

        self.next_pts = 0

    def close(self):
        logger.info("Stopping background TTS worker...")
        self.command_queue.put(('stop',))
        self.service_process.join()
        logger.info("Closing command_queue...")
        self.command_queue.close()
        logger.info("Closing response_queue...")
        self.response_queue.close()
        logger.info("Stopped background TTS worker.")

    def generate_silence_packet(self, duration_seconds):
        chunk = bytes.fromhex('f8 ff fe')

        packet = av.packet.Packet(chunk)
        packet.pts = self.next_pts
        packet.dts = self.next_pts
        packet.time_base = time_base_fraction

        pts_count = round(duration_seconds * time_base)
        self.next_pts += pts_count

        #logger.info(f"silence pts_count = {pts_count}")

        return packet

    # Grab either the next TTS Opus packet to play back,
    # or a silence packet if no data is available.
    def poll_packet(self):
        try:
            duration, pts_count, chunk = self.response_queue.get_nowait()

            packet = av.packet.Packet(chunk)
            packet.pts = self.next_pts
            packet.dts = self.next_pts
            packet.time_base = time_base_fraction

            self.next_pts += pts_count

            return packet, duration

        except:
            pass # Ignore Empty exception

        return self.generate_silence_packet(self.silence_duration), self.silence_duration


    async def Speak(self, text):
        async with self.lock:
            self.command_queue.put(('speak', text))
