# Logging

from utils import logger

import socketio

sio = socketio.AsyncServer(cors_allowed_origins='*')

# Background services

from service_asr import ASRServiceRunner
asr_runner = ASRServiceRunner()

from service_llm import LLMServiceRunner
llm_runner = LLMServiceRunner()

# WebRTC peer listening for a single browser to connect
# We run each WebRTC peer in a separate process to avoid stalls in playback

from aiortc import RTCIceCandidate, RTCSessionDescription, RTCPeerConnection
from aiortc.mediastreams import AudioStreamTrack, VideoStreamTrack, MediaStreamError, MediaStreamTrack
from queue import Queue
from fractions import Fraction
import asyncio

import re

import io
from PIL import Image
import base64

import numpy as np
import time

import av, fractions
import opuslib, samplerate

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
    def granule_position(self):
        return struct.unpack_from('<Q', self.buffer, self.offset + 6)[0]

    @property
    def page_sequence(self):
        return struct.unpack_from('<I', self.buffer, self.offset + 18)[0]

    @property
    def is_first_page(self):
        return self.type['first_page']

    @property
    def is_last_page(self):
        return self.type['last_page']

def read_opus_head(buffer, offset):
    # Unpack the OpusHead structure
    opus_head_format = '<8sBBHIhB'  # Little-endian byte order
    opus_head_size = struct.calcsize(opus_head_format)

    (magic_signature, version, channel_count, pre_skip, input_sample_rate, output_gain, channel_mapping) = struct.unpack_from(opus_head_format, buffer, offset)

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
                self.info = read_opus_head(buffer, page_header.offset + page_header.header_size)

        try:
            self.scan_pages(buffer, on_page)
        except:
            pass # Handle any out of bounds reads by stopping the parsing early

        if not audio_pages:
            raise ValueError('Invalid Ogg Opus file. No audio pages found')

        self.header_bytes = buffer[:audio_pages[0]]
        self.audio_page_boundaries = audio_pages

    # Returns None if no more data is available
    def get_page(self, index):
        boundaries = self.audio_page_boundaries
        if index >= len(boundaries):
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

# OpenAI
import api_key
import openai, requests
client = openai.OpenAI(api_key=api_key.api_key)

def streamed_audio(input_text, voice='alloy', model='tts-1', speed=1.0):
    t0 = time.time()

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

        for chunk in response.iter_content(chunk_size=16384):
            buffer += chunk

            logger.info("*** Got chunk")

            parsed = OpusFileSplitter(buffer)

            while True:
                chunks= parsed.get_page(next_page_index)
                if not chunks:
                    break

                t1 = time.time()
                logger.info(f"Page {next_page_index} demuxed at t={(t1 - t0) * 1000.0} msec. count={len(chunks)}")

                if parsed.info:
                    channel_count = parsed.info["channel_count"]
                    input_sample_rate = parsed.info["input_sample_rate"]

                    #logger.info(f"Audio format: channel_count={channel_count} input_sample_rate={input_sample_rate}")

                    for i, chunk in enumerate(chunks):
                        with open(f"chunk{next_page_index}_{i}.opus", 'wb') as file:
                            file.write(chunk)

                    next_page_index += 1

        parsed = OpusFileSplitter(buffer)

        while True:
            chunks= parsed.get_page(next_page_index)
            if not chunks:
                break

            t1 = time.time()
            logger.info(f"Page {next_page_index} demuxed at t={(t1 - t0) * 1000.0} msec. count={len(chunks)}")

            if parsed.info:
                channel_count = parsed.info["channel_count"]
                input_sample_rate = parsed.info["input_sample_rate"]

                #logger.info(f"Audio format: channel_count={channel_count} input_sample_rate={input_sample_rate}")

                for i, chunk in enumerate(chunks):
                    with open(f"chunk{next_page_index}_{i}.opus", 'wb') as file:
                        file.write(chunk)

                next_page_index += 1

        with open(f"all_chunks.opus", 'wb') as file:
            file.write(buffer)

    return True

streamed_audio("The BetterTransformer implementation does not support padding during training, as the fused kernels do not support attention masks. Beware that passing padded batched data during training may result in unexpected outputs. Please refer to https://huggingface.co/docs/optimum/bettertransformer/overview for more details.")

logger.info("Done")

exit()

sample_rate = 48000

# 20 milliseconds is a good trade off between latency and quality/size
frame_size = int(0.020 * sample_rate)  # 20 ms * 48000 samples/s = 960 samples

opus_encoder = opuslib.api.encoder.create_state(sample_rate, 1, opuslib.APPLICATION_AUDIO)

output_file = io.BytesIO()
output_container = av.open(output_file, format="opus", mode='w')
output_audio_stream = output_container.add_stream('opus', rate=sample_rate)
output_audio_stream.channels = 1
output_audio_stream.rate = sample_rate

def generate_pcm_floats_av_packets(pcm_floats, input_sample_rate):
    packets = []

    # Resample data from 22050 Hz to 48000 Hz
    resampler = samplerate.Resampler('sinc_best', channels=1)
    pcm_floats = resampler.process(np.array(pcm_floats), 48000 / input_sample_rate)

    next_chunk_index = 0

    # iterate over the data in chunks of frame_size
    for i in range(0, len(pcm_floats), frame_size):
        frame_chunk = pcm_floats[i:i + frame_size]

        # if the last frame is shorter than frame_size, pad it with zeros
        if len(frame_chunk) < frame_size:
            frame_chunk = np.pad(frame_chunk, ((frame_size - len(frame_chunk)), (0)))

        encoded_data = opuslib.api.encoder.encode_float(
            opus_encoder,
            frame_chunk.tobytes(),
            frame_size,
            frame_size * 4)

        # Open the output file in binary write mode
        with open("silence.opus", 'wb') as output_file:
            logger.info(f"buffer len = {len(encoded_data)}")
            output_file.write(encoded_data)

        if encoded_data:
            packet = av.packet.Packet(encoded_data)
            packet.pts = next_chunk_index
            packet.stream = output_audio_stream
            packet.time_base = Fraction(1, sample_rate)

            next_chunk_index += frame_size

            packets.append(packet)

    return packets

silence_packets = generate_pcm_floats_av_packets(np.zeros(frame_size), 48000)

exit()


# WebRTC Connection

class VideoReceiver(VideoStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()  # Initialize the MediaStreamTrack
        self.track = track
        self.recording = False
        self.recorded_frame = None

    def startRecording(self):
        self.recording = True
        self.recorded_frame = None

    def endRecording(self):
        self.recording = False
        image = self.recorded_frame
        self.recorded_frame = None
        return image

    async def recv(self):
        frame = await self.track.recv()

        # Process the frame (e.g., save to a file, play audio, etc.)
        if self.recording:
            if not self.recorded_frame:
                self.recorded_frame = frame

        return frame

class CustomAudioStream(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self):
        super().__init__()  # don't forget this!

        self.q = Queue()
        self._start = None

        self.next_chunk_index = 0

        self.sample_rate = 48000

        # 20 milliseconds is a good trade off between latency and quality/size
        self.frame_size = int(0.020 * self.sample_rate)  # 20 ms * 48000 samples/s = 960 samples

        self.opus_encoder = opuslib.api.encoder.create_state(self.sample_rate, 1, opuslib.APPLICATION_AUDIO)

        self.output_file = io.BytesIO()
        self.output_container = av.open(self.output_file, format="opus", mode='w')
        self.output_audio_stream = self.output_container.add_stream('opus', rate=self.sample_rate)
        self.output_audio_stream.channels = 1
        self.output_audio_stream.rate = self.sample_rate

        self.silence_packets = self.generate_pcm_floats_av_packets(np.zeros(self.frame_size), 48000)

        self.speak("Please put your text in my box and chat with me!")

    async def close(self):
        super().stop()
        self.silence_packets = None
        if self.opus_encoder:
            opuslib.api.encoder.destroy(self.opus_encoder)
            self.opus_encoder = None
        if self.output_container:
            self.output_container.close()
            self.output_container = None
        self.output_file = None
        self.output_audio_stream = None

    def next_pts(self):
        self.next_chunk_index += self.frame_size
        return self.next_chunk_index

    def get_silence_packet(self):
        packet = self.silence_packets[0]

        packet.pts = self.next_pts()

        return packet

    def generate_pcm_floats_av_packets(self, pcm_floats, input_sample_rate):
        packets = []

        # Resample data from 22050 Hz to 48000 Hz
        resampler = samplerate.Resampler('sinc_best', channels=1)
        pcm_floats = resampler.process(np.array(pcm_floats), 48000 / input_sample_rate)

        # iterate over the data in chunks of frame_size
        for i in range(0, len(pcm_floats), self.frame_size):
            frame_chunk = pcm_floats[i:i + self.frame_size]

            # if the last frame is shorter than frame_size, pad it with zeros
            if len(frame_chunk) < self.frame_size:
                frame_chunk = np.pad(frame_chunk, ((self.frame_size - len(frame_chunk)), (0)))

            encoded_data = opuslib.api.encoder.encode_float(
                self.opus_encoder,
                frame_chunk.tobytes(),
                self.frame_size,
                self.frame_size * 4)

            if encoded_data:
                packet = av.packet.Packet(encoded_data)
                packet.pts = self.next_pts()
                packet.stream = self.output_audio_stream
                packet.time_base = Fraction(1, self.sample_rate)

                packets.append(packet)

        # Open the output file in binary write mode
        with open("silence.opus", 'wb') as output_file:
            for packet in packets:
                # Write the raw packet data to the file
                output_file.write(packet.to_bytes())

        return packets

    def add_pcm_floats(self, pcm_floats, input_sample_rate):
        packets = self.generate_pcm_floats_av_packets(pcm_floats, input_sample_rate)

        for packet in packets:
            self.add_av_packet(packet)

    def speak(self, message):
        t0 = time.time()
        pcm = tts.tts(message, speaker=tts_speaker)
        t1 = time.time()
        self.add_pcm_floats(pcm, tts_sample_rate)
        t2 = time.time()

        logger.info(f"TTS generated audio in {(t1-t0)/1000.0} compressed in {(t2-t1)/1000.0} milliseconds: message='{message}'")

    def add_av_packet(self, packet):
        self.q.put(packet)

    async def recv(self):
        try:
            packet = self.q.get_nowait()
        except:
            packet = None # Ignore Empty exception

        if not packet:
            packet = self.get_silence_packet()

        frame_time = packet.pts / self.sample_rate
        if self._start is None:
            self._start = time.time() - frame_time
        else:
            wait = self._start + frame_time - time.time()
            await asyncio.sleep(wait)

        return packet

class WebRTCConnection:
    def __init__(self, sid):
        self.sid = sid

        self.pc = RTCPeerConnection()
        self.video_track = None

        self.processing_audio = False
        self.recording = False

        self.opus_track = OpusTrack()

        @self.pc.on("icecandidate")
        def on_icecandidate(candidate):
            sio.emit("ice_candidate", {"candidate": candidate}, room=sid)

        @self.pc.on("track")
        def on_track(track):
            logger.info(f"Track received: {track.kind}")

            if track.kind == "audio":
                if not self.processing_audio:
                    self.processing_audio = True
                    asyncio.create_task(self.process_audio_track(track))
                pass
            elif track.kind == "video":
                # This will mirror video frames back to user since we add an outgoing track based on input
                self.video_track = VideoReceiver(track)
                self.pc.addTrack(self.video_track)
                pass

            @track.on("ended")
            async def on_ended():
                logger.info(f"Track ended: {track.kind}")

        logger.info(f"Created WebRTC peer connection")

    def recordStart(self):
        self.recording = True
        self.video_track.startRecording()
        self.audio_frames = []

    def recordEnd(self):
        if not self.recording:
            return None
        self.recording = False
        return self.video_track.endRecording()

    async def create_session_answer(self, browser_sdp):
        logger.info(f"[{self.sid}] Creating session answer")

        description = RTCSessionDescription(sdp=browser_sdp["sdp"], type=browser_sdp["type"])

        await self.pc.setRemoteDescription(description)

        self.pc.addTrack(self.opus_track)

        await self.pc.setLocalDescription(await self.pc.createAnswer())

        return self.pc.localDescription

    async def process_audio_track(self, track):
        while True:
            try:
                frame = await track.recv()

                if self.recording:
                    self.audio_sample_rate = frame.sample_rate
                    self.audio_channels = len(frame.layout.channels)
                    self.audio_frames.append(frame.to_ndarray())
            except MediaStreamError:
                # This exception is raised when the track ends
                break
        logger.info(f"Exited audio processing loop")

# Chat Messages

class ChatLine:
    def __init__(self, id, sender, message, image):
        self.id = id
        self.sender = sender
        self.message = message
        self.image = image
        self.base64_image = None

def convert_yuv420_to_ycbcr(data, w, h):
    # Extract Y, U, and V channels
    Y = data[:h]
    U = data[h:(h + h//4)].reshape((h // 2, w // 2))
    V = data[(h + h//4):].reshape((h // 2, w // 2))

    # Upsample U and V channels
    U_upsampled = U.repeat(2, axis=0).repeat(2, axis=1)
    V_upsampled = V.repeat(2, axis=0).repeat(2, axis=1)

    # Stack the channels
    ycbcr_image = np.stack([Y, U_upsampled, V_upsampled], axis=-1)

    return ycbcr_image

class ChatManager:
    def __init__(self):
        self.chat_lines = []
        self.current_id = 1

    def clear(self):
        self.chat_lines = []
        self.current_id = 1

    def append_line(self, sender, message, image=None):
        new_line = ChatLine(self.current_id, sender, message, image)
        self.chat_lines.append(new_line)
        self.current_id += 1
        return new_line

    def remove_line_by_id(self, id):
        for i, chat_line in enumerate(self.chat_lines):
            if chat_line.id == id:
                del self.chat_lines[i]
                return True
        return False

    def to_prompt(self):
        prompt_messages = []

        prompt_messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful AI assistant that can see the user."
                },
            ],
        })

        for i, line in enumerate(self.chat_lines):
            image = None

            detail = "auto"

            if i + 1 == len(self.chat_lines):
                image = line.image
            else:
                # If the line contains the word "remember":
                if re.search(r"\bremember\b", line.message, re.IGNORECASE):
                    image = line.image
                    detail = "high"

            if image is not None:
                # If the line contains the word "look":
                if re.search(r"\blook\b", line.message, re.IGNORECASE):
                    detail = "high"
                elif re.search(r"\bremember\b", line.message, re.IGNORECASE):
                    detail = "high"

                # TBD: We may want to move this into a background thread if it causes audio hitches
                if not line.base64_image:
                    # The input is a 2D image with concantenated planes.
                    # The first wxh bytes are the Y channel, followed by (w/2 x h/2) U then V channels.

                    t0 = time.time()

                    width = image.shape[1]
                    yuv_h = image.shape[0]
                    height = round(yuv_h * 2 / 3)

                    image = convert_yuv420_to_ycbcr(image, width, height)

                    img = Image.fromarray(image, 'YCbCr')

                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=95)

                    buffer.seek(0)
                    line.base64_image = base64.b64encode(buffer.read()).decode('utf-8')

                    # Seems to take about 1 millisecond.  Maybe fine to leave on main thread?
                    t1 = time.time()
                    logger.info(f"Cached base64 webcam image in {(t1 - t0) * 1000.0} msec")

                prompt_messages.append({
                    "role": line.sender,
                    "content": [
                        {
                            "type": "text",
                            "text": line.message
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{line.base64_image}",
                                "detail": detail
                            }
                        }
                    ],
                })
            else:
                prompt_messages.append({
                    "role": line.sender,
                    "content": [
                        {
                            "type": "text",
                            "text": line.message
                        }
                    ],
                })

        return prompt_messages

# Sessions

class Session:
    def __init__(self, sid):
        self.sid = sid
        self.browser_id = None
        self.user_name = "user"
        self.ai_name = "assistant"
        self.rtc_peer_connection = None
        self.recording = False
        self.chat = ChatManager()
        self.line_in_progress = None
        self.last_push_time = time.time()
        self.use_vision = True

    def on_vision_toggle(self, toggle):
        self.use_vision = toggle
        logger.info(f"Use Vision toggle = {self.use_vision}")

    async def cleanup(self):
        if self.rtc_peer_connection:
            await self.rtc_peer_connection.pc.close()
            self.rtc_peer_connection = None

    async def on_response_part(self, message):
        if not self.line_in_progress:
            self.line_in_progress = self.chat.append_line(self.ai_name, message)
        line = self.line_in_progress
        line.message = message

        t = time.time()
        if t - self.last_push_time > 0.2:
            self.last_push_time = t

            await sio.emit("add_chat_message", {"id": line.id, "sender": line.sender, "message": line.message}, room=self.sid)

    async def on_response(self, message):
        logger.info(f"result = {message}")

        if not self.line_in_progress:
            self.line_in_progress = self.chat.append_line(self.ai_name, message)
        line = self.line_in_progress
        line.message = message

        self.last_push_time = time.time()

        await sio.emit("add_chat_message", {"id": line.id, "sender": line.sender, "message": line.message}, room=self.sid)

        self.line_in_progress = None

        streamed_audio(message, self.rtc_peer_connection.opus_track.opus_queue)

    async def get_response(self):
        prompt_messages = self.chat.to_prompt()

        #logger.info(f"prompt_messages = {prompt_messages}")
        if self.use_vision:
            await llm_runner.VisionCompletionBegin(prompt_messages)
        else:
            await llm_runner.TextCompletionBegin(prompt_messages)

        text = ""

        while True:
            r = await llm_runner.CompletionPoll()
            if r is None:
                break
            text = r
            await self.on_response_part(text)
        await self.on_response(text)


class SessionManager:
    def __init__(self):
        self.sessions = []

    def create_session(self, sid):
        session = Session(sid)
        self.sessions.append(session)
        logger.info(f"Created new session: {sid}")
        return session

    async def upgrade_session_on_browser_id(self, sid, browser_id):
        new_session = None
        old_session = None

        for session in self.sessions:
            if session.sid == sid:
                new_session = session
                break
        for session in self.sessions:
            logger.info(f"OLD: {session.sid} {session.browser_id} {session.chat} NEW: {sid} {browser_id}")
            if session.browser_id == browser_id:
                old_session = session
                break

        new_session.browser_id = browser_id

        if old_session:
            if old_session.chat:
                new_session.chat = old_session.chat

            # Clean up the old session
            await old_session.cleanup()
            self.sessions.remove(old_session)

            logger.info(f"Removed old session with browser ID {browser_id} on upgrading new session {sid}")
        else:
            logger.info(f"Upgraded session {sid} to browser ID {browser_id} - No old session found")

        return new_session

    async def clear_sessions(self):
        for session in self.sessions:
            session.cleanup()
        self.sessions.clear()

    def find_session_by_sid(self, sid):
        for session in self.sessions:
            if session.sid == sid:
                return session
        return None

    def find_session_by_browser_id(self, browser_id):
        for session in self.sessions:
            if session.browser_id == browser_id:
                return session
        return None

    async def on_socket_io_disconnect(self, sid):
        session = self.find_session_by_sid(sid)
        if session:
            await session.cleanup()
            # We do not remove old sessions in case the user reconnects
            #self.sessions.remove(session)

sessions = SessionManager()

# Socket.io server

@sio.event
async def connect(sid, environ):
    session = sessions.find_session_by_sid(sid)
    if session:
        logger.warn("Got a second connect event for the same sid")
        return

    session = sessions.create_session(sid)

    logger.info(f"Client connected: {sid}")

    session.rtc_peer_connection = WebRTCConnection(sid)

    logger.info(f"Added WebRTC peer connection for {sid}")

@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")

    await sessions.on_socket_io_disconnect(sid)

@sio.event
async def session_message(sid, data):
    session = sessions.find_session_by_sid(sid)
    if session:
        logger.info(f"Received Session message from {sid}: {data}")

        browser_sdp = data["sdp"]

        local_description = await session.rtc_peer_connection.create_session_answer(browser_sdp)

        if local_description:
            await sio.emit("session_response", {"sdp": local_description.sdp, "type": local_description.type}, room=sid)

@sio.event
async def candidate_message(sid, data):
    session = sessions.find_session_by_sid(sid)
    if session:
        logger.info(f"Received ICE candidate message from {sid}: {data}")

@sio.event
async def id_message(sid, data):
    session = await sessions.upgrade_session_on_browser_id(sid, data["id"])
    if session:
        await sio.emit("clear_chat_message", {}, room=sid)
        for line in session.chat.chat_lines:
            await sio.emit("add_chat_message", {"id": line.id, "sender": line.sender, "message": line.message}, room=sid)
        await sio.emit("allow_message", {}, room=sid)

@sio.event
async def user_name_message(sid, data):
    session = sessions.find_session_by_sid(sid)
    if session:
        logger.info(f"Received User Name message from {sid}: {data}")
        # Vision model does not support this yet
        #session.user_name = data.get(data["name"], session.user_name)

@sio.event
async def ai_name_message(sid, data):
    session = sessions.find_session_by_sid(sid)
    if session:
        logger.info(f"Received AI Name message from {sid}: {data}")
        # Vision model does not support this yet
        #session.ai_name = data.get(data["name"], session.ai_name)

@sio.event
async def record_message(sid, data):
    session = sessions.find_session_by_sid(sid)
    if session:
        logger.info(f"Received Record message from {sid}: {data}")
        session.recording = data.get("recording", False)
        if session.recording:
            session.rtc_peer_connection.recordStart()
        else:
            recorded_frame = session.rtc_peer_connection.recordEnd()
            if recorded_frame:
                audio_frames = session.rtc_peer_connection.audio_frames
                if len(audio_frames) > 0:
                    audio_sample_rate = session.rtc_peer_connection.audio_sample_rate
                    audio_channels = session.rtc_peer_connection.audio_channels

                    message = await asr_runner.Transcribe(audio_frames, audio_channels, audio_sample_rate)
                    if message and len(message) > 0:
                        image = recorded_frame.to_ndarray(format='yuv420p')

                        line = session.chat.append_line(session.user_name, message, image)

                        await sio.emit("add_chat_message", {"id": line.id, "sender": line.sender, "message": line.message}, room=sid)

                        await session.get_response()

@sio.event
async def chat_message(sid, data):
    session = sessions.find_session_by_sid(sid)
    if session:
        logger.info(f"Received Chat message from {sid}: {data}")
        message = data.get("message", "")
        if len(message) > 0:
            line = session.chat.append_line(session.user_name, message)
            await sio.emit("add_chat_message", {"id": line.id, "sender": line.sender, "message": line.message}, room=sid)

            await session.get_response()

        await sio.emit("allow_message", {}, room=sid)

@sio.event
async def linex_message(sid, data):
    session = sessions.find_session_by_sid(sid)
    if session:
        logger.info(f"Received LineX message from {sid}: {data}")
        id = data.get("id", 0)
        session.chat.remove_line_by_id(id)
        await sio.emit("remove_chat_message", {"id": id}, room=sid)

@sio.event
async def cancel_message(sid, data):
    session = sessions.find_session_by_sid(sid)
    if session:
        # FIXME
        await sio.emit("allow_message", {}, room=sid)

@sio.event
async def use_vision(sid, data):
    session = sessions.find_session_by_sid(sid)
    if session:
        session.on_vision_toggle(data.get("value"))

# HTTP server

import aiohttp.web

app = aiohttp.web.Application()
sio.attach(app)

async def handle_index(request):
    if request.path == "/":
        with open("./static/index.html") as f:
            text = f.read()
            return aiohttp.web.Response(text=text, content_type="text/html")

app.router.add_route("GET", "/", handle_index)  # Serve index.html for the root URL path
app.router.add_route("GET", "/index.html", handle_index)  # Serve index.html for the /index.html URL path
app.router.add_static("/", "./static")  # Serve files from the ./static/ directory using the root URL path

# Shutdown hook

import asyncio, signal

def sig_handler(sig, frame):
    logger.info("Terminating on signal...")
    loop = asyncio.get_event_loop()
    async def clear_all_sessions():
        await sessions.clear_sessions()
    loop.run_until_complete(clear_all_sessions())
    sio.stop(loop)
    loop.stop()

def AddShutdownHook():
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

# Entrypoint

import ssl
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="WebRTC signaling server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the signaling server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8443,
        help="Port for the signaling server (default: 8443)",
    )
    parser.add_argument(
        "--cert-file",
        type=str,
        default="cert.pem",
        help="Path to the SSL certificate file (default: cert.pem)",
    )
    parser.add_argument(
        "--key-file",
        type=str,
        default="key.pem",
        help="Path to the SSL key file (default: key.pem)",
    )
    return parser.parse_args()

def main():
    try:
        args = get_args()

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(args.cert_file, args.key_file)

        AddShutdownHook()

        logger.info("Starting HTTP server...")

        aiohttp.web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)

        logger.info("HTTP server stopped")

    except KeyboardInterrupt:
        pass
    finally:
        loop = asyncio.get_event_loop()
        loop.close()

        logger.info("Terminating background services")

        asr_runner.close()
        llm_runner.close()

if __name__ == "__main__":
    main()
