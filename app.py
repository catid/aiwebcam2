# Logging

from utils import logger

import socketio

sio = socketio.AsyncServer(cors_allowed_origins='*')

# Background services

from service_asr import ASRServiceRunner
asr_runner = ASRServiceRunner()

from service_llm import LLMServiceRunner
llm_runner = LLMServiceRunner()

from service_tts import TTSServiceRunner

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

        self.tts = TTSServiceRunner()

        self.stream_time = None

    async def close(self):
        super().stop()
        self.tts.close()

    async def recv(self):
        packet, duration = self.tts.poll_packet()

        #logger.info(f"opus duration={duration} pts={packet.pts}")

        if self.stream_time is None:
            self.stream_time = time.time()

        wait = self.stream_time - time.time()
        if wait > 0.001:
            await asyncio.sleep(wait)

        self.stream_time += duration
        return packet

class WebRTCConnection:
    def __init__(self, sid):
        self.sid = sid

        self.pc = RTCPeerConnection()
        self.video_track = None

        self.processing_audio = False
        self.recording = False

        self.opus_track = CustomAudioStream()

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"self.pc.connectionState = {self.pc.connectionState}")
            if self.pc.connectionState == 'closed':
                await self.close()

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

    async def close(self):
        await self.pc.close()
        if self.opus_track:
            await self.opus_track.close()
            self.opus_track = None

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
            await self.rtc_peer_connection.close()
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

        await self.rtc_peer_connection.opus_track.tts.Speak(message)

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
            await session.cleanup()
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

import asyncio

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

        logger.info("Starting HTTP server...")

        aiohttp.web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)

        logger.info("HTTP server stopped")

    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Terminating asyncio event loop...")

        async def clear_all_sessions():
            await sessions.clear_sessions()

        loop = asyncio.get_event_loop()
        loop.run_until_complete(clear_all_sessions())
        loop.close()

        logger.info("Terminating background services...")

        asr_runner.close()
        llm_runner.close()

        logger.info("Background services terminated gracefully")

if __name__ == "__main__":
    main()
