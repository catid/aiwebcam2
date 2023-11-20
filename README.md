# AI Webcam2

Fully-interactive AI assistant you can run at home!

This is my second attempt at implementing this.  It was too much work for a Python side project last time, but now that OpenAI has provided implementations for vision and TTS it's very easy to get running now.

Implemented with latest Whisper3 + GPT-4-Vision + OpenAI TTS and a WebRTC browser front-end for speed.

![Logo](static/logo256.png)

Demo here: https://www.youtube.com/watch?v=G_L8t3EQMcs

## Prerequisites

Designed for Ubuntu server with an Nvidia GPU.  It might work with other setups, but I have only tested this one.

You'll want to set up Conda first: https://docs.conda.io/en/latest/miniconda.html

## Setup

Create a Conda environment, clone the repo, and install the requirements:

```bash
conda create -n aiwebcam python=3.10
conda activate aiwebcam

git clone https://github.com/catid/aiwebcam2
cd aiwebcam

#sudo apt install libopenblas-dev
#sudo apt install libopus-dev libopusfile0

# Follow instructions from https://pytorch.org/get-started/locally/
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -U -r requirements.txt

openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 3650
# Press enter to accept defaults for all questions
```

## Run

```bash
python app.py
```

Open a Chrome browser to https://localhost:8443
On my network I host it on a server at https://gpu3.lan:8443

When you get the `Your connection is not private` screen, click `Advanced` and then `Proceed to localhost (unsafe)`.

When you see the "localhost:8443 wants to: Use your camera" permission popup, select `[Allow]`.

You should see the webcam feed in the browser window.  Click or press and hold the `[space bar]` on the keyboard to speak.  The AI will respond to what you say, and it will be provided a picture from the webcam stream so that it can see you for context.

Include "look" in your query to use more tokens to improve its eyesight.  Include "remember" to keep a high resolution image for the remainder of the session.
