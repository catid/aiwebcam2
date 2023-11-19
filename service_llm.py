# This runs LLM processing in a background thread to avoid blocking the main Python thread

# OpenAI
import api_key
import openai
client = openai.OpenAI(api_key=api_key.api_key)

from multiprocessing import Process, Queue
import asyncio

# Logging
from utils import logger

class LLMService:
    def __init__(self, command_queue: Queue, response_queue: Queue):
        self.command_queue = command_queue
        self.response_queue = response_queue

    def completion(self, prompt_messages):
        try:

            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=prompt_messages,
                temperature=0.5,
                max_tokens=4096,
                n=1,
                stream=True
            )

            fulltext = ""

            for completion in response:
                text = completion.choices[0].delta.content
                if text:
                    fulltext += text

                    self.response_queue.put(fulltext)

            self.response_queue.put(fulltext)

        except Exception as e:
            self.response_queue.put(f"Unexpected exception {e}")

        self.response_queue.put(None)

    def run(self):
        while True:
            command, *args = self.command_queue.get()
            if command == 'stop':
                # Exit!
                break
            elif command == 'completion':
                self.completion(*args)

# This is run from a background process
def run_loop(command_queue: Queue, response_queue: Queue):
    service = LLMService(command_queue, response_queue)
    service.run()

# Runner for the service
# Also provides an API wrapper around the queues
# There should just be one instance of this class in the main process

class LLMServiceRunner:
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

    async def CompletionBegin(self, prompt_messages):
        async with self.lock:
            self.command_queue.put(('completion', prompt_messages))

    # Returns None on final one
    async def CompletionPoll(self):
        return await asyncio.get_running_loop().run_in_executor(None, self.response_queue.get)
