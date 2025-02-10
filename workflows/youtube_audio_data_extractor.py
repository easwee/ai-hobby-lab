import os
import base64

from concurrent.futures import ThreadPoolExecutor
from yt_dlp import YoutubeDL
from openai import OpenAI
from markdown2 import Markdown

# Sample input:
# [
#     "https://www.youtube.com/watch?v=AyOxay5AtMM",
#     "https://www.youtube.com/watch?v=vlEGY8IPD-Q"
# ]

# USER_PROMPT_TEMPLATE defines the audio intelligence task.
# It is currently instructing Soniox Omnio multimodal LLM
# to extract formatted food recipes data from input audio
USER_PROMPT_TEMPLATE = """Input audio contains cooking recipe data. Output formatted markdown document containing:
1. Recipe title
2. Short summary paragraph of what is being cooked.
3. List of ingredients formatted as:
- <ingredient> <amount>
4. Step by step cooking instructions as explained in the audio.

Use wording as if you were conveying the recipe over radio. Make sure every listed ingredient is also used in the cooking process, otherwise do not list it.
"""


class YoutubeAudioDataExtractor:

    # We will be triggering the run method and pass in a list of youtube video urls
    # from which we want to extract data
    def run(self, input: list[str]):
        print("Workflow started.")
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = executor.map(self.process, input)
            print(list(results))

    # Our process method will be in charge of chaining together multiple steps of data extraction
    # We want to run the process in a separate thread for each url, to speed things up in case of multiple urls
    def process(self, url: str):
        print(f"Processing {url}")

        # combine all the steps of this workflow together
        # 1. download youtube video and obtain an audio file and it's name
        file_name = self.download(url)
        # 2. run Soniox Omnio multimodal llm and extract data from audio
        data = self.extract_data(file_name)
        # 3. write output to pdf
        self.create_pdf(file_name, data)

        return (file_name, "done")

    # In download step we use yt-dlp to download video/audio from youtube
    def download(self, url: str) -> str:
        print(f"[download] Start {url}.")

        # configuration options for yt-dlp:-
        # - we are interested in audio only, we like flac
        # - after the video is downloaded the yt-dlp postprocessor will convert video to .flac instantly
        options = {
            'format': 'bestaudio/best',
            'outtmpl': "./%(title)s.%(ext)s",
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'flac',
                'preferredquality': '0',
            }],
        }

        # start downloading the file using yt-dlp library, and also extract audio file meta information
        # like downloaded file name, so we can reference the file later
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            # we want to store and return the fle name because we need it later
            # Gfirst get the original filename  that yt-dlp read from web (.m4a or .webm)
            original_file_name = ydl.prepare_filename(info)

            # Manually change the extension to .flac to match file on disk
            downloaded_file_name = os.path.splitext(original_file_name)[0] + '.flac'

            print(f"[download] {downloaded_file_name} done.")
            return downloaded_file_name

    def extract_data(self, file_name: str) -> str:
        print(f"[ExtractData]: {file_name}.")

        # We need to read the downloaded audio file and convert it to base64
        # so we can pass it to Soniox Omnio API
        with open(file_name, "rb") as file:
            audio_data = file.read()
            audio_data_b64 = base64.b64encode(audio_data).decode('utf-8')

        # Soniox Omnio API is fully compatible with the now standard OpenAI SDK,
        # we just need to point the base_url to Soniox api url, instead of the default OpenAI api
        # and also we use our SONIOX_API_KEY that is set in .env file
        client = OpenAI(
            api_key=os.getenv("SONIOX_API_KEY"),
            base_url="https://api.llm.soniox.com/v1",
        )
        print("[ExtractData]: Running audio intelligence...")

        # we use completions method to create a new request
        # important part here is that audio base64 data has to be set as a partial content message
        # as "audio_data_b64" prop, so the API will grab audio properly
        completion = client.chat.completions.create(
            model="omnio-chat-audio-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"audio_data_b64": audio_data_b64},
                        {"text": USER_PROMPT_TEMPLATE},
                    ],
                }
            ],
        )

        # this is optional part, but if we don't need the downloaded file anymore
        # we can just remove it to not waste space
        if os.path.exists(file_name):
            os.remove(file_name)

        # read llm response - non-streamed all in one
        data = completion.choices[0].message.content

        print("-- [ExtractData] Done.")
        print(data)

        return data

    def create_pdf(self, file_name: str, data: str):
        markdowner = Markdown()
        output_dir = os.getenv("OUTPUT_DIR") or "./output"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        recipe_file = open(f"./output/{file_name}.html","w")
        recipe_file.write(markdowner.convert(data))
        recipe_file.close()
