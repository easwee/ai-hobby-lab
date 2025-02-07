import asyncio
from dotenv import load_dotenv
from workflows.youtube_audio_data_extractor import YoutubeAudioDataExtractor

load_dotenv()

def main():
    print("Starting...")
    workflow = YoutubeAudioDataExtractor()
    workflow.run(urls=[
        "https://www.youtube.com/watch?v=vlEGY8IPD-Q"
    ])


if __name__ == "__main__":
    main()
