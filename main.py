import argparse
import json
from dotenv import load_dotenv
from workflows.image_background_remover import ImageBackgroundRemover
from workflows.youtube_audio_data_extractor import YoutubeAudioDataExtractor

# if you have created the .env file on your local machine you can use dotenv
# package to make variables defined in .env  available in our scripts
# we will read them using os.getenv("SOME_VARIABLE") later when needed
load_dotenv()

# list of workflow classes available to run - extend it with new workflows
AVAILABLE_WORKFLOWS = {
    "youtube_audio_data_extractor": YoutubeAudioDataExtractor,
    "image_background_remover": ImageBackgroundRemover
}

def main(workflow: str, input: any):
    print("Starting...")

    # check if input workflow actually exists on our list
    workflow_class = AVAILABLE_WORKFLOWS.get(workflow)
    if not workflow_class:
        raise ValueError(f"Workflow '{workflow}' is not recognized.")

    # if it exists let's create an instance of it
    workflow_instance = workflow_class()

    # each workflow will have a run method that will accept our input
    # let's call it and pass the input
    workflow_instance.run(input)


if __name__ == "__main__":
    # Initialize argument parser so we can pass arguments when running:
    # python main.py -w workflow_name -i ["one", "two"]
    parser = argparse.ArgumentParser(description='AI Hobby Lab')

    # Define which arguments can be used with our main runner
    parser.add_argument("-w", "--workflow", help="The workflow to execute.")
    parser.add_argument("-i", "--input", help="Workflow input data - can be any type, but has to be parsable by called workflow.")

    # Parse arguments
    args = parser.parse_args()
    input = json.loads(args.input)

    # pass them to main workflow handler class
    main(args.workflow, input)
