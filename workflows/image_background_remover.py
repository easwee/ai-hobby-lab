import os
import requests
import math
from PIL import Image
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor

# Sample input - a list of image paths:
# [
#     "folder/image1.png",
#     "folder/another_image.png"
# ]

# define our workflow class
class ImageBackgroundRemover:

    # Maximum number of pixels allowed by StabilityAI API
    MAX_PIXELS = 4_194_304

    # We will be triggering the run method and pass in a list of youtube video urls
    # from which we want to extract data
    def run(self, input: list[str]):
        print("Workflow started.")

        # We can speed up the processing with multi-threading - increase max_workers amount if you can afford more
        with ThreadPoolExecutor(max_workers=2) as executor:
            # execute the process method on each of our input objects
            results = executor.map(self.process, input)
            # Print out informational tuple list
            print(list(results))

    # Our process method will be in charge of chaining together multiple steps of data extraction
    # We want to run the process in a separate thread for each url, to speed things up in case of multiple urls
    def process(self, image_path: str) -> Tuple[str, str]:
        print(f"Processing {image_path}")

        try:
            # Step 1: Resize the image if needed
            resized_image_path = self.resize_image_if_needed(image_path)

            # Step 2: Remove background from the resized image
            self.remove_background(resized_image_path)

            return ("Done:", image_path)
        except Exception as e:
            # indicate failure with error message
            print(f"Error processing {image_path}: {str(e)}")
            return ("Failed.", image_path)

    def resize_image_if_needed(self, image_path: str) -> str:

        # Resize image if it exceeds the maximum pixel count allowed by the API.
        # Preserves aspect ratio.
        # Returns the path to the resized image (or original if no resize needed).
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                total_pixels = width * height

                # Check if image exceeds the max pixel count
                if total_pixels <= self.MAX_PIXELS:
                    print(f"Image {image_path} is within size limits ({total_pixels} pixels)")
                    return image_path

                # Calculate the scaling factor needed
                scale_factor = math.sqrt(self.MAX_PIXELS / total_pixels)

                # Calculate new dimensions, ensuring we round down to be safe
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                # Ensure we're under the limit (just in case of rounding issues)
                while new_width * new_height > self.MAX_PIXELS:
                    new_width -= 1
                    new_height -= 1

                print(f"Resizing {image_path} from {width}x{height} ({total_pixels} pixels) to {new_width}x{new_height} ({new_width * new_height} pixels)")

                # Create resized image
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)

                # Create a temporary file or use a predictable path for the resized image
                resized_path = f"{os.path.splitext(image_path)[0]}_resized{os.path.splitext(image_path)[1]}"
                resized_img.save(resized_path)

                return resized_path
        except Exception as e:
            print(f"Error resizing image {image_path}: {str(e)}")
            raise e

    def remove_background(self, image_path: str) -> str:
        # We will use StableDiffusion API to remove background from our image
        # https://platform.stability.ai/docs/api-reference#tag/Edit/paths/~1v2beta~1stable-image~1edit~1remove-background/post
        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/edit/remove-background",
            headers={
                "authorization": f"Bearer {os.getenv('STABLE_DIFFUSION_API_KEY')}",
                "accept": "image/*"
            },
            files={
                "image": open(image_path, "rb")
            },
            data={
                "output_format": "png"
            },
        )

        if response.status_code == 200:
            output_dir = os.getenv("OUTPUT_DIR") or "./output"

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Extract only the filename from the image path
            filename = os.path.basename(image_path)

            # If we're using a temporary resized image, use the original filename for output
            if "_resized" in filename:
                filename = filename.replace("_resized", "_nobg")

            with open(f"{output_dir}/{filename}", 'wb') as file:
                file.write(response.content)

            # Cleanup the resized temporary file if it was created
            if "_resized" in image_path and os.path.exists(image_path) and image_path != filename:
                try:
                    os.remove(image_path)
                    print(f"Removed temporary file {image_path}")
                except:
                    print(f"Could not remove temporary file {image_path}")
        else:
            error = str(response.json())
            raise Exception(error)

        return image_path
