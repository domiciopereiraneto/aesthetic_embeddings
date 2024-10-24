import os
import subprocess
import tempfile
from PIL import Image

class NIMAAsthetics():
    def predict(self, pil_image):
        model_name = 'MobileNet'
        weights_file = 'image-quality-assessment/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5'
        docker_image = 'nima-cpu'
        try:
            # Create a temporary file to save the PIL image
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image_file:
                temp_image_path = temp_image_file.name
                # Save the PIL image to the temporary file
                pil_image.save(temp_image_path, format="JPEG")

            # Build the docker run command
            command = [
                './predict',
                '--docker-image', docker_image,
                '--base-model-name', model_name,
                '--weights-file', weights_file,
                '--image-source', temp_image_path
            ]

            # Execute the command using subprocess
            result = subprocess.run(command, capture_output=True, text=True)

            # Check for errors
            if result.returncode != 0:
                print(f"Error running Docker container: {result.stderr}")
                return None
            
            # Return the classification result (stdout)
            return result.stdout

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

        finally:
            # Remove the temporary image file after classification is done
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)