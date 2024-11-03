import asyncio
import aiohttp
import random
import time
import io
from PIL import Image
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Base directory for generated images
BASE_OUTPUT_DIR = "artificial_data"

class GeneratorAgent:
    def __init__(self):
        self.API_URL = "https://api-inference.huggingface.co/models/XLabs-AI/flux-RealismLora"
        token = os.getenv('HUGGINGFACE_TOKEN')
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")
            
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "x-use-cache": "false"
        }
        self.guidance_scale = 7.5
        self.num_inference_steps = 50
        self.negative_prompt = ["blurry", "bad quality", "worst quality"]

        # Create base output directory if it doesn't exist
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    async def generate_single_image(self, session, description, class_name, image_index):
        try:
            # Create class directory if it doesn't exist
            class_dir = os.path.join(BASE_OUTPUT_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)

            seed = int(time.time() * 1000) + random.randint(0, 1000000)
            
            payload = {
                "inputs": description,
                "parameters": {
                    "guidance_scale": self.guidance_scale,
                    "num_inference_steps": self.num_inference_steps,
                    "seed": seed
                }
            }
            
            async with session.post(self.API_URL, headers=self.headers, json=payload) as response:
                print(f"Generating image {image_index} for {class_name}")
                print(f"Using seed: {seed}")
                
                if response.status != 200:
                    error_content = await response.text()
                    print(f"Error {response.status}: {error_content}")
                    return
                
                image_bytes = await response.read()
                image = Image.open(io.BytesIO(image_bytes))
                
                # Save image in class-specific directory
                filename = f"{class_name}_{image_index}_{seed}.png"
                save_path = os.path.join(class_dir, filename)
                image.save(save_path)
                print(f"Saved {save_path}")
                
        except Exception as e:
            print(f"Error generating {class_name} image {image_index}: {str(e)}")

    async def generate_images(self, descriptions_dict, nb_images):
        """Generate images for each class in the dictionary."""
        tasks = []
        async with aiohttp.ClientSession() as session:
            for class_name, description in descriptions_dict.items():
                for i in range(nb_images):
                    task = self.generate_single_image(session, description, class_name, i)
                    tasks.append(task)
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    agent = GeneratorAgent()
    descriptions = {
        'dogs': 'high quality photograph of a dog, random background',
        'cats': 'high quality photograph of a cat, random background',
    }
    nb_images = 1
    asyncio.run(agent.generate_images(descriptions, nb_images))