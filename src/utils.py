from PIL import Image
import base64
import tempfile
import logging
import queue
import torch
import numpy as np
from .RealESRGAN import RealESRGAN

logger = logging.getLogger(__name__)

def collect_results(result_queue, results, dataset, stop_event):
    while True:
        try:
            result = result_queue.get(timeout=1.0)
            if result is None:  # Sentinel
                break
            results.append(result)
            if len(results) % 50 == 0:
                logger.info(f"Collected {len(results)}/{len(dataset)} results")
        except queue.Empty:
            if stop_event.is_set():
                break

def encode_image(image):
  # Convert image to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Save image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        temp_image_path = tmp_file.name
        image.save(temp_image_path, format="JPEG")

    with open(temp_image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class ImageProcessor:
    @staticmethod
    def pad_image_to_square(img: Image.Image, resize: bool = True, dim: int = 448, 
                           background_color: tuple = (255, 255, 255)) -> Image.Image:
        width, height = img.size
        max_dim = max(width, height)
        new_img = Image.new("RGB", (max_dim, max_dim), background_color)
        paste_position = ((max_dim - width) // 2, (max_dim - height) // 2)
        new_img.paste(img, paste_position)
        return new_img.resize((dim, dim)) if resize else new_img
    
    @staticmethod
    def add_white_field(img: Image.Image, 
                        percent: float = 0.2, 
                        position: str = 'center', 
                        squire: bool = False,
                        aspect: bool = False,
                        background_color: tuple = (255, 255, 255)
                        ) -> Image.Image:
        width, height = img.size

        if squire: # Make the image square
            max_dimension = max(width, height)
            background_size = int(max_dimension * (1 + percent))
            background = Image.new("RGB", (background_size, background_size), background_color)
        elif aspect: # Maintain aspect ratio
            background_width = int(width * (1 + percent))
            background_height = int(height * (1 + percent))
            background = Image.new("RGB", (background_width, background_height), background_color)
        else: # Add fields based on the larger dimension for aspect ratio sensitivity stress testing
            if width > height:
                background_width = int(width * ((1 + percent)**2))
                background_height = height
                background = Image.new("RGB", (background_width, background_height), background_color)
            else:
                background_height = int(height * ((1 + percent)**2))
                background_width = width
                background = Image.new("RGB", (background_width, background_height), background_color)
        
        if position == 'center':
            pos = ((background.size[0] - width) // 2, (background.size[1] - height) // 2)
        elif position == 'bottom':
            pos = (0, 0)
        elif position == 'top':
            pos = (background.size[0] - width, background.size[1] - height)
        else:
            raise ValueError(f"Invalid position: {position}")
            
        background.paste(img, pos)
        return background
    

class ImageProcessor_upscaling:
    """Image super-resolution class using Real-ESRGAN model"""
    def __init__(self, scale):
        self.scale = scale
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RealESRGAN(device, scale=self.scale)
        self.model.load_weights(f'weights/RealESRGAN_x{self.scale}.pth', download=True)  
    
    def upscale(self, img: Image.Image) -> Image.Image:            
        return self.model.predict(img)