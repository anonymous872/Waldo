from datasets import load_dataset
import logging
from torch.utils.data import Dataset, DataLoader
from .utils import ImageProcessor, ImageProcessor_upscaling
import random
from PIL import Image

logger = logging.getLogger(__name__)

class MMVetDataset(Dataset):
    """Dataset class for MMVet benchmark."""
    
    def __init__(self, dataset_name: str = "whyu/mm-vet", field = None, scale = None, sample_size:int = 200, seed: int = 42):
        random.seed(seed)

        full_dataset = load_dataset(dataset_name)['test']
        indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(indices, min(sample_size, len(full_dataset)))
        self.samples = full_dataset.select(sampled_indices)

        self.field = field
        self.scale = scale

        if self.scale:
            if self.scale in [2,4,8]:
                self.upscaler = ImageProcessor_upscaling(scale = self.scale)
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale}.")
            else: 
                self.scale_new = min(s for s in [2,4,8] if s >= self.scale) 
                self.upscaler = ImageProcessor_upscaling(scale = self.scale_new)
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale_new}.")
                logger.info(f"Image will be downsized to {self.scale} to match the requirement.")

        logger.info(f"Loaded {len(self.samples)} samples from MMVet dataset")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.field:
            img = ImageProcessor.add_white_field(sample['image'], percent=self.field)
        elif self.scale: 
            img_orig = sample['image'].convert('RGB')
            img = self.upscaler.upscale(img_orig)
            
            if self.scale not in [2,4,8]:
                w, h = img_orig.size
                target_size = (
                    int(round(w * self.scale)),
                    int(round(h * self.scale))
                ) 
                img = img.resize(target_size, resample=Image.LANCZOS)
        else:
            img = sample['image']

        return {
            "id": sample["id"],
            "image": img,
            "question": sample["question"],
            "answer": sample["answer"]
        }
    
class MMEvalDataset(Dataset):
    """Dataset class for MMEval benchmark."""
    
    def __init__(self, dataset_name: str = "darkyarding/MME", field = None, scale = None, position='center', seed:int = 42, sample_size:int = 200):
        full_dataset = load_dataset(dataset_name)['test']
        
        # Sample 200 items with fixed seed for reproducibility
        random.seed(seed)
        indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(indices, min(sample_size, len(full_dataset)))
        
        # Select the sampled items
        self.samples = full_dataset.select(sampled_indices)
        self.field = field
        self.scale = scale

        if self.scale:
            if self.scale in [2,4,8]:
                self.upscaler = ImageProcessor_upscaling(scale = self.scale)
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale}.")
            else: 
                self.scale_new = min(s for s in [2,4,8] if s >= self.scale) 
                self.upscaler = ImageProcessor_upscaling(scale = self.scale_new)
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale_new}.")
                logger.info(f"Image will be downsized to {self.scale} to match the requirement.")

        self.position = position
        logger.info(f"Loaded {len(self.samples)} samples from MMEval dataset")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.field:
            img = ImageProcessor.add_white_field(sample['image'], percent=self.field, position=self.position)
        elif self.scale: 
            img_orig = sample['image'].convert('RGB')
            img = self.upscaler.upscale(img_orig)
            
            if self.scale not in [2,4,8]:
                w, h = img_orig.size
                target_size = (
                    int(round(w * self.scale)),
                    int(round(h * self.scale))
                ) 
                img = img.resize(target_size, resample=Image.LANCZOS)
        else:
            img = sample['image']
    
        return {
            "id": sample["question_id"],
            "image": img,
            "question": sample["question"],
            "answer": sample["answer"]
        }

class MathVerseDataset(Dataset):
    """Dataset class for MathVerse benchmark."""
    
    def __init__(self, dataset_name: str = "AI4Math/MathVerse", field = None, scale = None, seed:int = 42, sample_size:int = 200):
        full_dataset = load_dataset(dataset_name, 'testmini')['testmini'] 
        
        # Sample 200 items with fixed seed for reproducibility
        random.seed(seed)
        indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(indices, min(sample_size, len(full_dataset)))
        
        # Select the sampled items
        self.samples = full_dataset.select(sampled_indices) 
        self.field = field
        self.scale = scale

        if self.scale:
            if self.scale in [2,4,8]:
                self.upscaler = ImageProcessor_upscaling(scale = self.scale)
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale}.")
            else: 
                self.scale_new = min(s for s in [2,4,8] if s >= self.scale) 
                self.upscaler = ImageProcessor_upscaling(scale = self.scale_new)
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale_new}.")
                logger.info(f"Image will be downsized to {self.scale} to match the requirement.")

        logger.info(f"Loaded {len(self.samples)} samples from MathVerse dataset")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.field:
            img = ImageProcessor.add_white_field(sample['image'], percent=self.field)
        elif self.scale: 
            img_orig = sample['image'].convert('RGB')
            img = self.upscaler.upscale(img_orig)
            
            if self.scale not in [2,4,8]:
                w, h = img_orig.size
                target_size = (
                    int(round(w * self.scale)),
                    int(round(h * self.scale))
                ) 
                img = img.resize(target_size, resample=Image.LANCZOS)
        else:
            img = sample['image']
    
        return {
            "id": sample["sample_index"],
            "image": img,
            "question": sample["question"],
            "answer": sample["answer"]
        }

class MMStar(Dataset):
    """Dataset class for MMStar benchmark."""
    
    def __init__(self, dataset_name: str = "Lin-Chen/MMStar", field = None, scale = None, seed:int = 42, sample_size:int = 200):
        full_dataset = load_dataset(dataset_name)['val'] 
        
        # Sample 200 items with fixed seed for reproducibility
        random.seed(seed)
        indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(indices, min(sample_size, len(full_dataset)))
        
        # Select the sampled items
        self.samples = full_dataset.select(sampled_indices) 
        
        self.field = field
        self.scale = scale

        if self.scale:
            if self.scale in [2,4,8]:
                self.upscaler = ImageProcessor_upscaling(scale = self.scale)
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale}.")
            else: 
                self.scale_new = min(s for s in [2,4,8] if s >= self.scale) 
                self.upscaler = ImageProcessor_upscaling(scale = self.scale_new)
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale_new}.")
                logger.info(f"Image will be downsized to {self.scale} to match the requirement.")

        logger.info(f"Loaded {len(self.samples)} samples from MMStar dataset")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.field:
            img = ImageProcessor.add_white_field(sample['image'], percent=self.field)
        elif self.scale: 
            img_orig = sample['image'].convert('RGB')
            img = self.upscaler.upscale(img_orig)
            
            if self.scale not in [2,4,8]:
                w, h = img_orig.size
                target_size = (
                    int(round(w * self.scale)),
                    int(round(h * self.scale))
                ) 
                img = img.resize(target_size, resample=Image.LANCZOS)
        else:
            img = sample['image']
    
        return {
            "id": sample["index"],
            "image": img,
            "question": sample["question"],
            "answer": sample["answer"]
        }

class CharXiv(Dataset):
    """Dataset class for CharXiv benchmark."""
    
    def __init__(self, dataset_name: str = "princeton-nlp/CharXiv", field = None, scale = None, seed:int = 42, sample_size:int = 200):
        full_dataset = load_dataset(dataset_name)['validation'] 
        
        # Sample 200 items with fixed seed for reproducibility
        random.seed(seed)
        indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(indices, min(sample_size, len(full_dataset)))
        
        # Select the sampled items
        self.samples = full_dataset.select(sampled_indices) 

        self.field = field
        self.scale = scale

        if self.scale:
            if self.scale in [2,4,8]:
                self.upscaler = ImageProcessor_upscaling(scale = self.scale)
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale}.")
            else: 
                self.scale_new = min(s for s in [2,4,8] if s >= self.scale) 
                self.upscaler = ImageProcessor_upscaling(scale = self.scale_new)
                logger.info(f"Loaded Real_ESRGAN upscaler with scale {self.scale_new}.")
                logger.info(f"Image will be downsized to {self.scale} to match the requirement.")
                
        logger.info(f"Loaded {len(self.samples)} samples from CharXiv dataset")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.field:
            img = ImageProcessor.add_white_field(sample['image'], percent=self.field)
        elif self.scale: 
            img_orig = sample['image'].convert('RGB')
            img = self.upscaler.upscale(img_orig)
            
            if self.scale not in [2,4,8]:
                w, h = img_orig.size
                target_size = (
                    int(round(w * self.scale)),
                    int(round(h * self.scale))
                ) 
                img = img.resize(target_size, resample=Image.LANCZOS)
        else:
            img = sample['image']
    
        return {
            "id": sample["original_figure_path"],
            "image": img,
            "question": sample["reasoning_q"],
            "answer": sample["reasoning_a"]
        }