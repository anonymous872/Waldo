import torch
from typing import Dict
from transformers import AutoProcessor, MllamaForConditionalGeneration, LlavaForConditionalGeneration, AutoModelForImageTextToText
import logging
import queue
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)

class LLavaWorker:
    """Worker class for individual GPU inference."""
    
    def __init__(self, gpu_id: int, model_name: str, work_queue: mp.Queue, 
                 result_queue: mp.Queue, stop_event: mp.Event):
        self.gpu_id = gpu_id
        self.model_name = model_name
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.device = f"cuda:{gpu_id}"
        
    def load_model(self):
        """Load model and tokenizer on specific GPU."""
        logger.info(f"Loading model on GPU {self.gpu_id}")
        
        torch.cuda.set_device(self.gpu_id)
        
        # Load LLaMA 3.2 Vision model (Mllama)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map={"": self.device},
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model.eval()
        logger.info(f"Model loaded successfully on GPU {self.gpu_id}")
    
    def process_sample(self, sample: Dict) -> Dict:
        """Process a single sample."""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": sample["question"]},
                    ]
                }
            ]
            
            # Apply chat template
            prompt = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=False
            )   
            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=sample["image"],
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return {
                "id": sample["id"],
                "question": sample["question"],
                "generated_answer": response,
                "ground_truth": sample.get("answer", ""),
            }
            
        except Exception as e:
            logger.error(f"Error processing sample {sample['id']} on GPU {self.gpu_id}: {str(e)}")
            return {
                "id": sample["id"],
                "question": sample["question"],
                "generated_answer": f"ERROR: {str(e)}",
                "ground_truth": sample.get("answer", ""),
            }
    
    def run(self):
        """Main worker loop."""
        try:
            self.load_model()
            
            processed_count = 0
            while not self.stop_event.is_set():
                try:
                    # Get next work item with timeout
                    sample = self.work_queue.get(timeout=1.0)
                    if sample is None:  # Sentinel value to stop
                        break
                    
                    # Process the sample
                    result = self.process_sample(sample)
                    self.result_queue.put(result)
                    
                    processed_count += 1
                    if processed_count % 10 == 0:
                        logger.info(f"GPU {self.gpu_id}: Processed {processed_count} samples")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Worker GPU {self.gpu_id} error: {str(e)}")
                    break
                logger.info(f"Worker on GPU {self.gpu_id} finished. Processed {processed_count} samples")
                    
        except Exception as e:
            logger.error(f"Failed to initialize worker on GPU {self.gpu_id}: {str(e)}")


class LLamaWorker:
    """Worker class for individual GPU inference."""
    
    def __init__(self, gpu_id: int, model_name: str, work_queue: mp.Queue, 
                 result_queue: mp.Queue, stop_event: mp.Event):
        self.gpu_id = gpu_id
        self.model_name = model_name
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.device = f"cuda:{gpu_id}"
        
    def load_model(self):
        """Load model and tokenizer on specific GPU."""
        logger.info(f"Loading model on GPU {self.gpu_id}")
        
        torch.cuda.set_device(self.gpu_id)
        
        # Load LLaMA 3.2 Vision model (Mllama)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map={"": self.device},
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model.eval()
        logger.info(f"Model loaded successfully on GPU {self.gpu_id}")
    
    def process_sample(self, sample: Dict) -> Dict:
        """Process a single sample."""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": sample["question"]},
                    ]
                }
            ]
            
            # Apply chat template
            prompt = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=False
            )   
            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=sample["image"],
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return {
                "id": sample["id"],
                "question": sample["question"],
                "generated_answer": response,
                "ground_truth": sample.get("answer", ""),
            }
            
        except Exception as e:
            logger.error(f"Error processing sample {sample['id']} on GPU {self.gpu_id}: {str(e)}")
            return {
                "id": sample["id"],
                "question": sample["question"],
                "generated_answer": f"ERROR: {str(e)}",
                "ground_truth": sample.get("answer", ""),
            }
    
    def run(self):
        """Main worker loop."""
        try:
            self.load_model()
            
            processed_count = 0
            while not self.stop_event.is_set():
                try:
                    # Get next work item with timeout
                    sample = self.work_queue.get(timeout=1.0)
                    if sample is None:  # Sentinel value to stop
                        break
                    
                    # Process the sample
                    result = self.process_sample(sample)
                    self.result_queue.put(result)
                    
                    processed_count += 1
                    if processed_count % 10 == 0:
                        logger.info(f"GPU {self.gpu_id}: Processed {processed_count} samples")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Worker GPU {self.gpu_id} error: {str(e)}")
                    break

                logger.info(f"Worker on GPU {self.gpu_id} finished. Processed {processed_count} samples")
                    
        except Exception as e:
            logger.error(f"Failed to initialize worker on GPU {self.gpu_id}: {str(e)}")
        
        

class QwenWorker:
    """Worker class for individual GPU inference."""
    
    def __init__(self, gpu_id: int, model_name: str, work_queue: mp.Queue, 
                 result_queue: mp.Queue, stop_event: mp.Event):
        self.gpu_id = gpu_id
        self.model_name = model_name
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.device = f"cuda:{gpu_id}"
        
    def load_model(self):
        """Load model and tokenizer on specific GPU."""
        logger.info(f"Loading model on GPU {self.gpu_id}")
        
        torch.cuda.set_device(self.gpu_id)
        
        # Load Qwen 2.5 Vision Language model (Qwen2.5-VL)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            device_map={"": self.device},
            #force_download=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            #force_download=True
        )
        
        self.model.eval()
        logger.info(f"Model loaded successfully on GPU {self.gpu_id}")
    
    def process_sample(self, sample: Dict) -> Dict:
        """Process a single sample."""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": sample["question"]},
                    ]
                }
            ]
            
            # Apply chat template
            prompt = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=False
            )   
            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=sample["image"],
                return_tensors="pt",
                #max_pixels = 8000000
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                )
            
            # Decode response
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return {
                "id": sample["id"],
                "question": sample["question"],
                "generated_answer": response,
                "ground_truth": sample.get("answer", ""),
                "input_tokens": inputs["input_ids"].shape,
            }
            
        except Exception as e:
            logger.error(f"Error processing sample {sample['id']} on GPU {self.gpu_id}: {str(e)}")
            return {
                "id": sample["id"],
                "question": sample["question"],
                "generated_answer": f"ERROR: {str(e)}",
                "ground_truth": sample.get("answer", ""),
            }
    
    def run(self):
        """Main worker loop."""
        try:
            self.load_model()
            
            processed_count = 0
            while not self.stop_event.is_set():
                try:
                    # Get next work item with timeout
                    sample = self.work_queue.get(timeout=1.0)
                    if sample is None:  # Sentinel value to stop
                        break
                    
                    # Process the sample
                    result = self.process_sample(sample)
                    self.result_queue.put(result)
                    
                    processed_count += 1
                    if processed_count % 10 == 0:
                        logger.info(f"GPU {self.gpu_id}: Processed {processed_count} samples")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Worker GPU {self.gpu_id} error: {str(e)}")
                    break

                logger.info(f"Worker on GPU {self.gpu_id} finished. Processed {processed_count} samples")
                    
        except Exception as e:
            logger.error(f"Failed to initialize worker on GPU {self.gpu_id}: {str(e)}")
        
       

class PixtralWorker:
    """Worker class for individual GPU inference."""
    
    def __init__(self, gpu_id: int, model_name: str, work_queue: mp.Queue, 
                 result_queue: mp.Queue, stop_event: mp.Event):
        self.gpu_id = gpu_id
        self.model_name = model_name
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.device = f"cuda:{gpu_id}"
        
    def load_model(self):
        """Load model and tokenizer on specific GPU."""
        logger.info(f"Loading model on GPU {self.gpu_id}")
        
        torch.cuda.set_device(self.gpu_id)
        
        # Load Pixtral Vision Language model
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            device_map={"": self.device},
            dtype=torch.float16
            #force_download=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            dtype=torch.float16
            #force_download=True
        )
        
        self.model.eval()
        logger.info(f"Model loaded successfully on GPU {self.gpu_id}")
    
    def process_sample(self, sample: Dict) -> Dict:
        """Process a single sample."""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": sample["question"]},
                    ]
                }
            ]
            
            # Apply chat template
            prompt = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=False
            )   
            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=sample["image"],
                return_tensors="pt"
            ).to(self.device, dtype=self.model.dtype)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                )
            
            # Decode response
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return {
                "id": sample["id"],
                "question": sample["question"],
                "generated_answer": response,
                "ground_truth": sample.get("answer", ""),
            }
            
        except Exception as e:
            logger.error(f"Error processing sample {sample['id']} on GPU {self.gpu_id}: {str(e)}")
            return {
                "id": sample["id"],
                "question": sample["question"],
                "generated_answer": f"ERROR: {str(e)}",
                "ground_truth": sample.get("answer", ""),
            }
    
    def run(self):
        """Main worker loop."""
        self.load_model()
        processed_count = 0
        try: 
            while not self.stop_event.is_set():
                try:
                    # Get next work item with timeout
                    sample = self.work_queue.get(timeout=1.0)
                    if sample is None:  # Sentinel value to stop
                        break
                    
                    # Process the sample
                    result = self.process_sample(sample)
                    self.result_queue.put(result)
                    
                    processed_count += 1
                    if processed_count % 10 == 0:
                        logger.info(f"GPU {self.gpu_id}: Processed {processed_count} samples")
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Worker GPU {self.gpu_id} error: {str(e)}")
                    break

                logger.info(f"Worker on GPU {self.gpu_id} finished. Processed {processed_count} samples")
                    
        except Exception as e:
            logger.error(f"Failed to initialize worker on GPU {self.gpu_id}: {str(e)}")
        
        