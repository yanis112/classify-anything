
import os
from PIL import Image
import torch
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification
from huggingface_hub import login

class ImageClassifier:
    """A class to perform image classification using Vision Transformer models."""

    def __init__(self, hub_model_name: str, base_model_name: str = "google/vit-base-patch16-224"):
        """
        Initialize the image classifier.

        Args:
            hub_model_name: Name of the fine-tuned model on HuggingFace Hub
            base_model_name: Name of the base ViT model
        """
        self.hub_model_name = hub_model_name
        self.base_model_name = base_model_name
        self.resize_size = (224, 224)
        self.supported_formats = ('.png', '.jpg', '.jpeg', '.webp')
        
        # Initialize model and processor
        self._load_models()

    def _load_models(self):
        """Load the ViT model and processor."""
        login(token="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")
        self.model = ViTForImageClassification.from_pretrained(self.hub_model_name)
        self.processor = ViTImageProcessor.from_pretrained(self.base_model_name)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std)
        ])

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image tensor
        """
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0)

    def predict_single_image(self, image_path: str) -> str:
        """
        Predict class for a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Predicted class label
        """
        image = self.preprocess_image(image_path)
        inputs = {'pixel_values': image}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        predicted_class = outputs.logits.argmax(-1).item()
        
        return self.model.config.id2label[predicted_class]

    def run_batch_inference(self, data_path: str):
        """
        Run inference on all supported images in a directory.

        Args:
            data_path: Path to directory containing images
        """
        for file_name in os.listdir(data_path):
            if file_name.lower().endswith(self.supported_formats):
                image_path = os.path.join(data_path, file_name)
                try:
                    prediction = self.predict_single_image(image_path)
                    print(f"File: {file_name} -> Predicted class: {prediction}")
                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")

def main():
    """Main function to demonstrate usage."""
    from dotenv import load_dotenv
    load_dotenv()
    
    classifier = ImageClassifier(hub_model_name=os.getenv("HUB_MODEL_NAME"))
    classifier.run_batch_inference("real_data")

if __name__ == "__main__":
    main()