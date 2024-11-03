import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv
from dataclasses import dataclass
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from transformers import (
    ViTImageProcessor, 
    ViTForImageClassification, 
    TrainingArguments, 
    Trainer
)
from huggingface_hub import login
from evaluate import load

@dataclass
class TrainingConfig:
    folder_path: str
    model_name: str
    resize_size: tuple
    train_percentage: float
    hub_model_id: str
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 2e-4

class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.classes = os.listdir(folder_path)
        self.images = []
        self.labels = []
        
        for label, class_name in enumerate(self.classes):
            class_folder = os.path.join(folder_path, class_name)
            for image_name in os.listdir(class_folder):
                self.images.append(os.path.join(class_folder, image_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class Visualizer:
    @staticmethod
    def plot_training_curves(trainer):
        log_history = trainer.state.log_history
        epochs, train_losses, eval_losses, eval_accuracies = [], [], [], []
        
        for entry in log_history:
            if 'loss' in entry and 'epoch' in entry and 'step' in entry:
                train_losses.append(entry['loss'])
                epochs.append(entry['epoch'])
            if 'eval_loss' in entry and 'eval_accuracy' in entry:
                eval_losses.append(entry['eval_loss'])
                eval_accuracies.append(entry['eval_accuracy'])

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs[:len(eval_losses)], eval_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs[:len(eval_accuracies)], eval_accuracies, 'g-', label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.close()

class TrainerAgent:
    @staticmethod
    def login():
        load_dotenv()
        token = os.getenv('HF_TOKEN')
        if not token:
            raise ValueError("HF_TOKEN not found in environment variables")
        login(token=token)

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.processor = ViTImageProcessor.from_pretrained(config.model_name)
        self.model = None
        self.trainer = None

    def _prepare_dataset(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.resize_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std)
        ])
        
        dataset = CustomImageDataset(self.config.folder_path, transform=transform)
        train_size = int(self.config.train_percentage * len(dataset))
        val_size = len(dataset) - train_size
        
        return random_split(dataset, [train_size, val_size], 
                          generator=torch.Generator().manual_seed(42)), dataset.classes

    def _collate_fn(self, batch):
        images, labels = zip(*batch)
        return {
            'pixel_values': torch.stack(images),
            'labels': torch.tensor(labels)
        }

    def _compute_metrics(self, p):
        accuracy_metric = load("accuracy")
        predictions = np.argmax(p.predictions, axis=1)
        return {"accuracy": accuracy_metric.compute(
            predictions=predictions, references=p.label_ids)["accuracy"]}

    def train(self):
        (train_dataset, val_dataset), class_names = self._prepare_dataset()

        self.model = ViTForImageClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(class_names),
            id2label={str(i): c for i, c in enumerate(class_names)},
            label2id={c: str(i) for i, c in enumerate(class_names)},
            ignore_mismatched_sizes=True
        )

        training_args = TrainingArguments(
            output_dir="./vit-finetuned",
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.config.learning_rate,
            load_best_model_at_end=True,
            push_to_hub=True,
            hub_model_id=self.config.hub_model_id,
            hub_strategy="end",
            report_to='none'
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            data_collator=self._collate_fn,
            tokenizer=self.processor,
        )

        self.trainer.train()
        self.trainer.save_model()
        self.trainer.push_to_hub()
        
        Visualizer.plot_training_curves(self.trainer)
        return self.trainer

    def predict(self, image_path):
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        transform = transforms.Compose([
            transforms.Resize(self.config.resize_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std)
        ])

        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)
        outputs = self.model(pixel_values=image)
        predicted_class = outputs.logits.argmax(-1).item()
        return self.model.config.id2label[str(predicted_class)]

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    config = TrainingConfig(
        folder_path="artificial_data",
        model_name="google/vit-base-patch16-224",
        resize_size=(224, 224),
        train_percentage=0.8,
        hub_model_id=os.getenv('HUB_MODEL_NAME')
    )
    
    trainer_agent = TrainerAgent(config)
    trainer_agent.login()
    trainer_agent.train()