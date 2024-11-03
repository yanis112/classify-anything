# Classify Anything 🖼️🔍

Welcome to **Classify Anything**, a project that leverages the power of Vision Transformers (ViT) to classify images into various  using 0 training data ! Only artificialialy generated images ! This repository provides tools to train a custom image classifier using the Hugging Face Transformers library and manage your models with the Hugging Face Hub.

## Table of Contents 📖

- Project Overview
- Project Structure
- Setup 🔧
- Training the Model 🏋️‍♂️
- Managing Hugging Face Hub 🤗
- Inference 🔮
- Folder Descriptions 📂
- Additional Notes 📝

## Project Overview 📝

This project aims to classify images using a fine-tuned Vision Transformer model. It utilizes the Hugging Face Transformers library for model implementation and training, allowing for easy integration with the Hugging Face ecosystem, including the Hub for model sharing and management.

## Project Structure 🗂️

```
classify-anything/
├── .env
├── .gitignore
├── artificial_data/
│   ├── cats/
│   └── dogs/
├── config.yaml
├── outputs/
│   └── vit-finetuned/
│       ├── all_results.json
│       ├── checkpoint-12/
│       ├── checkpoint-4/
│       ├── config.json
│       └── ...
├── poetry.lock
├── pyproject.toml
├── README.md
├── real_data/
├── src/
│   ├── classification_utils/
│   │   ├── fine_tuning_vit.py
│   │   └── ...
│   ├── generation_utils/
│   ├── inference.py
│   └── ...
└── vit-finetuned/
    ├── checkpoint-1/
    ├── checkpoint-10/
    ├── config.json
    ├── model.safetensors
    └── ...
```

## Setup 🔧

To get started with the project, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your_username/classify-anything.git
   cd classify-anything
   ```

2. **Install Dependencies**:

   We use [Poetry](https://python-poetry.org/) for dependency management. Install Poetry if you haven't:

   ```bash
   pip install poetry
   ```

   Then, install the project dependencies:

   ```bash
   poetry install
   ```

3. **Environment Variables**:

   Create a .env file in the root directory and add your Hugging Face credentials:

   ```env
   HUGGINGFACE_TOKEN=your_huggingface_token
   HUB_MODEL_NAME=your_model_name_on_hub
   ```

4. **Configure Settings**:

   Edit the config.yamlc file to adjust parameters like dataset paths, model configurations, and training arguments.

## Training the Model 🏋️‍♂️

To train the Vision Transformer model:

1. **Prepare Datasets**:

   - Place your training images in the artificial_data directory, organized into subdirectories for each class (e.g., `cats/`, `dogs/`).
   - For real-world data, use the real_data directory.

2. **Run Training Script**:

   Navigate to the `classification_utils` directory and execute the training script:

   ```bash
   cd src/classification_utils
   poetry run python fine_tuning_vit.py
   ```

   This script will:

   - Load and preprocess the dataset.
   - Fine-tune the ViT model.
   - Save checkpoints in the vit-finetuned directory.
   - Push the model to the Hugging Face Hub if configured.

## Managing Hugging Face Hub 🤗

The project is integrated with the Hugging Face Hub for model versioning and sharing.

- **Login to Hugging Face**:

  The TrainerAgent class handles authentication. Ensure your `HUGGINGFACE_TOKEN` is set in the .env file.

- **Push Model to Hub**:

  The training script automatically pushes the model to the Hub at the end of training. You can adjust this behavior in the TrainingArguments:
  ```python
  training_args = TrainingArguments(
      ...
      push_to_hub=True,
      hub_model_id=config.hub_model_id,
      hub_strategy="end",
      ...
  )
  ```

## Inference 🔮

To perform inference using the trained model:

1. **Instantiate the Classifier**:

   ```python
   from src.inference import ImageClassifier

   classifier = ImageClassifier(
       hub_model_name="path/to/your/model",
       base_model_name="google/vit-base-patch16-224"
   )
   ```

2. **Predict an Image**:

   ```python
   prediction = classifier.predict_single_image("path/to/your/image.jpg")
   print(f"Predicted class: {prediction}")
   ```

## Folder Descriptions 📂

- artificial_data: Contains synthetic training images organized by class.

- real_data: Holds real-world data for testing or additional training.

- outputs: Stores output files, logs, and results from training sessions.

- vit-finetuned: Contains the fine-tuned ViT model checkpoints and configuration files.

- src: Main source code directory.

- `classification_utils/`: Utilities and scripts for training the classifier.

- fine_tuning_vit.py: Main training script for fine-tuning the ViT model.

- `generation_utils/`: Contains scripts related to data generation (if applicable).

- inference.py: Script for loading the trained model and performing inference.

- .env: Environment variables file (should not be committed to version control).
- .gitignore: Specifies intentionally untracked files to ignore.

- pyproject.toml & poetry.lock: Project dependency and environment configuration files managed by Poetry.

## Additional Notes 📝

- **Model Configuration**:

  All configurations for training are centralized in the config.yaml file for easy adjustments.

- **Logging and Visualization**:

  The Visualizer class (found in fine_tuning_vit.py) provides utilities to plot training curves and metrics.

- **Environment Management**:

  Ensure that the Python version matches the one specified in pyproject.toml (`python = "^3.11"`).

- **Handling Checkpoints**:

  Checkpoints are saved after each epoch. Use them to resume training or for version control.

- **Using Accelerate**:

  The project leverages the accelerate library for optimized training, especially on multi-GPU setups. Feel free to contribute to this project or raise issues if you find any bugs! 🙌