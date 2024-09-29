# SDXL Model Training Interface

This repository provides a Streamlit application for training your own Stable Diffusion XL (SDXL) models. The interface allows you to configure training parameters, manage datasets, generate images, and upload datasets to the Hugging Face Hub.

## Features

- Interactive Streamlit UI for configuring model training parameters.
- Support for custom datasets with image-caption pairs.
- Image generation at specified intervals during training.
- Capability to generate images from prompts using the trained model.
- Upload datasets to the Hugging Face Hub.

## Requirements

- **Python 3.8** or higher.
- **CUDA-enabled GPU** for training and inference.
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Datasets](https://huggingface.co/docs/datasets/index)
- [Hugging Face Hub](https://github.com/huggingface/huggingface_hub)
- **Diffusers** library (must be installed separately; see [Installation](#installation))
- A [Hugging Face account](https://huggingface.co/join) and [access token](https://huggingface.co/settings/tokens)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/birdhouses/SDXL-Trainer.git
   cd SDXL-Trainer
   ```

2. **Install the required Python packages:**

   ```bash
   pip install streamlit torch transformers datasets huggingface_hub
   ```

3. **Install the `diffusers` library separately:**

   The `diffusers` library is a core component for running diffusion models like SDXL. It is not included as a dependency in this repository and must be installed separately.

   ```bash
   pip install diffusers
   ```

   Alternatively, install it from the source for the latest features:

   ```bash
   git clone https://github.com/huggingface/diffusers.git
   cd diffusers
   pip install -e .
   ```

   For detailed installation instructions and troubleshooting, refer to the [Diffusers GitHub Repository](https://github.com/huggingface/diffusers).

4. **Ensure CUDA is properly configured:**

   - Verify that your system has a CUDA-enabled GPU.
   - Install the appropriate NVIDIA drivers and CUDA toolkit compatible with PyTorch.
   - For guidance, refer to the [PyTorch CUDA Installation Guide](https://pytorch.org/get-started/locally/).

## Usage

### 1. Obtain a Hugging Face Access Token

- Sign up for a Hugging Face account [here](https://huggingface.co/join).
- Navigate to your [access tokens page](https://huggingface.co/settings/tokens).
- Create a new token with the necessary permissions (write access) and save it securely.

### 2. Prepare Your Dataset

- **Dataset Structure:**

  Organize your dataset in a directory where each image has an accompanying `.txt` file containing its caption. For example:

  ```
  dataset/
  ├── image1.jpg
  ├── image1.txt
  ├── image2.jpg
  ├── image2.txt
  └── ...
  ```

- **Automatic Annotation:**

  If you need to automatically generate annotations (the `.txt` files), use the [ImageAnnotator](https://github.com/birdhouses/ImageAnnotator) tool.

- **Image Scraping:**

  To scrape images for your dataset, use the [image_scraper](https://github.com/birdhouses/image_scraper) tool.

### 3. Set Up the Training Script

- **Training Script Path:**

  You need to point to the `train_text_to_image_sdxl.py` script from the `diffusers` library. This script is essential for training the SDXL model.

- **Locate the Script:**

  If you've installed `diffusers` from the source, the training script is typically located at:

  ```
  /path/to/diffusers/examples/text_to_image/train_text_to_image_sdxl.py
  ```

- **Modify the Script (Optional):**

  Ensure that the training script is compatible with your training parameters and setup. You might need to adjust paths or parameters within the script.

### 4. Run the Streamlit Application

Start the Streamlit app from the terminal:

```bash
streamlit run app.py
```

### 5. Configure Training Parameters

In the Streamlit interface:

- **Model Path:** Path to your SDXL model (e.g., `stabilityai/stable-diffusion-xl-base-1.0`).
- **Dataset Path:** Path to your dataset directory.
- **VAE Model Path:** Path to your VAE model (e.g., `madebyollin/sdxl-vae-fp16-fix`).
- **Output Directory:** Path where outputs and checkpoints will be saved.
- **Training Script Path:** Path to your `train_text_to_image_sdxl.py` script from the `diffusers` library.
- **Validation Prompt:** Prompt used for generating validation images during training.
- **Training Parameters:** Set resolution, batch size, learning rate, etc.
- **Hugging Face Token:** Enter your Hugging Face access token when prompted.

### 6. Start Training

- Click **Create Dataset and Train Model** to begin training.
- The application will:

  1. Load your dataset and create a `metadata.csv` file if it doesn't exist.
  2. Initialize the diffusion pipeline using your specified model.
  3. Start the training process by invoking the training script.
  4. Generate images at specified intervals during training.

- Monitor training progress and generated images in the interface.

### 7. Generate Images

- Use the **Generate Image** section to generate images from prompts using the trained model.
- Enter a prompt and click **Generate Image**.
- The generated image will be displayed and saved in the output directory.

### 8. Upload Dataset to Hugging Face Hub

- Provide the path to your dataset and the desired dataset name on Hugging Face Hub.
- Enter your Hugging Face access token.
- Click **Upload Dataset To HuggingFace** to upload your dataset.

## Additional Resources

- **Diffusers Documentation:** [Diffusers GitHub Repository](https://github.com/huggingface/diffusers)
- **Automatic Annotation:** [ImageAnnotator](https://github.com/birdhouses/ImageAnnotator)
- **Image Scraping:** [image_scraper](https://github.com/birdhouses/image_scraper)

## Notes

- **Diffusers Library:**

  The `diffusers` library is not included as a dependency in this repository. You must install it separately as per the instructions above. This library provides the necessary tools and scripts for diffusion models like SDXL.

- **Training Script Path:**

  Ensure that the `script_path` provided in the application points to the `train_text_to_image_sdxl.py` script from the `diffusers` library. This script is essential for training and must be correctly referenced.

- **Hardware Requirements:**

  Training diffusion models is resource-intensive. Ensure your hardware meets the requirements (a powerful GPU with sufficient VRAM).

- **Paths and Permissions:**

  Make sure all paths provided in the interface are correct and accessible, and you have the necessary read/write permissions.

## Troubleshooting

- **CUDA Errors:**

  - Update your GPU drivers and verify CUDA compatibility with PyTorch.
  - Ensure that PyTorch is installed with CUDA support.

- **Missing Dependencies:**

  - Install missing packages using `pip install -r requirements.txt`.

- **Authentication Errors:**

  - Confirm your Hugging Face token is correct and has the necessary permissions.
  - Re-enter your token if authentication fails.

- **Diffusers Installation Issues:**

  - If you encounter issues with `diffusers`, refer to their [GitHub Repository](https://github.com/huggingface/diffusers) for installation guides and troubleshooting.

## How the Application Works

### Core Components

- **Streamlit Interface:**

  Provides an interactive UI for configuring training parameters and managing datasets.

- **Dataset Preparation:**

  The application creates a `metadata.csv` file pairing images with their captions if it doesn't exist.

- **Training Process:**

  Utilizes the `train_text_to_image_sdxl.py` script from the `diffusers` library to train the model based on your configurations.

- **Image Generation:**

  Generates images at specified intervals during training and allows for prompt-based image generation using the trained model.

- **Uploading to Hugging Face Hub:**

  Allows you to upload your prepared dataset to the Hugging Face Hub for sharing or further training.

## Contribution

Contributions are welcome! Please open an issue or submit a pull request.
