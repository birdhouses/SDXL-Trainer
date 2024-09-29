import streamlit as st
from pathlib import Path
import subprocess
import os
import csv
import sys
import re
from diffusers import DiffusionPipeline
import torch
import uuid
from datasets import load_dataset, Dataset, Features, Image, Value
import pandas as pd
from huggingface_hub import login

def create_metadata_csv(data_dir, metadata_path):
    image_extensions = {'.png', '.jpg', '.jpeg'}
    with open(metadata_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_name', 'text'])
        for file in os.listdir(data_dir):
            if os.path.splitext(file)[1].lower() in image_extensions:
                txt_file = os.path.splitext(file)[0] + '.txt'
                txt_path = os.path.join(data_dir, txt_file)
                caption = ""
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                writer.writerow([file, caption])

def load_image_dataset(data_dir):
    metadata_path = os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        create_metadata_csv(data_dir, metadata_path)
    dataset = load_dataset("imagefolder", data_dir=data_dir)
    return dataset

def generate_image(pipe, prompt, output_dir, step=None):
    os.makedirs(output_dir, exist_ok=True)
    base_name = str(uuid.uuid4())
    image = pipe(prompt=prompt).images[0]
    if step is not None:
        image_path = os.path.join(output_dir, f"{base_name}_step_{step}.png")
    else:
        image_path = os.path.join(output_dir, f"{base_name}.png")
    image.save(image_path)
    st.image(image, caption=f"Generated Image at Step {step}" if step is not None else "Generated Image")
    print(f"Generated image at step {step} and saved to {image_path}")
    return image

def train_model(script_path, model_path, dataset_path, output_dir, vae_name, validation_prompt, resolution, batch_size, accumulation_steps, max_train_steps, learning_rate, validation_epochs, checkpointing_steps, proportion_empty_prompts, image_interval):
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True)
    pipe.to("cuda")
    command = [
        "accelerate", "launch", script_path,
        f"--pretrained_model_name_or_path={model_path}",
        f"--pretrained_vae_model_name_or_path={vae_name}",
        f"--train_data_dir={dataset_path}",
        f"--resolution={resolution}",
        "--center_crop",
        "--random_flip",
        f"--proportion_empty_prompts={proportion_empty_prompts}",
        f"--train_batch_size={batch_size}",
        f"--gradient_accumulation_steps={accumulation_steps}",
        "--gradient_checkpointing",
        f"--max_train_steps={max_train_steps}",
        "--use_8bit_adam",
        f"--learning_rate={learning_rate}",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--mixed_precision=fp16",
        f"--validation_prompt={validation_prompt}",
        f"--validation_epochs={validation_epochs}",
        f"--checkpointing_steps={checkpointing_steps}",
        f"--output_dir={output_dir}",
        "--push_to_hub"
    ]
    progress_bar = st.progress(0)
    status_text = st.empty()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    current_step = 0
    total_steps = max_train_steps
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        step_match = re.search(r'Step (\d+)/(\d+)', line)
        if step_match:
            current_step = int(step_match.group(1))
            progress_percentage = current_step / total_steps
            progress_bar.progress(progress_percentage)
            status_text.text(f"Training progress: Step {current_step}/{total_steps}")
            if current_step % image_interval == 0:
                generate_image(pipe, validation_prompt, output_dir, current_step)
        sys.stdout.flush()
    process.stdout.close()
    process.wait()
    generate_image(pipe, validation_prompt, output_dir, step=current_step)
    progress_bar.progress(1.0)
    status_text.text("Training completed.")
    print("Training completed.")

st.title("Train Your SDXL Model")
model_path = st.text_input("Enter the path to your SDXL model:", value="stabilityai/stable-diffusion-xl-base-1.0")
dataset_path = st.text_input("Enter the path to your dataset:", value="/path/to/your/dataset")
vae_name = st.text_input("Enter the path to your VAE model:", value="madebyollin/sdxl-vae-fp16-fix")
output_dir = st.text_input("Enter the path to your output directory:", value="/path/to/output/dir")
script_path = st.text_input("Enter the path to your training script:", value="/path/to/train_text_to_image_sdxl.py")
validation_prompt = st.text_input("Enter the validation prompt:", value="a cute creature")
resolution = st.number_input("Enter the image resolution:", value=512, min_value=128, max_value=1024)
batch_size = st.number_input("Enter the training batch size:", value=1, min_value=1, max_value=64)
accumulation_steps = st.number_input("Enter the gradient accumulation steps:", value=4, min_value=1, max_value=100)
max_train_steps = st.number_input("Enter the maximum training steps:", value=10000, min_value=1, max_value=1000000)
learning_rate = st.number_input("Enter the learning rate:", value=1e-06, format="%e")
validation_epochs = st.number_input("Enter the validation epochs:", value=5, min_value=1, max_value=100)
checkpointing_steps = st.number_input("Enter the checkpointing steps:", value=5000, min_value=1, max_value=100000)
proportion_empty_prompts = st.slider("Enter the proportion of empty prompts:", value=0.2, min_value=0.0, max_value=1.0, step=0.01)
image_interval = st.number_input("Generate an image every X steps:", value=1000, min_value=1)
if st.button("Create Dataset and Train Model"):
    if Path(dataset_path).exists():
        st.write("Loading dataset...")
        dataset = load_image_dataset(dataset_path)
        st.write(f"Dataset loaded successfully from {dataset_path}")
        st.write("Starting training...")
        train_model(
            script_path,
            model_path,
            dataset_path,
            output_dir,
            vae_name,
            validation_prompt,
            resolution,
            batch_size,
            accumulation_steps,
            max_train_steps,
            learning_rate,
            validation_epochs,
            checkpointing_steps,
            proportion_empty_prompts,
            image_interval
        )
    else:
        st.write("Please make sure the specified dataset path exists.")
prompt_input = st.text_input("Enter a prompt to generate an image:")
if st.button("Generate Image"):
    if prompt_input:
        pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe.to("cuda")
        generated_image = generate_image(pipe, prompt_input, output_dir)
        st.image(generated_image, caption="Generated Image", use_column_width=True)
    else:
        st.write("Please enter a prompt to generate an image.")
data_dir = st.text_input("Enter the path to your dataset to upload:", value="/path/to/your/dataset")
data_name = st.text_input("Enter the name of your dataset on HuggingFace Hub:", value="your-username/your-dataset-name")
hf_token = st.text_input("Enter your HuggingFace token:", type="password")
if st.button("Upload Dataset To HuggingFace"):
    if hf_token:
        login(token=hf_token)
        metadata_path = os.path.join(data_dir, "metadata.csv")
        create_metadata_csv(data_dir, metadata_path)
        df = pd.read_csv(metadata_path)
        df['image'] = df['file_name'].apply(lambda x: os.path.join(data_dir, x))
        df.drop(columns=['file_name'], inplace=True)
        df = df[df['text'].apply(lambda x: isinstance(x, str))]
        df.reset_index(drop=True, inplace=True)
        features = Features({'image': Image(), 'text': Value("string")})
        dataset = Dataset.from_pandas(df, features=features)
        dataset.push_to_hub(data_name, private=True)
        st.success(f"Dataset '{data_name}' uploaded successfully to Hugging Face Hub!")
    else:
        st.write("Please enter your HuggingFace token.")
