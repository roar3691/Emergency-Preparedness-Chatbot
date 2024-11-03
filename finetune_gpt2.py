import os
import torch
import pdfplumber
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Generalized path to your dataset directory (update this path as needed)
dataset_path = 'data/Dataset-2'

# Generalized path to save and load checkpoints (update this path as needed)
checkpoint_dir = 'gpt2-finetuned-emergency/checkpoint-18'

def extract_text_from_pdfs(directory):
    """
    Extract text from all PDF files in the given directory.
    
    Args:
        directory (str): The path to the directory containing PDF files.
        
    Returns:
        list: A list of strings where each string contains the extracted text from a PDF.
    """
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            with pdfplumber.open(file_path) as pdf:
                # Extract text from all pages of the PDF and combine them into one string
                full_text = ''.join([page.extract_text() for page in pdf.pages])
                texts.append(full_text)
    return texts

# Extract text from all PDFs in the dataset directory
pdf_texts = extract_text_from_pdfs(dataset_path)

# Convert extracted texts into a Hugging Face Dataset
data_dict = {"text": pdf_texts}
dataset = Dataset.from_dict(data_dict)

# Split into train and eval datasets (e.g., 80% train, 20% eval)
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Load GPT-2 tokenizer and set pad_token as eos_token
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    """
    Tokenize the input text data using GPT-2 tokenizer.
    
    Args:
        examples (dict): A dictionary containing the input text data.
        
    Returns:
        dict: A dictionary containing tokenized input data and labels.
    """
    inputs = tokenizer(examples['text'], padding='max_length', truncation=True)
    inputs["labels"] = inputs["input_ids"].copy()  # Set labels as input_ids for autoregressive training
    return inputs

# Tokenize both train and eval datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Load pre-trained GPT-2 model and move it to MPS or CPU based on availability.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Start training from scratch (ignore checkpoints)
print("Starting training from scratch.")
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

# Define training arguments with support for saving checkpoints.
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-emergency",  # Output directory for saving checkpoints
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    per_device_train_batch_size=1,  # Reduce batch size to fit in memory (MPS has limited memory)
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    gradient_accumulation_steps=4,  # Simulate larger batch size via gradient accumulation.
    bf16=True if torch.backends.mps.is_available() else False,  # Enable bf16 mixed precision if using MPS.
)

# Initialize Trainer with both train and eval datasets.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

# Start training (fine-tuning) from scratch.
trainer.train()

# Save model and tokenizer after fine-tuning.
model.save_pretrained(checkpoint_dir)
tokenizer.save_pretrained(checkpoint_dir)

print(f"Model and tokenizer saved to {checkpoint_dir}")
