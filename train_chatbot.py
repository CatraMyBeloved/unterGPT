#!/usr/bin/env python3
"""
train_chatbot.py
----------------
QLoRA fine-tune of Llama-3.2-3B on the Twitch-chat dataset produced by
prepare_chat_data.py, using Unsloth.

Trains on a BASE model (not instruct) -- we want chaotic chat completion, not
a polite assistant. Loss is masked to the COMPLETION only (the <next>...</next>
target), so the model is graded purely on producing the next chat message.

Tested target hardware: single RTX 4060 (8GB). The 3B model at
max_seq_length=1024 (32-message context) is tight on 8GB, so we default to
per_device_train_batch_size=1 with gradient_accumulation_steps=8 (effective
batch 8). If you still OOM, drop MAX_SEQ_LEN back toward 768 or reduce LORA_RANK.

Usage:
    pip install unsloth
    python train_chatbot.py

Outputs a LoRA adapter in ./chatbot_lora and (optionally) a merged GGUF for
Ollama in ./chatbot_gguf.
"""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = "unsloth/Llama-3.2-3B-bnb-4bit"   # base model, pre-quantized (3B)
# Alternative: "unsloth/Qwen2.5-1.5B-bnb-4bit"  (1.5B, faster, less range)

MAX_SEQ_LEN = 1024       # 32-message context ~doubles length vs the old 15;
                         # 1024 covers the p95 with headroom for the target
EPOCHS = 3               # checkpoint each epoch and eyeball output; 3 is the sweet spot
LR = 2e-4                # Unsloth's recommended LoRA starting LR
LORA_RANK = 16
LORA_ALPHA = 16

TRAIN_FILE = "train.jsonl"
VAL_FILE   = "val.jsonl"

# The response marker: loss is computed only on text AFTER this string.
# Must match the prompt/completion boundary from prepare_chat_data.py.
RESPONSE_MARKER = "<next>"

# ---------------------------------------------------------------------------
# Load model + LoRA
# ---------------------------------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,            # auto
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,        # Unsloth default; not useful here
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # all attn + MLP
    use_gradient_checkpointing="unsloth",  # big VRAM saver
    random_state=42,
)

# Make sure there is a pad token distinct from eos (avoids the model learning
# to never stop). Most Qwen/Llama tokenizers already set this; guard anyway.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data_files = {"train": TRAIN_FILE}
import os
if os.path.exists(VAL_FILE):
    data_files["validation"] = VAL_FILE

ds = load_dataset("json", data_files=data_files)

# We train on the "text" field (prompt + completion). train_on_responses_only
# below restricts the loss to the part after RESPONSE_MARKER.
EOS = tokenizer.eos_token

def add_eos(example):
    # Ensure every sample ends with EOS so the model learns to STOP.
    # </next> is our semantic stop, EOS is the hard stop -- include both.
    text = example["text"]
    if not text.endswith(EOS):
        text = text + EOS
    return {"text": text}

ds = ds.map(add_eos)

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
cfg = SFTConfig(
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    per_device_train_batch_size=1,     # 3B @ 1024 seq len is tight on 8GB
    gradient_accumulation_steps=8,     # effective batch size 8
    warmup_ratio=0.03,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=20,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    output_dir="chatbot_outputs",
    save_strategy="epoch",             # checkpoint every epoch -> compare them
    report_to="none",
    # eval
    do_eval="validation" in ds,
    eval_strategy="epoch" if "validation" in ds else "no",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds.get("validation"),
    args=cfg,
)

# Mask loss to the completion only: everything after the LAST RESPONSE_MARKER.
# This is the single most important line for grading the model on the right thing.
trainer = train_on_responses_only(
    trainer,
    instruction_part=RESPONSE_MARKER,   # text up to & incl. this is masked out
    response_part="",                   # (Unsloth handles the split internally)
)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Training on: {gpu}")
    trainer.train()

    # Save the LoRA adapter (tiny, ~50-100MB)
    model.save_pretrained("chatbot_lora")
    tokenizer.save_pretrained("chatbot_lora")
    print("Saved adapter -> ./chatbot_lora")

    # Export a merged GGUF for Ollama (q4_k_m = good size/quality balance)
    # Comment out if you only want the adapter.
    model.save_pretrained_gguf("chatbot_gguf", tokenizer,
                               quantization_method="q4_k_m")
    print("Saved GGUF -> ./chatbot_gguf  (point your Ollama Modelfile at this)")
