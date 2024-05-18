from datasets import load_dataset # type: ignore
from trl import SFTTrainer, setup_chat_format # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging # type: ignore
import accelerate # type: ignore
import bitsandbytes # type: ignore
import torch # type: ignore
import sys
from peft import LoraConfig # type: ignore

hfReadToken = sys.argv[1]
hfWriteToken = sys.argv[2]

#Choose the model that will be fine tuned
modelID = "meta-llama/Llama-2-7b-chat-hf"

#load the training dataset
dataset = load_dataset("DreadN0ugh7/ChatAcademyTrainDataset", split = "train", token = hfReadToken)

#This loads the model and tokenizer
quantizationConfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    modelID,
    device_map="auto",
    torch_dtype=torch.bfloat16, 
    quantization_config=quantizationConfig,
    token = hfReadToken)

tokenizer = AutoTokenizer.from_pretrained(modelID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#Setup the peft config
peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)

#Setup the training arguments
args = TrainingArguments(
    output_dir = "/mnt/parscratch/users/aca19sjs/ChatAcademy/ChatAcademy-Trained-7b",
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=3,          # batch size per device during training
    gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="no",                     # save only final model
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper                              # use bfloat16 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    save_total_limit = 1,
   # push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)

#Setup the trainer
max_seq_length = 3072

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field='text',
    max_seq_length=max_seq_length,
    tokenizer=tokenizer
)

# start training, the model will be automatically saved to the hub and the output directory
trainer.train()

trainer.save_model("/mnt/parscratch/users/aca19sjs/ChatAcademy/ChatAcademy-Trained-7b")
trainer.tokenizer.save_pretrained("/mnt/parscratch/users/aca19sjs/ChatAcademy/ChatAcademy-Trained-7b")
# save model
trainer.push_to_hub("ChatAcademy-Trained-7b", token = hfWriteToken)