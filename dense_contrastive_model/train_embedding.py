import os
import numpy as np

from datasets import load_dataset, Dataset


from sklearn.metrics.pairwise import cosine_similarity
            
from sentence_transformers.losses import MultipleNegativesRankingLoss, AnglELoss
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
import pandas as pd



MODEL_NAME = "intfloat/multilingual-e5-large-instruct"#"BAAI/bge-m3"
OUTPUT_PATH = "./e5_full_hard_neg"
MODEL_OUTPUT_PATH = f"{OUTPUT_PATH}/trained_model"


EPOCH = 2
LR = 2e-05
BS = 8
GRAD_ACC_STEP = 128 // BS

import torch
torch.cuda.empty_cache()



df_train = pd.read_csv('full_hard_neg.csv')
train = Dataset.from_pandas(df_train)

model = SentenceTransformer(MODEL_NAME)

loss = MultipleNegativesRankingLoss(model)

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=OUTPUT_PATH,
    # Optional training parameters:
    num_train_epochs=EPOCH,
    per_device_train_batch_size=BS,
    gradient_accumulation_steps=GRAD_ACC_STEP,
    per_device_eval_batch_size=BS,
    eval_accumulation_steps=GRAD_ACC_STEP,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    lr_scheduler_type="cosine_with_restarts",
    save_strategy="epoch",
    save_steps=0.1,
    logging_steps=100,
    save_total_limit=3,
    report_to='none',  # Will be used in W&B if `wandb` is installed
    do_eval=False,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train,
    loss=loss
)

trainer.train()
model.save_pretrained(MODEL_OUTPUT_PATH)



