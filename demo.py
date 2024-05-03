# -*- coding: utf-8 -*-
import os

import sys
import nest_asyncio

import numpy as np
import gradio as gr

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging

from llama_index.core import VectorStoreIndex, PromptTemplate, Settings, Document, Response
from llama_index.core.evaluation import FaithfulnessEvaluator,RelevancyEvaluator,CorrectnessEvaluator, BatchEvalRunner
from llama_index.core.llama_dataset import LabelledRagDataset, CreatedBy, CreatedByType, LabelledRagDataExample
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI

from readers.authorReader import authorReader
from readers.grantReader import grantReader
from readers.journalReader import journalReader
from readers.publicationReader import publicationReader

from bs4 import BeautifulSoup
import torch
import calendar
import spacy
import asyncio
import random

nest_asyncio.apply()

#CONTROL PANEL - USE THIS TO CHANE THINGS FOR EXPERIMENTATION

#Control the language models in use
baseModel = "meta-llama/Llama-2-7b-chat-hf"

#Control the embedding models in use
embedModel = "local:BAAI/bge-small-en-v1.5"

#Control the chunk Size and Overlap (Due to metadata, using less than 700 is not possible.)
chunkSize = 1024
chunkOverlap = 50

#Control the number of documents retrieved from the index
topK_Retrieved = 10

#Control which dataset is loaded and where the results are saved.
dataSetFileName = "DefaultEvaluationSet"
saveFileName = "results7b"

#Control how many questions are used to test on the LLM (Using more will take longer)
sampleSize = 200

#END OF THE CONTROL PANEL

hf_token = "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"

#Actually builds the LLM
compute_dtype = getattr(torch, "float16")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

llm = HuggingFaceLLM(
    model_name= baseModel,
    tokenizer_name= baseModel,
    query_wrapper_prompt=PromptTemplate("<s> [INST] {query_str} [/INST] "),
    context_window=3900,
    model_kwargs={"token": hf_token, "quantization_config": quantization_config},
    tokenizer_kwargs={"token": hf_token},
    device_map="auto",
)

#Defines a method to get the filepaths of the XML files
def absoluteFilePaths(directory):
    files = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            files.append(os.path.abspath(os.path.join(dirpath, f)))
    return files

#Actually gets the filepaths for all the data
publicationFiles = absoluteFilePaths("data/publications")
authorFiles = absoluteFilePaths("data/staff")
grantFiles = absoluteFilePaths("data/grants")
journalFiles = absoluteFilePaths("data/journals")

#Uses the files to create documents for our index
docs = publicationReader(publicationFiles) + authorReader(authorFiles) + grantReader(grantFiles) + journalReader(journalFiles)

#Sets the config settings for the system
Settings.chunk_size = chunkSize
Settings.chunk_overlap = chunkOverlap
Settings.embed_model = embedModel
Settings.llm = llm

#Generates the index from the documents
index = VectorStoreIndex.from_documents(docs)

#Creates the query engine.
query_engine = index.as_query_engine(llm=llm, similarity_top_k= topK_Retrieved)

def generate_response(msg, history):
  response = str(query_engine.query(msg))
  return response

demo = gr.ChatInterface(fn=generate_response, title="ChatAcademy Demo")

# Add share=True to create a shareable link
demo.launch(share=True, debug=True)