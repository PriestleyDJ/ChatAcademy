# -*- coding: utf-8 -*-

from bert_score import score
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging

from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex, PromptTemplate, Settings, Document, Response, get_response_synthesizer
from llama_index.core.evaluation import FaithfulnessEvaluator,RelevancyEvaluator,CorrectnessEvaluator, BatchEvalRunner
from llama_index.core.llama_dataset import LabelledRagDataset, CreatedBy, CreatedByType, LabelledRagDataExample
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import XMLReader

from readers.authorReader import authorReader
from readers.grantReader import grantReader
from readers.journalReader import journalReader
from readers.publicationReader import publicationReader

from bs4 import BeautifulSoup
from pathlib import Path
import torch
import calendar
import spacy
import asyncio
import random
import bert_score
import os
import sys
import nest_asyncio
import numpy as np
import gradio as gr

nest_asyncio.apply()
#oAI_token = sys.argv[1]
hfToken = sys.argv[2]
#os.environ['OPENAI_API_KEY'] = oAI_token

#CONTROL PANEL - USE THIS TO CHANE THINGS FOR EXPERIMENTATION

#Control the language models in use
baseModel = "meta-llama/Llama-2-13b-chat-hf"
evalModel = "meta-llama/Llama-2-13b-chat-hf"

#Control the embedding models in use
#embedModel = "local:BAAI/bge-small-en-v1.5"

#Control whether to use custom readers or the default readers
customReaders = True

#Control the chunk Size and Overlap (Due to metadata, using less than 700 is not possible.)
chunkSize = 804
chunkOverlap = 50

#Control the number of documents retrieved from the index
topK_Retrieved = 10

#Control which dataset is loaded and where the results are saved.
dataSetFileName = "DefaultEvaluationSet"
saveFileName = "results13bTop10"

#Control how many questions are used to test on the LLM (Using more will take longer)
sampleSize = 200

#END OF THE CONTROL PANEL

#Actually builds the LLM
compute_dtype = getattr(torch, "float16")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

llm = HuggingFaceLLM(
    model_name= baseModel,
    tokenizer_name= baseModel,
    query_wrapper_prompt=PromptTemplate("<s> [INST] {query_str} [/INST] "),
    context_window=3900,
    model_kwargs={"token": hfToken, "quantization_config": quantization_config},
    tokenizer_kwargs={"token": hfToken},
    device_map="auto"
)


#DO NOT CHANGE THIS LINE UNDER ANY CIRCUMNSTANCE, UNLESS YOU WANT TO COST ME UNNECCESARY MONEY.
gptLLM  = OpenAI("gpt-3.5-turbo-0125")
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
docs = []

readerConfig = ""
if customReaders == True:
    docs = publicationReader(publicationFiles) + authorReader(authorFiles) + grantReader(grantFiles) + journalReader(journalFiles)
    readerConfig = "Custom readers are in use.\n"

#Sets the config settings for the system
Settings.chunk_size = chunkSize
Settings.chunk_overlap = chunkOverlap
#Settings.embed_model = embedModel

#Generates the index from the documents
index = VectorStoreIndex.from_documents(docs)

#Assemble query engine
query_engine = index.as_query_engine()

def generate_response(msg, history):
  response = str(query_engine.query(msg))
  return response

demo = gr.ChatInterface(fn=generate_response, title="ChatAcademy Demo")

# Add share=True to create a shareable link
demo.launch(share=True)