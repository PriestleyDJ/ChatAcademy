# -*- coding: utf-8 -*-

from datasets import load_dataset # type: ignore
from pathlib import Path
import torch # type: ignore
import random # type: ignore
import os
import sys
import nest_asyncio # type: ignore
import numpy as np # type: ignore
import gradio as gr

nest_asyncio.apply()
oAI_token = sys.argv[1]
hfToken = sys.argv[2]
os.environ['OPENAI_API_KEY'] = oAI_token
# os.environ['TRANSFORMERS_CACHE'] = '/mnt/parscratch/users/aca20djp/models'
# os.environ['HF_HOME'] = '/mnt/parscratch/users/aca20djp/models'

from bert_score import score # type: ignore
#from accelerate import Accelerator # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging # type: ignore
#from deepeval import evaluate # type: ignore
#from deepeval.models.base_model import DeepEvalBaseLLM # type: ignore
#from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric # type: ignore
#from deepeval.metrics.ragas import RAGASAnswerRelevancyMetric, RAGASFaithfulnessMetric # type: ignore
#from deepeval.test_case import LLMTestCase # type: ignore
from peft import AutoPeftModelForCausalLM, PeftModel # type: ignore
from llama_index.core import VectorStoreIndex, PromptTemplate, Settings, SimpleDirectoryReader # type: ignore
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, CorrectnessEvaluator, SemanticSimilarityEvaluator # type: ignore
#from llama_index.core.llama_dataset import LabelledRagDataset # type: ignore
from llama_index.llms.huggingface import HuggingFaceLLM # type: ignore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # type: ignore
from llama_index.llms.openai import OpenAI # type: ignore
from llama_index.readers.file import XMLReader # type: ignore

from readers.authorReader import authorReader # type: ignore
from readers.grantReader import grantReader # type: ignore
from readers.journalReader import journalReader # type: ignore
from readers.publicationReader import publicationReader # type: ignore

#from LlamaEvaluator import LlamaEvaluator # type: ignore

#CONTROL PANEL - USE THIS TO CHANE THINGS FOR EXPERIMENTATION

#Control the language models in use
baseModel = "meta-llama/Llama-2-13b-chat-hf"
evalModel = "meta-llama/Llama-2-13b-chat-hf"

#Control the embedding models in use
embedModel = "BAAI/bge-base-en-v1.5"

#Control whether to use custom readers or the default readers
customReaders = True

#Control the chunk Size and Overlap (Due to metadata, using less than 900 is not possible.)
chunkSize = 4096
chunkOverlap = 50

#Control the number of documents retrieved from the index
topK_Retrieved = 3

#Control which dataset is loaded and where the results are saved.
evalDataSetName = "DreadN0ugh7/ChatAcademyEvalDataset"
saveFileName = "resultsChunk4096"

#END OF THE CONTROL PANEL

#Actually builds the LLM
compute_dtype = getattr(torch, "float16")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

llm = HuggingFaceLLM(
    model_name = baseModel,
    tokenizer_name = baseModel,
    query_wrapper_prompt=PromptTemplate("<s> [INST] {query_str} [/INST] "),
    context_window=3900,
    model_kwargs={"token": hfToken, "quantization_config": quantization_config},
    tokenizer_kwargs={"token": hfToken, "quantization_config": quantization_config},
    device_map="auto"
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
"""
publicationFiles = absoluteFilePaths("data/publications")
authorFiles = absoluteFilePaths("com/staff")
grantFiles = absoluteFilePaths("com/grants")
journalFiles = absoluteFilePaths("com/journals")
relationFiles = absoluteFilePaths("com/relationships")
"""
#Uses the files to create documents for our index
docs = []

readerConfig = ""
if customReaders == True:
    #relReader = SimpleDirectoryReader(input_dir = "com/relationships/")
    #relDocs = relReader.load_data()
    docs = publicationReader(publicationFiles) + authorReader(authorFiles) + grantReader(grantFiles) + journalReader(journalFiles)
    readerConfig = "Custom readers are in use.\n"
else:
    allFiles = publicationFiles + authorFiles + grantFiles + journalFiles
    loader = XMLReader()
    for file in allFiles:
        try:
            docs=loader.load_data(file=Path(file))
        except:
            print("read error")
    readerConfig = "The standard xml reader is in use.\n"


#Sets the config settings for the system
Settings.chunk_size = chunkSize
Settings.chunk_overlap = chunkOverlap
Settings.embed_model = HuggingFaceEmbedding(model_name = embedModel, trust_remote_code=True)
Settings.llm = llm

#Generates the index from the documents
index = VectorStoreIndex.from_documents(docs)

#Assemble query engine
query_engine = index.as_query_engine(llm = llm, similarity_top_k= topK_Retrieved)

def generate_response(msg, history):
  response = str(query_engine.query(msg))
  return response

demo = gr.ChatInterface(fn=generate_response, title="ChatAcademy Demo")

# Add share=True to create a shareable link
demo.launch(share=True, debug=True)