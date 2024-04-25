# -*- coding: utf-8 -*-
import os

import sys
import nest_asyncio

import numpy as np

from bert_score import score
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
import bert_score
nest_asyncio.apply()
oAI_token = sys.argv[1]
os.environ['OPENAI_API_KEY'] = oAI_token

#CONTROL PANEL - USE THIS TO CHANE THINGS FOR EXPERIMENTATION

#Control the language models in use
baseModel = "meta-llama/Llama-2-7b-chat-hf"
evalModel = "gpt-3.5-turbo"

#Control the embedding models in use
embedModel = "local:BAAI/bge-small-en-v1.5"

#Control the chunk Size and Overlap (Due to metadata, using less than 700 is not possible.)
chunkSize = 768
chunkOverlap = 50

#Control the number of documents retrieved from the index
topK_Retrieved = 2

#Control which dataset is loaded and where the results are saved.
dataSetFileName = "DefaultEvaluationSet"
saveFileName = "results7b"

#Control how many questions are used to test on the LLM (Using more will take longer)
sampleSize = 200

#END OF THE CONTROL PANEL

#Loads the RAG eval dataset
evalDataset = LabelledRagDataset.from_json("Datasets/" + dataSetFileName + ".json")

#Converts the dataset into a dictionary of questions and answers.
evalDictionary = {"questions":[], "answers": []}
evalDictionary["questions"] = [example.dict()["query"] for example in evalDataset.examples]
evalDictionary["answers"] = [example.dict()["reference_answer"] for example in evalDataset.examples]

#Creates an array of integers which represent the questions and answers that will be used.
sampleIndex =random.sample(range(len(evalDataset.examples)), sampleSize)

evalDictionary["questions"] = [evalDictionary["questions"][i] for i in sampleIndex]
evalDictionary["answers"] = [evalDictionary["answers"][i] for i in sampleIndex]

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

#DO NOT CHANGE THIS LINE UNDER ANY CIRCUMNSTANCE, UNLESS YOU WANT TO COST ME UNNECCESARY MONEY.
evalLLM  = OpenAI("gpt-3.5-turbo-0125")
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

#Generates the index from the documents
index = VectorStoreIndex.from_documents(docs)

#Creates a string with the config settings for the results file.
fileOutput ="*** Current Config ***\n"
fileOutput = fileOutput + f"The llm is: {baseModel}\n"
fileOutput = fileOutput + f"The evaluation llm is: {evalModel}\n"
fileOutput = fileOutput + f"The chunk size is: {chunkSize}\n"
fileOutput = fileOutput + f"The chunk overlap is: {chunkOverlap}\n"
fileOutput = fileOutput + f"The embedding model is: {embedModel}\n"
fileOutput = fileOutput + f"There are {len(evalDataset.examples)} examples in the dataset.\n"
fileOutput = fileOutput + f"There are {sampleSize} examples being used from this to evaluate the system.\n"

#Creates the query engine.
query_engine = index.as_query_engine(llm=llm)

#Creates the evaluators and evaluates the dataset
faithfulnessEval = FaithfulnessEvaluator(llm=evalLLM)
relevancyEval = RelevancyEvaluator(llm=evalLLM)
correctnessEval = CorrectnessEvaluator(llm=evalLLM)

runner = BatchEvalRunner(
    {"faithfulness": faithfulnessEval, "relevancy": relevancyEval, "correctness": correctnessEval},
    workers=8,
)

keys =["faithfulness","relevancy","correctness"]

evalResults = runner.evaluate_queries(
    query_engine, queries= evalDictionary["questions"], reference= evalDictionary["answers"]
)

#Collates the evaluations into a single value for the system
def get_eval_results(keys, eval_results):
    resultsStr = str("")                                     
    for key in keys:
        results = eval_results[key]
        print(results )
        correct = 0
        for result in results:
            if result.passing:
                correct += 1
        score = correct / len(results)
        resultsStr = resultsStr + f"{key} Score: {score} \n"
    return resultsStr

#Calculates the bert score
refs =[x.response for x in evalResults["faithfulness"]]
P, R, F1 = score(evalDictionary["answers"], refs, lang='en', verbose=True)

#Prints the output to the results file
file1 = open(os.path.join("Results", saveFileName +".txt"), "w")
fileOutput = fileOutput + get_eval_results(keys, evalResults)
fileOutput = fileOutput + f"System level F1 score: {F1.mean():.3f}"
fileOutput = fileOutput + f"System level recall score: {R.mean():.3f}"
fileOutput = fileOutput + f"System level precision score: {P.mean():.3f}"
file1.write(str(fileOutput))
file1.close()