# -*- coding: utf-8 -*-

from readers import authorReader, grantReader, journalReader,publicationReader

import os
oAI_token =  "sk-proj-3Pc0Pm2HchZuwhvtH8W2T3BlbkFJjyCiUsZJBFMj90QvCpCN"
os.environ['OPENAI_API_KEY'] =  oAI_token
import sys
import nest_asyncio
nest_asyncio.apply()

import numpy as np
import bert_score
from bert_score import score
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging
from llama_index.core import VectorStoreIndex, PromptTemplate, Settings, Document, Response
from llama_index.core.evaluation import FaithfulnessEvaluator,RelevancyEvaluator,CorrectnessEvaluator, BatchEvalRunner
from llama_index.core.llama_dataset import LabelledRagDataset, CreatedBy, CreatedByType, LabelledRagDataExample
from llama_index.llms.huggingface import HuggingFaceLLM
from bs4 import BeautifulSoup
import torch
import calendar
import spacy
import asyncio

hf_token = sys.argv[1]

evalDataset = LabelledRagDataset.from_json("rag_dataset.json")

qas_dict = {"questions":[], "answers": []}
qas_dict["questions"] = [example.dict()["query"] for example in evalDataset.examples]
qas_dict["answers"] = [example.dict()["reference_answer"] for example in evalDataset.examples]

#Sets the name of the llm we are going to use (Use llama index to find a different one if wanted)
baseModel = "meta-llama/Llama-2-7b-chat-hf"
evalModel = "meta-llama/Llama-2-7b-chat-hf"
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

print("loaded llama-3")

evalLLM = HuggingFaceLLM(
    model_name= evalModel,
    tokenizer_name= evalModel,
    query_wrapper_prompt=PromptTemplate("<s> [INST] {query_str} [/INST] "),
    context_window=3900,
    model_kwargs={"token": hf_token, "quantization_config": quantization_config},
    tokenizer_kwargs={"token": hf_token},
    device_map="auto",
)

print("loaded llama-2")

def absoluteFilePaths(directory):
    files = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            files.append(os.path.abspath(os.path.join(dirpath, f)))
    return files

publicationFiles = absoluteFilePaths("data/publications")
authorFiles = absoluteFilePaths("data/staff")
grantFiles = absoluteFilePaths("data/grants")
journalFiles = absoluteFilePaths("data/journals")

print(publicationFiles + authorFiles + grantFiles + journalFiles)

docs = publicationReader(publicationFiles) + authorReader(authorFiles) + grantReader(grantFiles) + journalReader(journalFiles)

Settings.chunk_size = 768
Settings.chunk_overlap = 50
Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

index = VectorStoreIndex.from_documents(docs)


fileOutput ="*** Current Config ***\n"
fileOutput = fileOutput + f"The llm is: {baseModel}\n"
fileOutput = fileOutput + f"The evaluation llm is: {evalModel}\n"
fileOutput = fileOutput + f"The chunk size is: {Settings.chunk_size}\n"
fileOutput = fileOutput + f"The chunk overlap is: {Settings.chunk_overlap}\n"
fileOutput = fileOutput + f"The embedding model is: {Settings.embed_model.model_name}\n"
fileOutput = fileOutput + f"There are {len(evalDataset.examples)} examples in the dataset.\n"

query_engine = index.as_query_engine(llm=llm)

faithfulnessEval = FaithfulnessEvaluator(llm=evalLLM)
relevancyEval = RelevancyEvaluator(llm=evalLLM)
correctnessEval = CorrectnessEvaluator(llm=evalLLM)

runner = BatchEvalRunner(
    {"faithfulness": faithfulnessEval, "relevancy": relevancyEval, "correctness": correctnessEval},
    workers=8,
)

keys =["faithfulness","relevancy","correctness"]

evalResults = runner.evaluate_queries(
    query_engine, queries= qas_dict["questions"][:3],  reference= qas_dict["answers"][:3]
)

def get_eval_results(keys, eval_results):
    resultsStr = str("")                                     
    for key in keys:
        results = eval_results[key]
        print(result.query)
        print(result.response)
        print(result.passing)
        correct = 0
        for result in results:
            if result.passing:
                correct += 1
        score = correct / len(results)
        resultsStr = resultsStr + f"{key} Score: {score} \n"
    return resultsStr

refs =[x.response for x in eval_results["faithfulness"]]
P, R, F1 = score(qas_dict["answers"][:5], refs, lang='en', verbose=True)
F1.mean()

print(get_eval_results(keys, evalResults))

file1 = open("results.txt", "w")
fileOutput = fileOutput + get_eval_results(keys, evalResults)
fileOutput = fileOutput + f"System level F1 score: {F1.mean():.3f}"
fileOutput = fileOutput + f"System level recall score: {R.mean():.3f}"
fileOutput = fileOutput + f"System level precision score: {P.mean():.3f}"
file1.write(str(fileOutput))
file1.close()