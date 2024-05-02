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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import XMLReader

from readers.authorReader import authorReader
from readers.grantReader import grantReader
from readers.journalReader import journalReader
from readers.publicationReader import publicationReader # type: ignore

from bs4 import BeautifulSoup
from datasets import load_dataset # type: ignore
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

nest_asyncio.apply()
oAI_token = sys.argv[1]
hfToken = sys.argv[2]
os.environ['OPENAI_API_KEY'] = oAI_token

#CONTROL PANEL - USE THIS TO CHANE THINGS FOR EXPERIMENTATION

#Control the language models in use
baseModel = "DreadN0ugh7/llama-7b-chat-academy"
evalModel = "meta-llama/Llama-2-13b-chat-hf"

#Control the embedding models in use
embedModel = "BAAI/bge-small-en-v1.5"

#Control whether to use custom readers or the default readers
customReaders = True

#Control the chunk Size and Overlap (Due to metadata, using less than 900 is not possible.)
chunkSize = 1024
chunkOverlap = 50

#Control the number of documents retrieved from the index
topK_Retrieved = 10

#Control which dataset is loaded and where the results are saved.
evalDataSetName = "DreadN0ugh7/ChatAcademyEvalDataset"
saveFileName = "results13btest"

#END OF THE CONTROL PANEL

#Loads the RAG eval dataset
evalDataset = load_dataset(evalDataSetName)

#Converts the dataset into a dictionary of questions and answers.
evalDictionary = {"questions":[], "answers": []}
evalDictionary["questions"] = evalDataset["query"]
evalDictionary["answers"] = evalDataset["answer"]

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

evalLLM = HuggingFaceLLM(
    model_name= evalModel,
    tokenizer_name= evalModel,
    query_wrapper_prompt=PromptTemplate("<s> [INST] {query_str} [/INST] "),
    context_window=3900,
    model_kwargs={"token": hfToken, "quantization_config": quantization_config},
    tokenizer_kwargs={"token": hfToken},
    device_map="auto"
)

#DO NOT CHANGE THIS LINE UNDER ANY CIRCUMNSTANCE
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

    print(publicationReader(publicationFiles) == None)
    print(authorReader(authorFiles)== None)
    print(journalReader(journalFiles)== None)
    print(grantReader(grantFiles)== None)

    docs = publicationReader(publicationFiles) + authorReader(authorFiles) + grantReader(grantFiles) + journalReader(journalFiles)
    readerConfig = "Custom readers are in use.\n"
else:
    evalDataset = LabelledRagDataset.from_json("Datasets/NoJournalsEvaluationSet.json")
    evalDictionary["questions"] = [example.dict()["query"] for example in evalDataset.examples]
    evalDictionary["answers"] = [example.dict()["reference_answer"] for example in evalDataset.examples]
    sampleIndex =random.sample(range(len(evalDataset.examples)), sampleSize)

    evalDictionary["questions"] = [evalDictionary["questions"][i] for i in sampleIndex]
    evalDictionary["answers"] = [evalDictionary["answers"][i] for i in sampleIndex]
    allFiles = publicationFiles + authorFiles + grantFiles
    loader = XMLReader()
    for file in allFiles:
        docs=loader.load_data(file=Path(file))
    readerConfig = "The standard xml reader is in use.\n"

#Sets the config settings for the system
Settings.chunk_size = chunkSize
Settings.chunk_overlap = chunkOverlap
Settings.embed_model = HuggingFaceEmbedding(model_name = embedModel)
Settings.llm = llm

#Generates the index from the documents
index = VectorStoreIndex.from_documents(docs)

#Assemble query engine
query_engine = index.as_query_engine(llm = llm, similarity_top_k= topK_Retrieved,)

#Creates a string with the config settings for the results file.
fileOutput ="*** Current Config ***\n"
fileOutput = fileOutput + f"The llm is: {baseModel}\n"
fileOutput = fileOutput + f"The evaluation llm is: {evalModel}\n"
fileOutput = fileOutput + readerConfig
fileOutput = fileOutput + f"The number of documents retrieved per query are: {topK_Retrieved}\n"
fileOutput = fileOutput + f"The chunk size is: {chunkSize}\n"
fileOutput = fileOutput + f"The chunk overlap is: {chunkOverlap}\n"
fileOutput = fileOutput + f"The embedding model is: {embedModel}\n\n"
fileOutput = fileOutput + f"There are {len(evalDataset.examples)} examples in the dataset.\n"
fileOutput = fileOutput + f"There are 500 examples being used from this to evaluate the system.\n\n"
fileOutput = fileOutput + f"*** LlamaIndex Evaluators ***\n\n"

#Creates the evaluators and evaluates the dataset
faithfulnessEval = FaithfulnessEvaluator(llm=evalLLM)
relevancyEval = RelevancyEvaluator(llm=evalLLM)
correctnessEval = CorrectnessEvaluator(llm=gptLLM)

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
        score = 0
        results = eval_results[key]
        if key == "correctness":
            correct = 0.0
            counter = 0
            for result in results:
                try:
                    correct += result.score
                    counter+=1
                except TypeError:
                    print("no score")
            score = correct / counter
        else:
            correct = 0
            for result in results:
                if result.score != None:
                    counter +=1
                    if result.passing:
                        correct += 1
            score = correct / len(results)
        resultsStr = resultsStr + f"The {key} score is {score}.\n"
    return resultsStr

#Calculates the bert score
refs =[x.response for x in evalResults["faithfulness"]]
P, R, F1 = score(evalDictionary["answers"], refs, lang='en', verbose=True)

#Prints the output to the results file
file1 = open(os.path.join("Results", saveFileName +".txt"), "w")
fileOutput = fileOutput + get_eval_results(keys, evalResults)
fileOutput = fileOutput + f"*** Bertscore Metrics ***\n\n"
fileOutput = fileOutput + f"System level F1 score is {F1.mean():.3f}.\n"
fileOutput = fileOutput + f"System level recall score is {R.mean():.3f}.\n"
fileOutput = fileOutput + f"System level precision score is {P.mean():.3f}.\n"
file1.write(str(fileOutput))
file1.close()