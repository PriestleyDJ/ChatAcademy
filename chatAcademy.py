# -*- coding: utf-8 -*-

from datasets import load_dataset # type: ignore
from pathlib import Path
import torch # type: ignore
import random # type: ignore
import os
import sys
import nest_asyncio # type: ignore
import numpy as np # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging # type: ignore


nest_asyncio.apply()
oAI_token = sys.argv[1]
hfToken = sys.argv[2]

# Define the directory paths
transformers_cache_dir = '/mnt/parscratch/users/acb19es/models'
hf_home_dir = '/mnt/parscratch/users/acb19es/models'

# Create the directories if they do not exist
os.makedirs(transformers_cache_dir, exist_ok=True)
os.makedirs(hf_home_dir, exist_ok=True)

# Set the environment variables
os.environ['TRANSFORMERS_CACHE'] = transformers_cache_dir
os.environ['HF_HOME'] = hf_home_dir
os.environ['OPENAI_API_KEY'] = oAI_token

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

#CONTROL PANEL - USE THIS TO CHANGE THINGS FOR EXPERIMENTATION

#Control the language models in use
baseModel = "meta-llama/Llama-2-13b-chat-hf"
evalModel = "meta-llama/Llama-2-13b-chat-hf"

#Control the embedding models in use
embedModel = "BAAI/bge-small-en-v1.5"

#Control whether to use custom readers or the default readers
customReaders = False

#Control the chunk Size and Overlap (Due to metadata, using less than 900 is not possible.)
chunkSize = 4096
chunkOverlap = 50

#Control the number of documents retrieved from the index
topK_Retrieved = 3

#Control which dataset is loaded and where the results are saved.
evalDataSetName = "DreadN0ugh7/ChatAcademyEvalDataset"
saveFileName = "resultsChunk4096"

#END OF THE CONTROL PANEL

#Loads the RAG eval dataset
evalDataset = load_dataset(evalDataSetName, split = "train")

#Converts the dataset into a dictionary of questions and answers.
evalDictionary = {"questions":[], "answers": []}
evalDictionary["questions"] = evalDataset["query"]
evalDictionary["answers"] = evalDataset["answer"]

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

evalLLM = HuggingFaceLLM(
    model_name = evalModel,
    tokenizer_name = evalModel,
    query_wrapper_prompt=PromptTemplate("<s> [INST] {query_str} [/INST] "),
    context_window=3900,
    model_kwargs={"token": hfToken, "quantization_config": quantization_config},
    tokenizer_kwargs={"token": hfToken, "quantization_config": quantization_config},
    device_map="auto"
)

#DO NOT CHANGE THIS LINE UNDER ANY CIRCUMSTANCE
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

#Creates a string with the config settings for the results file.
fileOutput ="*** Current Config ***\n"
fileOutput = fileOutput + f"The llm is: {baseModel}\n"
fileOutput = fileOutput + f"The evaluation llm is: {evalModel}\n"
fileOutput = fileOutput + readerConfig
fileOutput = fileOutput + f"The number of documents retrieved per query are: {topK_Retrieved}\n"
fileOutput = fileOutput + f"The chunk size is: {chunkSize}\n"
fileOutput = fileOutput + f"The chunk overlap is: {chunkOverlap}\n"
fileOutput = fileOutput + f"The embedding model is: {embedModel}\n\n"
fileOutput = fileOutput + f"There are 500 examples being used to evaluate the system.\n\n"
fileOutput = fileOutput + f"*** LlamaIndex Evaluators ***\n\n"

#Creates the evaluators and evaluates the dataset
correctnessEval = CorrectnessEvaluator(llm=gptLLM)
faithfulnessEval = FaithfulnessEvaluator(llm=evalLLM)
relevancyEval = RelevancyEvaluator(llm=evalLLM)
semSimilarEval = SemanticSimilarityEvaluator()

correctTotal = faithfulTotal = relevantTotal = semSimilarityTotal = 0.0
correctCounter = faithfulCounter = relevantCounter= semSimilarityCounter = 0

responses = []

# Open file to write results
with open(os.path.join("Results", saveFileName + ".txt"), "w") as file1:
    # Write initial configuration details
    file1.write(fileOutput)

    for i in range(30):  # Limiting to the first 30 Q&A pairs
        query = evalDictionary["questions"][i]
        expectedOutput = evalDictionary["answers"][i]

        responseObject = query_engine.query(query)

        retrievalContext = [node.get_content() for node in responseObject.source_nodes]
        response = responseObject.response

        responses.append(response)

        # Write query and generated response
        file1.write(f"Query {i + 1}: {query}\n")
        file1.write(f"Generated Response {i + 1}: {response}\n")
        file1.write(f"Expected Output {i + 1}: {expectedOutput}\n\n")

        try:
            correctnessResult = correctnessEval.evaluate_response(
                query=query,
                response=responseObject,
                reference=expectedOutput,
            )
        except (TypeError, ValueError) as e:
            file1.write(f"Error in correctness evaluation: {e}\n")
        else:
            correctTotal += correctnessResult.score
            correctCounter += 1
            file1.write(f"Correctness Score: {correctnessResult.score}\n")

        relevancyResult = relevancyEval.evaluate_response(
            query=query,
            response=responseObject,
        )
        try:
            relevantTotal += relevancyResult.score
            relevantCounter+=1
            file1.write(f"Relevancy Score: {relevancyResult.score}\n")
        except TypeError:
            file1.write("No relevancy score\n")

        faithfulResult = faithfulnessEval.evaluate_response(
            query=query,
            response=responseObject,
        )
        try:
            faithfulTotal += faithfulResult.score
            faithfulCounter+=1
            file1.write(f"Faithfulness Score: {faithfulResult.score}\n")
        except TypeError:
            file1.write("No faithfulness score\n")

        semSimilarityResult = semSimilarEval.evaluate_response(
            reference=expectedOutput,
            response=responseObject
        )
        try:
            semSimilarityTotal += semSimilarityResult.score
            semSimilarityCounter+=1
            file1.write(f"Semantic Similarity Score: {semSimilarityResult.score}\n")
        except TypeError:
            file1.write("No semantic similarity score\n")

    # Calculate the final scores
    correctScore = correctTotal / correctCounter
    faithfulScore = faithfulTotal / faithfulCounter
    relevantScore = relevantTotal / relevantCounter
    semSimilarityScore = semSimilarityTotal / semSimilarityCounter

    # Calculates the BERT score
    P, R, F1 = score(evalDictionary["answers"], responses, lang='en', verbose=True)

    # Write final scores to the results file
    fileOutput = fileOutput + f"The correctness score is {correctScore}.\n"
    fileOutput = fileOutput + f"The faithfulness score is {faithfulScore}.\n"
    fileOutput = fileOutput + f"The relevancy score is {relevantScore}.\n"
    fileOutput = fileOutput + f"The semantic similarity score is {semSimilarityScore}.\n"
    fileOutput = fileOutput + f"*** Bertscore Metrics ***\n\n"
    fileOutput = fileOutput + f"System level F1 score is {F1.mean():.3f}.\n"
    fileOutput = fileOutput + f"System level recall score is {R.mean():.3f}.\n"
    fileOutput = fileOutput + f"System level precision score is {P.mean():.3f}.\n"
    file1.write(str(fileOutput))
    file1.close()
