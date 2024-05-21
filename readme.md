# Project Decription
This is chatAcademy a darwin project developed to help with asksing questions about research at the University of sheffield. This github contains the code and data assocaited with out project.

# Files and Directories

To fill you in on some of the files available in this github we've placed some information here on what they are.

## COM XML Data and XML Data

These folders contain information from the MyPublications API that can be used to provide our chatbot with knowledge. The COM folder specifically has information just about the computer science department.
The /XML Data directory contains two Python files to extract XML files from the API. One gets all the staff members part of a group (e.g., the Computer science department), and the other gets all the associated XML files (e.g., publications, activities, and grants) associated with each staff member.  

## Notebooks

This folder contains jupyter notebooks that we used to achieve various tasks such as generating and uploading information to huggingface.

## Results

Contains results from evaluation on our model run under various configs.

## Readers

Contains python files that can be used to scrape information from our XML files for use in our chatbots.

## Datasets

Deprecated QA pairs that can be loaded as an LabelledRAGDataset. It is recommended to use the datasets available on huggingface instead.

## Python files

Thes can be used to run various versions of the chatbot. The chatAcademy.py file will just run automatic evaluation of the model and store the results. While the ModelTrainer can be used to fine tune a language model and uplaod the results to huggingface.

## Bash files

These files contain the necessary instructions for running chatacedemy on the university HPC.

## Requirements files

Can be run to install the necessary dependencies for the project.

# Installation and running

1 - ssh to Stanage  - to access the Stanage HPC, open a terminal window and use the following command, MAKE SURE YOU ARE CONNECTED TO THE VPN: 

ssh {username}@stanage.shef.ac.uk

2 - Clone the repository

git clone -b HPCDeployment git@github.com:PriestleyDJ/ChatAcademy.git

3 - Load anaconda

module load Anaconda3/2022.05

4 - Create a conda environment

conda create -n chatAcademy python=3.9
source activate chatAcademy

5 - Install the dependencies

pip install -r requirements.txt

6 - Login to huggingface

huggingface-cli login --token {huggingface_access_token}

7 - Run the Model

sbatch run_chatAcademy.sh #openAIToken #HuggingFaceToken
