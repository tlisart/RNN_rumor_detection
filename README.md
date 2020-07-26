# Microblogging rumor detection using Neural Networks

Written in the context of the Machine-learning and Big Data processing course at VUB (ELEC-Y591)

RNN as three different architecture, using SimpleRNN, LSTM layering and GRU layering. 

### Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Python depedencies :
    Tensorflow
    Chinese


### Installing

Run DownloadData.py to download the data files from google drive
Run Preprocessing.py, this generates shuffled and preprocessed numpy array files from the txt files for 3 different k values
Load them as how is done in RNN_template.py


### Running the tests

Run Experiments.py to run experiments on different architectures, to change the parameters of the experiments change them in the run_experiment function call at the bottom of the file.

### Extra 

The preprocessing folder contains scripts to transform the Weibo data to tfidf values

### Authors

Jolan HUYVAERT, Théo LISART, Logan SIEBERT




