# Language Model

## Requirements

Python 3.5+ and PyTorch 1.0+
Using CPU alone is enough.

## Data

The dataset for this paper is the text8 collection. This is a dataset taken from the first 100M characters of Wikipedia. Only 27 character types are present (lowercase characters and spaces); special characters are replaced by a single space and numbers are spelled out as individual digits.

## Task 1

`train-vowel-examples.txt` and `train-consonant-examples.txt` each contain 5000 strings of length 20, and `dev-vowel-examples.txt` and `dev-consonant-examples.txt` each contain 500. The task, which is a simplified version of the language modeling task, is to predict whether the first letter following each string is a vowel or a consonant by implementing a recurrent neural network(RNN).

After cloning this repository, `cd` into it and run

`python3 lm_classifier.py --model RNN`

This loads the data for this part, learns an RNN classifier on the data, and evaluates it.

## Task 2

This task involves building a complete RNN language model. For this part, we use the first 100,000 characters of text8 as the training set. The development set is 500 characters taken from elsewhere in the collection.

Run the following at the top-level of this directory:

`python3 lm.py --model RNN`

This will train the model using the data, make predictions and evaluate them.