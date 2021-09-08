# Automatic detection of sleep micro-events with convolutional recurrent neural networks code
Uses python (3.8.10)

## Libraries required (Version)
Using pip package manager to install the libraries

**Preprocessing environment**

numpy (1.21.0),
scipy (1.7.0),
scikit-learn (0.24.2),
pyedflib (0.1.22),
librosa (0.8.1),
matplotlib (3.4.2)

**Network environment** (+ all other aspects)

tensorflow (2.5.0),
numpy (1.19.5),
matplotlib (3.4.2)

for tensorflow gpu, use configuration specified in https://www.tensorflow.org/install/source_windows

## Dataset
The MrOS sleep study dataset can be requested from:
https://sleepdata.org/datasets/mros

## Files
**variables.py**
This is where the global variables are set, including the the file locations, directories and network parameters.

**preprocessing.py**
Preprocessing of the training, validation and testing data

**filterbank_shape.py**
Computes a linear-frequency triangular filter bank used as a filtering technique for the spectrogram

**sliding_window.py**
Converts an input into a set of windows using numpy sliding window

**network.py**
The proposed network architecture implementation along with a datagenerator to prepare the input data

**sample_weights.py**
Data sampling technique used to balance event classes and active/non-active event states

**training.py**
The network is trained using the defined parameter settings

**hpsearch.py**
A hyper-parameter search used to find optimal parameter configuration

**testing.py**
The testing of the network model including the inference scheme


## Order of execution

1) variables.py - Set variables
2) preprocessing.py - Preprocess data
3) sample_weights.py - Compute sample weights
4) hpsearch.py - Find optimal parameters
5) training.py - Train model
6) testing.py - Test and evaluate model performance



