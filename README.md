#MHGA_CDM

Source code and data set for the paper *Can probabilistic methods with evolutionary optimization rival neural network methods for cognitive diagnosis?*.

The code is the implementation of MHGA_CDM model, and the data sets are the public data set Frcsub、Math1、Math2.

## Dependencies:
-python 3.8
-deap
-sklearn
-numpy
-pandas
-igraph
-multiprocessing
-tools

## Usage

The datasets are given in the 'dataSets' folder.

Run the model:

`python main.py`

## Data Set

Each data set only needs to use the data.csv and q.csv files, where the data.csv file is the matrix for the students to answer the exercises, and the q.csv file is the matrix of the knowledge points for the exercises.
