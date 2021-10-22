# BDNN
Code related to the publication: Efficient and Robust Mixed-Integer Optimization Methods for Training Binarized Deep Neural Networks, J. Kurtz and B. Bah, 2021

Iterative Data-Splitting Method:

File: Networks.py

- contains class BDNN
- class contains all necessary network parameters and functions

File: Functions.py

- trainDNN(...): trains classical DNN via tensorflow
- trainBDNN(...): trains BDNN via iterative data-splitting method
- solve NetworkMIP(...): solves the MILP formulation for given partition of the data
- split_k_means(...): splits a set of data points via k-means into two subsets

Files: Main.py, Main_Robust.py

- starts experiments for data-splitting method and classical DNNs and evaluates output (for non-robust and robust case respectively)
- parameters for experiments can be adjusted at the beginning


Exact method and local search heuristic:

File: BDNN.py

- predictBDNN(...): receives test-set and the trained network parameters for networks with one hidden layer; returns predictions
- solveHeuristicBDNN(...): receives network architecture, dataset and labels and trains the network via local search procedure
- solveExactBDNN(...): receives network architecture, dataset and labels and trains the network via exact MILP formulation
