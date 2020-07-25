
## Introduction

The code of combiner to do **truth discovery** for the FedServing paper.

This repository includes:
1. `combiner_sgx`: contains the program to do truth discovery in SGX;
2. `outputs`: contains predictions output by participant models, stored as  `csv` files;
3. `outputs_combined`: contains predictions combined by the combiner;
3. `image`: contains scripts to build models of task MNIST and ImageNet;
3. `nlp`: contains scripts to build models of task 20News;
4. `utils`: contains tools to calculate accuracies of combined results.

 
## Tested environment and dependencies
1. Ubuntu 16.04
2. SGX SDK 2.5
3. SGX PSW
4. Pytorch 1.1.0
5. Tensorflow 1.12.2


## Truth discovery

To do truth discovery, enter `combiner_sgx` and run the following command:

```bash
./run.sh
```
The program will be compiled and executed. If executed correctly, the program will print logs as follows.
```
GEN  =>  App/Enclave_u.c
CC   <=  App/Enclave_u.c
CXX  <=  App/App.cpp
CXX  <=  App/sgx_utils/sgx_utils.cpp
LINK =>  app
GEN  =>  Enclave/Enclave_t.c
CC   <=  Enclave/Enclave_t.c
CXX  <=  Enclave/Enclave.cpp
LINK =>  enclave.so
<!-- Please refer to User's Guide for the explanation of each field -->
<EnclaveConfiguration>
    <ProdID>0</ProdID>
    <ISVSVN>0</ISVSVN>
    <StackMaxSize>0x400000</StackMaxSize>
    <HeapMaxSize>0x700000</HeapMaxSize>
    <TCSNum>10</TCSNum>
    <TCSPolicy>1</TCSPolicy>
    <DisableDebug>0</DisableDebug>
    <MiscSelect>0</MiscSelect>
    <MiscMask>0xFFFFFFFF</MiscMask>
</EnclaveConfiguration>
tcs_num 10, tcs_max_num 10, tcs_min_pool 1
The required memory is 49627136B.
Succeed.
SIGN =>  enclave.signed.so
Initialization takes 0.3644s
Task:MNIST
read data from: ../outputs/MNIST/query_4000/type_label/KNN.csv
read data from: ../outputs/MNIST/query_4000/type_label/SVM.csv
read data from: ../outputs/MNIST/query_4000/type_label/CNN.csv
read data from: ../outputs/MNIST/query_4000/type_label/RNN.csv
read data from: ../outputs/MNIST/query_4000/type_label/LR.csv
read data from: ../outputs/MNIST/query_4000/type_label/MLP.csv
Data reading takes 0.020489s
Combining results inside enclave...
Results combined! Exiting the enclave
Combination takes 0.010545s
Results are saved to ./../outputs_combined/combination_results_new/MNIST/noise_0/MNIST_4000_label_iter_16.csv
```
After the truth discovery is done, results will be saved in `outputs_combined`，saved as `csv` file.


Configuration can be found in `combiner_sgx/aggregator.h`. Please refer to the file to see the configurable fields.


## Get truth discovery accuracies

To get the truth discovery accuracy, a ground truth file should be prepared in advance.

We provide the ground truth files as well as the combined results used in our experiments in this repo. To obtain the accuracy, enter `utils` and run the following commands.
``` bash
python acc_utils.py
```
If the program executes correctly, accuracies will be printed as follows.
```
============= MNIST =============
> noise level is : 0
> MNIST_1000_label_iter_20.csv          label - acc:    0.975000
> MNIST_4000_label_iter_20.csv          label - acc:    0.978250
> MNIST_7000_label_iter_20.csv          label - acc:    0.978000
> MNIST_1000_rank_iter_20.csv           rank - acc:     0.972000
> MNIST_4000_rank_iter_20.csv           rank - acc:     0.973500
> MNIST_7000_rank_iter_20.csv           rank - acc:     0.973286
> MNIST_1000_probs_iter_20.csv          probs - acc:    0.975000
> MNIST_4000_probs_iter_20.csv          probs - acc:    0.979500
> MNIST_7000_probs_iter_20.csv          probs - acc:    0.980429
```




## Turth discovery results
### Results on MNIST (iteration = 20)
| Noise         |    0    |          |          |            | 3        |         |          |            |     5    |          |          |            |
|---------------|:-------:|:--------:|:--------:|:----------:|----------|---------|----------|------------|:--------:|:--------:|:--------:|:----------:|
| Sample number | label   | rank     | probs    | probs (KL) | label    | rank    | probs    | probs (KL) | label    | rank     | probs    | probs (KL) |
| 1000          | 0.975   | 0.972    | 0.975    | 0.875      | 0.936    | 0.742   | 0.951    | 0.324      | 0.272    | 0.253    | 0.467    | 0.124      |
| 4000          | 0.97825 | 0.9735   | 0.9795   | 0.878      | 0.93575  | 0.74725 | 0.95325  | 0.1405     | 0.33475  | 0.2155   | 0.46325  | 0.09525    |
| 7000          | 0.978   | 0.973286 | 0.980429 | 0.878286   | 0.936857 | 0.754   | 0.953714 | 0.097143   | 0.274429 | 0.232286 | 0.468714 | 0.101286   |


### Results on 20News (iteration = 20)

| Noise         | 0        |          |          |            | 5      |        |          |            | 9        |          |          |            |
|---------------|----------|----------|----------|------------|--------|--------|----------|------------|----------|----------|----------|------------|
| Sample number | label    | rank     | probs    | probs (KL) | label  | rank   | probs    | probs (KL) | label    | rank     | probs    | probs (KL) |
| 1000          | 0.846    | 0.819    | 0.84     | 0.844      | 0.825  | 0.614  | 0.828    | 0.824      | 0.173    | 0.115    | 0.284    | 0.321      |
| 3000          | 0.850333 | 0.826333 | 0.849333 | 0.847333   | 0.83   | 0.61   | 0.826333 | 0.821      | 0.147333 | 0.111333 | 0.276667 | 0.32       |
| 5000          | 0.8616   | 0.836    | 0.8618   | 0.8596     | 0.8412 | 0.6228 | 0.839    | 0.8318     | 0.135    | 0.1142   | 0.2766   | 0.3232     |


### Results on ImageNet (iteration = 20)
| Noise         | 0        |          |          |            | 7        |          |          |            | 14       |        |          |            |
|---------------|----------|----------|----------|------------|----------|----------|----------|------------|----------|--------|----------|------------|
| Sample number | label    | rank     | probs    | probs (KL) | label    | rank     | probs    | probs (KL) | label    | rank   | probs    | probs (KL) |
| 5000          | 0.7774   | 0.752    | 0.7754   | 0.72       | 0.7566   | 0.0666   | 0.7504   | 0.718      | 0.5914   | 0.0026 | 0.279    | 0.0218     |
| 10000         | 0.7787   | 0.7536   | 0.7768   | 0.7348     | 0.7565   | 0.067    | 0.7517   | 0.7035     | 0.0095   | 0.0043 | 0.282    | 0.0232     |
| 16000         | 0.789687 | 0.763625 | 0.789188 | 0.713187   | 0.766188 | 0.070563 | 0.767813 | 0.712938   | 0.009687 | 0.0035 | 0.287687 | 0.024      |



