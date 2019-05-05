# Unbiased LambdaMart

Unbiased LambdaMart is a unbiased version of traditional LambdaMart, which can jointly estimate the biases at click positions and the biases at unclick positions, and learn an unbiased ranker using a pairwise loss function. 

The repository contains two parts, firstly an implementation of Unbiased LambdaMart based on LightGBM. Secondly a simulated click dataset with its generation scripts for evalution.

You can see our WWW 2019 (know as The Web Conference) paper “**Unbiased LambdaMART: An Unbiased PairwiseLearning-to-Rank Algorithm**” for more details.

## Overview

- Unbiased_LambdaMart：

  An implementation of Unbiased LambdaMart based on LightGBM (https://github.com/Microsoft/LightGBM). Note that LightGBM contains a wide variety of applications using gradient boosting decision tree algorithms. Our modification is mainly on the `src/objective/rank_objective.hpp`, which is the LambdaMart Ranking objective file.
- evaluation：

  contains the synthetic click dataset generated using click models. This part of code is mainly forked from https://github.com/QingyaoAi/Unbiased-Learning-to-Rank-with-Unbiased-Propensity-Estimation. We also add the configs file to run our Unbiased LambdaMart on this synthetic dataset.

## Setup

First compile the Unbias_LightGBM (Original LightGBM with the implementation of Unbiased LambdaMart)

On Linux LightGBM can be built using CMake and gcc or Clang.

Install CMake.

Run the following commands:
```
cd Unbias_LightGBM/
mkdir build ; cd build
cmake .. 
make -j4
```
Note: glibc >= 2.14 is required.
After compilation, we will get a "lighgbm" executable file in the folder.

## Example

We modified the original example file to give an illustration. 

Compile, then run the following commands:
```
cd Unbias_LightGBM
cp ./lightgbm ./examples/lambdarank/
cd ./examples/lambdarank/
./lightgbm config="train.conf"
```

Despite the original XXX.train (provide feature) and XXX.train.query (provide which query a document belongs to), our modified lambdamart required a XXX.train.rank file to provide the position information to conduct debiasing. For later usage, remember to add this file.

## Evaluation

Firstly, download the ranked dataset by an initial SVM ranker from 
https://drive.google.com/file/d/1459mQDnj-0yPtYMIc1LAqLg7Q5VJUw-K/view?usp=sharing
And then put it into the evaluation directory. Also, one can generate this from scratch by their own, by refering to the procedure of https://github.com/QingyaoAi/Unbiased-Learning-to-Rank-with-Unbiased-Propensity-Estimation.

Then, generate the synthetic dataset from click models by:
```
cd evaluation
mkdir test_data
cd scripts
python generate_data.py ../click_model/user_browsing_model_0.1_1_4_1.json
```
Their are also other click model configurations in `evaluation/click_model/`, one can use any of them.

Finally, move the compiled `lighgbm` file into `evaluation/configs`, and then run:
```
./lightgbm config='train.conf'
./lightgbm config='test.conf'
```
In this way, the test results will be generated.

### Citation

Please consider citing the following paper when using our code for your application.

```
@inproceedings{unbias_lambdamart,
  title={Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm},
  author={Ziniu Hu, Yang Wang, Qu Peng, Hang Li},
  booktitle={Proceedings of the 2019 World Wide Web Conference},
  year={2019}
}
```

 

