# Unbiased LambdaMart

Unbiased LambdaMart is a unbiased version of traditional LambdaMart, which can jointly estimate the biases at click positions and the biases at unclick positions, and learn an unbiased ranker using a pairwise loss function. 

The repository contains two parts, firstly an implementation of Unbiased LambdaMart based on LightGBM. Secondly a simulated click dataset with its generation scripts for evalution.

You can see the WWW 2019 (know as The Web Conference) paper “**Unbiased LambdaMART: An Unbiased PairwiseLearning-to-Rank Algorithm**” for more details.

## Overview

- Unbiased_LambdaMart
  An implementation of Unbiased LambdaMart based on LightGBM (https://github.com/Microsoft/LightGBM). Note that LightGBM contains a wide variety of applications using gradient boosting decision tree algorithms. Our modification is mainly on the *src/objective/rank_objective.hpp*, which is the LambdaMart Ranking objective file.
- evaluation/ 
  contains the synthetic click dataset generated using click models. This part of code is mainly forked from https://github.com/QingyaoAi/Unbiased-Learning-to-Rank-with-Unbiased-Propensity-Estimation. We also add the configs file to run our Unbiased LambdaMart on this synthetic dataset.

### Citation

Please consider citing the following paper when using our code for your application.

```
@inproceedings{chenshen2019,
  title={Unbiased LambdaMART: An Unbiased PairwiseLearning-to-Rank Algorithm},
  author={Ziniu Hu, Yang Wang, Qu Peng, Hang Li},
  booktitle={Proceedings of the 2019 World Wide Web Conference},
  year={2019}
}
```

 

