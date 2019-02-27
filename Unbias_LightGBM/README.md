Unbiased_LambdaMart An implementation of Unbiased LambdaMart based on LightGBM (https://github.com/Microsoft/LightGBM). Note that LightGBM contains a wide variety of applications using gradient boosting decision tree algorithms. Our modification is mainly on the src/objective/rank_objective.hpp, which is the LambdaMart Ranking objective file.

There are mainly two hyper-parameters that can be tuned, which is all in the src/objective/rank_objective.hpp files. They are:

```
  double _eta               : this denotes how much regularization is posed to the position bias.
  size_t _position_bins     : this denotes the maximum positions taken into account.
```

Note that after the modification, each time for running Unbiased LambdaMart, one should prepare the following two files:
```
x.query: which contains the session length of each query
x.rank:  which contains the position for each item in a session
```
For details, please refer to evaluation/scripts/generate_data.py
