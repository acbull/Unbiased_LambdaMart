Unbiased_LambdaMart An implementation of Unbiased LambdaMart based on LightGBM (https://github.com/Microsoft/LightGBM). Note that LightGBM contains a wide variety of applications using gradient boosting decision tree algorithms. Our modification is mainly on the src/objective/rank_objective.hpp, which is the LambdaMart Ranking objective file.

There are mainly two hyper-parameters that can be tuned, which you can set by config files. They are:

```
  eta = 0.5               : this denotes how much regularization is posed to the position bias.
  position_bins =12       : this denotes the maximum positions taken into account.
```

If you want to re-implement the result of our table, you can change the regularization term back to 0, by modifying 
```
eta = 0.5; 
```
in the config file to 
```
eta = 0; 
```

Another thing need to be noted is that in our Unbiased LambdaMart, each time for running Unbiased LambdaMart, one should prepare the following two files:
```
x.query: which contains the session length of each query
x.rank:  which contains the position for each item in a session
```
For details, please refer to evaluation/scripts/generate_data.py
