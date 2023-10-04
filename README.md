## Linear regression
### Task
_[full version in pdf](https://github.com/gerus66/linear_regression/blob/master/readme/lin_reg.pdf)_ \
_[wiki](https://en.wikipedia.org/wiki/Linear_regression) about linear regression_

### Run
for ex: `./training.py && ./prediction.py -v 100000 -d data/data.csv`

### Utils
 `./training.py  [-h] [-d datafile] [-i iterations] [-lr learning rate] [-s storage]`

This one trains linear model on `datafile`, saves coefficients to `storage` and demonstrates error decreasing through `iterations` (it also depends on `learning rate` though).
 
 ![Alt text](https://github.com/gerus66/linear_regression/blob/master/readme/training.png)
 
 ![Alt text](https://github.com/gerus66/linear_regression/blob/master/readme/rmse.png)
 
  `./prediction.py   [-h] -v value [-d datafile] [-s storage]` \
 
 This one predicts result for `value`, based on coefficients from `storage`, and with given `datafile` plots it all together with data.
 
  ![Alt text](https://github.com/gerus66/linear_regression/blob/master/readme/result.png)
 
 ![Alt text](https://github.com/gerus66/linear_regression/blob/master/readme/prediction.png)
 
