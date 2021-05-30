## Linear regression
### Task
more details [here](https://github.com/gerus66/linear_regression/blob/master/readme/lin_reg.pdf)

[wiki](https://en.wikipedia.org/wiki/Linear_regression)

### Run
for ex: `./training.py && ./prediction.py -v 100000 -d data/data.csv`

### Utils
 `./training.py  [-h] [-d datafile] [-i iterations] [-lr learning rate] [-s storage]` 
 
 train linear model on _datafile_, save coefficients to _storage_
 
 ![Alt text](https://github.com/gerus66/linear_regression/blob/master/readme/training.png)
 
 and demonstrate decreasing error through _iterations_ (it depends on _learning rate_ though)
 
 ![Alt text](https://github.com/gerus66/linear_regression/blob/master/readme/rmse.png)
 
  `./prediction.py   [-h] -v value [-d datafile] [-s storage]` 
  
 predict result for _value_ (with presicion) based on coefficients from _storage_
 
  ![Alt text](https://github.com/gerus66/linear_regression/blob/master/readme/result.png)
  
 with given _datafile_ plot it all in context of data
 
 ![Alt text](https://github.com/gerus66/linear_regression/blob/master/readme/prediction.png)
 
