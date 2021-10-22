# Yelp-Recruiting-Competition
Data Mining G07 Final Project (Kaggle competition link: https://www.kaggle.com/c/yelp-recruiting)
### Requirements
1. numpy == 1.14.3
2. pandas == 0.23.4
3. sklearn == 0.20.0
4. keras == 2.1.6
5. tensorflow == 1.9.0
### Dataset
Please first download the dataset from Kaggle to the directory 'data/'. (https://www.kaggle.com/c/yelp-recruiting/data)
### Directories and files
```
$ tree
.
├── data (need to download first)
│   ├── yelp_test_set
│   │   ├── yelp_test_set_business.json
│   │   ├── yelp_test_set_checkin.json
│   │   ├── yelp_test_set_review.json
│   │   └── yelp_test_set_user.json
│   ├── yelp_training_set
│   │   ├── yelp_training_set_business.json
│   │   ├── yelp_training_set_checkin.json
│   │   ├── yelp_training_set_review.json
│   │   └── yelp_training_set_user.json
│   └── sample_submission.csv
├── feature
│   ├── test_X.pkl
│   ├── X.pkl
│   └── Y.pkl
├── model
│   └── DNN.h5
├── submit
│   ├── DNN.csv
│   ├── ensemble.csv
│   ├── LR.csv
│   ├── RFR.csv
│   └── SVR.csv
├── data_preprocessing.py
├── DNN.py
├── ensemble.py
├── LR.py
├── README.md
├── Report.pdf
├── RFR.py
└── SVR.py

6 directories, 26 files
```
### Data Preprocessing
> After downloading the dataset, run:
```Shell
$ python3 data_preprocessing.py
```
Check the saved file 'test_X.pkl', 'X.pkl', 'Y.pkl' under the directory 'feature/'.
### Prediction
> Using Linear Regression (LR), run:
```Shell
$ python3 LR.py
```
Check the saved file 'LR.csv' under the directory 'submit/'.
> Using Deep Neural Network (DNN), run:
```Shell
$ python3 DNN.py
```
Check the saved file 'DNN.csv' under the directory 'submit/'.
> Using Support Vector Regressor (SVR), run:
```Shell
$ python3 SVR.py
```
Check the saved file 'SVR.csv' under the directory 'submit/'.
> Using Random Forest Regressor (RFR), run:
```Shell
$ python3 RFR.py
```
Check the saved file 'RFR.csv' under the directory 'submit/'.
> Before using ensemble, please first check that 'SVR.csv', 'DNN.csv' and 'RFR.csv' are under the directory 'submit/'.
> Then run:
```Shell
$ python3 ensemble.py
```
Check the saved file 'ensemble.csv' under the directory 'submit/'.
### Evaluation
|Method|Private Score|Public Score|Ranking
|---|---|---|---
|LR|0.50826|0.50889|94/350
|DNN|0.48545|0.48405|62/350
|SVR|0.50575|0.50356|87/350
|RFR|0.52554|0.52579|120/350
|ensemble|0.48340|0.48241|59/350
> Evaluation Metric: Root Mean Squared Logarithmic Error ("RMSLE") 
