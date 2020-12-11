---
title: "Human Activity Recognition - A Machine Learning Approach"
author: "Marko Intihar"
date: "12/11/2020"
output:
  html_document:
    keep_md: true
---





## Introduction


## Data preparation

In this project data was provided by group of researchers who made provided publication titled **Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements** (reference here [url](http://groupware.les.inf.puc-rio.br/work.jsf?p1=10335)).


We have obtained two separate data files (stored in *.csv* format):

* **pmltraining.csv**
* **pmltesting.csv**


The case is we have to be fair and we will behave that we are not able to see **pmltesting.csv** file, which corresponds to new measurements we receive, and where we must predict the classes. Therefore given file will only be used for final model predictions.


* **train** data set (75% of measurements)
* **test** data set (25% of measurements)

Splitting will be done based on our outcome variable (**classe**) and doing a random shuffling of rows and then splitting into two parts (based on selected percentage). In the model training procedure we will use so called **k-fold cross-validation** technique om **train** data set, meaning our train data set will be randomly splitted into *k*-different folds and we will use *k-1* folds to train our model and check model performance on the fold left out We will repeat *k* iterations of training, in each iteration leaving out one new fold, therefore each fold will be once left out. By doing so we will get better model estimates and try to reduce the over-fitting of the model parameters. For data exploration we will also use only **train** data set. The **test** data set will be used for model benchmark, in order to select the top performing models.



### Data import


```r
# Check if data folder exists
if(!dir.exists("data")){
  dir.create("data")
}

# download csv files train/test
if(!file.exists("./data/pmltraining.csv")){
  download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
                destfile = "./data/pmltraining.csv")
}
if(!file.exists("./data/pmltesting.csv")){
  download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
                destfile = "./data/pmltesting.csv")
}


# import csv
pml_training <- read.csv(file = "./data/pmltraining.csv", header = T, 
                         sep = ",", quote = '"', row.names = 1) %>% 
  clean_names()
pml_testing <- read.csv(file = "./data/pmltesting.csv", header = T, 
                         sep = ",", quote = '"', row.names = 1) %>% 
  clean_names()
```



### Data split

Now let's do the splitting of **pml_training** data source using caret package.


```r
train_rows <- createDataPartition(y = pml_training$classe, p = .75, list = F) # do the splitting 

train <- pml_training[train_rows, ] # train data set
test  <- pml_training[-train_rows, ] # test data set
```




### Data wrangle




## Exploratory Data Analysis (EDA)



## Modeling


## Results
