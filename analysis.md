---
title: "Human Activity Recognition - A Machine Learning Approach"
author: "Marko Intihar"
date: "12/11/2020"
output:
  html_document:
    keep_md: true
---






## Introduction

In this analysis we are focusing on data gathered from accelerometers, which were attached on participants' belt, forearm, arm, and dumbbell, and were colecting data regarding exercises. In the data collection process 6 participants were participating. The main goal of the analysis is to build a machine learning algorithm that will be able to predict the manner in which participants did the exercise using the data collected on accelometers.


## Data 

In this project data was provided by group of researchers who written publication titled **Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements** (reference here [url](http://groupware.les.inf.puc-rio.br/work.jsf?p1=10335)).


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
set.seed(11235) # seed for split
train_rows <- createDataPartition(y = pml_training$classe, p = .75, list = F) # do the splitting 

train <- pml_training[train_rows, ] # train data set
test  <- pml_training[-train_rows, ] # test data set
```




### Data wrangle


Now we will check if there are any columns that have a lot of missing values NAs (potential imputation or removing columns):


```r
# calculate % of missing rows in each column 
NAs <- map(train, ~sum(is.na(.))) %>% 
  unlist() / nrow(train) 

# create a DF: var and % of missing values in that var
NAs <- data.frame(var = names(NAs),
                  missing = NAs) %>% 
  mutate(missing = round(missing * 100, digits = 1)) %>% 
  rename(`missing rows %` = missing) %>% 
  arrange(desc(`missing rows %`))
rownames(NAs) <- NULL

# extract names of variables to drop
drop.vars <- NAs %>% filter(`missing rows %` > 90) %>% pull(var) # list of columns to drop (more then 50 % NA)
```

There are 67 variables with almost all missing values. We will drop these variables, since there aren't any added value, if we put them into our modeling procedure. Lets drop columns from train and test data set:



```r
train <- train %>% select(-drop.vars) 
test  <- test %>% select(-drop.vars) 
```


Let see which columns are character type (potential conversion to numeric or factor, or to drop columns). On a first sight we think we must have most of the columns numeric. Some strange values can prevent numeric variables to becoming numeric, lets find out which values are preventing this to happen. Isolate first character vectors:


```r
# check which columns are characters (potential transformation to numeric or to drop columns)
charvars <- colnames(train)[train %>% lapply(class) == "character"]
charvars
```

```
##  [1] "user_name"               "cvtd_timestamp"         
##  [3] "new_window"              "kurtosis_roll_belt"     
##  [5] "kurtosis_picth_belt"     "kurtosis_yaw_belt"      
##  [7] "skewness_roll_belt"      "skewness_roll_belt_1"   
##  [9] "skewness_yaw_belt"       "max_yaw_belt"           
## [11] "min_yaw_belt"            "amplitude_yaw_belt"     
## [13] "kurtosis_roll_arm"       "kurtosis_picth_arm"     
## [15] "kurtosis_yaw_arm"        "skewness_roll_arm"      
## [17] "skewness_pitch_arm"      "skewness_yaw_arm"       
## [19] "kurtosis_roll_dumbbell"  "kurtosis_picth_dumbbell"
## [21] "kurtosis_yaw_dumbbell"   "skewness_roll_dumbbell" 
## [23] "skewness_pitch_dumbbell" "skewness_yaw_dumbbell"  
## [25] "max_yaw_dumbbell"        "min_yaw_dumbbell"       
## [27] "amplitude_yaw_dumbbell"  "kurtosis_roll_forearm"  
## [29] "kurtosis_picth_forearm"  "kurtosis_yaw_forearm"   
## [31] "skewness_roll_forearm"   "skewness_pitch_forearm" 
## [33] "skewness_yaw_forearm"    "max_yaw_forearm"        
## [35] "min_yaw_forearm"         "amplitude_yaw_forearm"  
## [37] "classe"
```

If we omit columns such as: "user_name", "cvtd_timestamp", "new_window", "classe", we think based on the column names we are dealing with numeric columns. Lets check check their summaries and try to find strange values:


```r
strangevalues <- train[,charvars] %>% 
  select(-c("user_name", "cvtd_timestamp", "new_window", "classe")) 

# lets build a long format df: variable name and value 
# so we can do group by each variable and check unique values
# and display top ten occurence of values
strangevalues %>% 
  pivot_longer(cols = colnames(strangevalues)) %>% 
  group_by(value) %>% 
  count() %>% 
  ungroup() %>% 
  arrange(desc(n)) %>% 
  head(10)
```

```
## # A tibble: 10 x 2
##    value          n
##    <chr>      <int>
##  1 ""        476157
##  2 "#DIV/0!"   2461
##  3 "0.00"       523
##  4 "0.0000"     276
##  5 "-1.2"        96
##  6 "-1.1"        86
##  7 "-1.5"        82
##  8 "-0.8"        78
##  9 "-1.4"        76
## 10 "-0.9"        74
```

As seen above a lot of missin values are written as blank string "", and also there are a lot of values "#DIV/0!" (probably some number division error). We will first force all this column to be converted to numeric, and then we will check missing values inside this columns, and finally remove some additional columns with a lot of missing values:


```r
vars.force.conv <- colnames(strangevalues) # list of variables to force conversion

# force conversion (train & test) - from char to num
train <- train %>% 
  mutate_at(.vars = vars.force.conv, .funs = as.numeric)
test <- test %>% 
  mutate_at(.vars = vars.force.conv, .funs = as.numeric)
```


Now we will check again the percentage of missing values for columns we converted from character to numeric in previous step:


```r
# calculate % of missing rows in each column (observed columns)
NAs <- train[, vars.force.conv] %>% 
  map(., ~sum(is.na(.))) %>% 
  unlist() / nrow(train) 

# create a DF: var and % of missing values in that var
NAs <- data.frame(var = names(NAs),
                  missing = NAs) %>% 
  mutate(missing = round(missing * 100, digits = 1)) %>% 
  rename(`missing rows %` = missing) %>% 
  arrange(desc(`missing rows %`))
rownames(NAs) <- NULL

# extract names of variables to drop
drop.vars <- NAs %>% filter(`missing rows %` > 90) %>% pull(var) # list of columns to drop (more then 50 % NA)

# show percentage of missing rows
NAs %>% 
  kbl() %>% 
  kable_paper() %>%
  scroll_box(width = "400px", height = "500px")
```

<div style="border: 1px solid #ddd; padding: 0px; overflow-y: scroll; height:500px; overflow-x: scroll; width:400px; "><table class=" lightable-paper" style='font-family: "Arial Narrow", arial, helvetica, sans-serif; margin-left: auto; margin-right: auto;'>
 <thead>
  <tr>
   <th style="text-align:left;position: sticky; top:0; background-color: #FFFFFF;"> var </th>
   <th style="text-align:right;position: sticky; top:0; background-color: #FFFFFF;"> missing rows % </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> kurtosis_yaw_belt </td>
   <td style="text-align:right;"> 100.0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> skewness_yaw_belt </td>
   <td style="text-align:right;"> 100.0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> kurtosis_yaw_dumbbell </td>
   <td style="text-align:right;"> 100.0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> skewness_yaw_dumbbell </td>
   <td style="text-align:right;"> 100.0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> kurtosis_yaw_forearm </td>
   <td style="text-align:right;"> 100.0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> skewness_yaw_forearm </td>
   <td style="text-align:right;"> 100.0 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> kurtosis_roll_forearm </td>
   <td style="text-align:right;"> 98.5 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> kurtosis_picth_forearm </td>
   <td style="text-align:right;"> 98.5 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> skewness_roll_forearm </td>
   <td style="text-align:right;"> 98.5 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> skewness_pitch_forearm </td>
   <td style="text-align:right;"> 98.5 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> max_yaw_forearm </td>
   <td style="text-align:right;"> 98.5 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> min_yaw_forearm </td>
   <td style="text-align:right;"> 98.5 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> amplitude_yaw_forearm </td>
   <td style="text-align:right;"> 98.5 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> kurtosis_roll_arm </td>
   <td style="text-align:right;"> 98.4 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> kurtosis_picth_arm </td>
   <td style="text-align:right;"> 98.4 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> skewness_roll_arm </td>
   <td style="text-align:right;"> 98.4 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> skewness_pitch_arm </td>
   <td style="text-align:right;"> 98.4 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> kurtosis_picth_belt </td>
   <td style="text-align:right;"> 98.2 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> skewness_roll_belt_1 </td>
   <td style="text-align:right;"> 98.2 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> kurtosis_roll_belt </td>
   <td style="text-align:right;"> 98.1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> skewness_roll_belt </td>
   <td style="text-align:right;"> 98.1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> max_yaw_belt </td>
   <td style="text-align:right;"> 98.1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> min_yaw_belt </td>
   <td style="text-align:right;"> 98.1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> amplitude_yaw_belt </td>
   <td style="text-align:right;"> 98.1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> kurtosis_yaw_arm </td>
   <td style="text-align:right;"> 98.1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> skewness_yaw_arm </td>
   <td style="text-align:right;"> 98.1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> kurtosis_roll_dumbbell </td>
   <td style="text-align:right;"> 98.1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> kurtosis_picth_dumbbell </td>
   <td style="text-align:right;"> 98.1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> skewness_roll_dumbbell </td>
   <td style="text-align:right;"> 98.1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> max_yaw_dumbbell </td>
   <td style="text-align:right;"> 98.1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> min_yaw_dumbbell </td>
   <td style="text-align:right;"> 98.1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> amplitude_yaw_dumbbell </td>
   <td style="text-align:right;"> 98.1 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> skewness_pitch_dumbbell </td>
   <td style="text-align:right;"> 98.0 </td>
  </tr>
</tbody>
</table></div>

Table above shows % of misisng rows for forcefully converted columns. As you can see, all columns have almost all missing values (due to blank strings and other characters). So we will also remove all this columns, since they do not bring any added value to the table:


```r
train <- train %>% select(-drop.vars) 
test  <- test %>% select(-drop.vars) 
```


Finally lets apply some essential variable data types transformations (on test and train data sets). We have to transform outcome variable to factor, and also there are some date time specific columns:


```r
train <- train %>%
  mutate(classe = as.factor(classe), # convert outcome to factor variable
         cvtd_timestamp = dmy_hm(cvtd_timestamp)) # convert to date time object
test <- test %>%
  mutate(classe = as.factor(classe), # convert outcome to factor variable
         cvtd_timestamp = dmy_hm(cvtd_timestamp)) # convert to date time object
```




## Exploratory Data Analysis (EDA)

First lets check how many users are included in the data:

```r
train %>% 
  count(user_name)
```

```
##   user_name    n
## 1    adelmo 2913
## 2  carlitos 2342
## 3   charles 2656
## 4    eurico 2310
## 5    jeremy 2527
## 6     pedro 1970
```

Now lets check summary of date & date-time related variables:


```r
# min/max raw timestamp part 1
train %>% 
  group_by(user_name) %>% 
  summarise(min_raw_ts1 = min(raw_timestamp_part_1),
            max_raw_ts1 = max(raw_timestamp_part_1))
```

```
## # A tibble: 6 x 3
##   user_name min_raw_ts1 max_raw_ts1
##   <chr>           <int>       <int>
## 1 adelmo     1322832772  1322832945
## 2 carlitos   1323084231  1323084356
## 3 charles    1322837808  1322837962
## 4 eurico     1322489605  1322489730
## 5 jeremy     1322673025  1322673166
## 6 pedro      1323094968  1323095081
```

```r
# min/max raw timestamp part 2
train %>% 
  group_by(user_name) %>% 
  summarise(min_raw_ts2 = min(raw_timestamp_part_2),
            max_raw_ts2 = max(raw_timestamp_part_2))
```

```
## # A tibble: 6 x 3
##   user_name min_raw_ts2 max_raw_ts2
##   <chr>           <int>       <int>
## 1 adelmo            294      998669
## 2 carlitos          367      998176
## 3 charles           315      997179
## 4 eurico           2647      998801
## 5 jeremy           2674      998716
## 6 pedro             317      996405
```
We think date and date-time related columns are irrelevant for model we are building (that will predict **classe** outcome). Therefore we will drop columns:

* **raw_timestamp_part_1**
* **raw_timestamp_part_2**
* **cvtd_timestamp**

Also we will drop column called **new_window**, but at this point we will keep column **num_window**.


```r
drop.vars <- c("raw_timestamp_part_1", "raw_timestamp_part_2",
               "cvtd_timestamp", "new_window")
train <- train %>% select(-drop.vars) 
test  <- test %>% select(-drop.vars) 
```

Now lets check how numerical variables are correlated between themselves. We will highlight predictor pairs with high correlation. High correlation might indicate some multicollinearity among predictors, but we hope that this phenomena won't cause any troubles due to selected prediction algorithms.


```r
# we check correlation for only numeric variables
relevant.vars <- colnames(train)[train %>% lapply(class) %in% c("numeric", "integer")]

cor.mat <- cor(train[, relevant.vars]) # correlation matrix
diag(cor.mat) <- 0 # set diagonal elements to zero
cor.mat[upper.tri(cor.mat)] <- 0 # set upper triangular elements to zero

# Top correlation
inds <- which(abs(cor.mat) > 0.8, arr.ind = TRUE) # index

# cerate table of pairs
cor.top <- data.frame(Var1 = rownames(cor.mat)[inds[, 1]], 
                      Var2 = colnames(cor.mat)[inds[, 2]], 
                      Cor = cor.mat[inds]) %>% 
  distinct() %>% 
  arrange(desc(abs(Cor)))
```

Now lets list predictors and their correlations:

```r
cor.top
```

```
##                Var1             Var2        Cor
## 1      accel_belt_z        roll_belt -0.9920233
## 2  gyros_dumbbell_z gyros_dumbbell_x -0.9836325
## 3  total_accel_belt        roll_belt  0.9805880
## 4      accel_belt_z total_accel_belt -0.9743945
## 5      accel_belt_x       pitch_belt -0.9644992
## 6   gyros_forearm_z gyros_dumbbell_z  0.9482223
## 7   gyros_forearm_z gyros_dumbbell_x -0.9333282
## 8      accel_belt_z     accel_belt_y -0.9322084
## 9      accel_belt_y total_accel_belt  0.9272485
## 10     accel_belt_y        roll_belt  0.9236888
## 11      gyros_arm_y      gyros_arm_x -0.9188930
## 12    magnet_belt_x     accel_belt_x  0.8871633
## 13    magnet_belt_x       pitch_belt -0.8780330
## 14  gyros_forearm_z  gyros_forearm_y  0.8694208
## 15 accel_dumbbell_z     yaw_dumbbell  0.8483225
## 16     magnet_arm_z     magnet_arm_y  0.8147423
## 17     magnet_arm_x      accel_arm_x  0.8143423
## 18         yaw_belt        roll_belt  0.8139859
## 19 accel_dumbbell_x   pitch_dumbbell  0.8077293
```

The table above shows pairs of predictors that are highly correlated (in positive or negative direction). We hope this will not cerate any problems for selected prediction algorithms. 

Now lets check how balanced is occurrence of each value in outcome variable (**classe**). Since we have a multi-class classification problem, we must select the adequate performance measure (it can be slightly more complicated than the case with a binary class outcome!):


```r
train %>% 
  group_by(classe) %>% 
  count()
```

```
## # A tibble: 5 x 2
## # Groups:   classe [5]
##   classe     n
##   <fct>  <int>
## 1 A       4185
## 2 B       2848
## 3 C       2567
## 4 D       2412
## 5 E       2706
```


We can see that classes "B", "C", "D", "E" are more than less balanced, but class "A" has slightly more observations compared to other classes. 



## Modeling

In the modeling stage we would like to fit parameters of selected model using train data set. In order to predict outcome variable using test data set (and later validation or final test data set). Selected model types are (keep in mind we are dealing with multi- class classification problem!):

* Multinomial logistic regression classification algorithm (R package - **nnet**)
* Naive Bayes classification algorithm (R package **naivebayes**)
* k-nearest neighbors (KNN) classification algorithm (inside **caret** package)
* Recursive Partitioning And Regression Trees classification algorithm (R package - **rpart**)
* Random Forests classification algorithm (R package *rf*)
* Xgboost ~ eXtreme Gradient Boosting classification algorithm (R package **xgboost**)


First lets set control environment for each model (we are using 10-fold cross-validation in the model fitting stage) and also we would:


```r
## Set control parameters for caret package
#  - cross validations settings (number of folds)
#  - type of output of models - prediction + probabilities 

nr.CV_folds <- 10 # number of folds - cross validation

# base setting
control_setting_ <- trainControl(method ="cv",  
                                number = nr.CV_folds,
                                classProbs = T)
```


### Model training

Some model fit procedure uses parallel computing, if you will use the RMarkdown code please set number of cores according to your PV (where you will evaluate the code !).

Now lets execute model training (fit model parameters) using train dataset:


```r
# Model fit

## k-nearest neighbors (KNN)
set.seed(11235)
knn_time_start <- Sys.time() # time start
knn_fit <- train(classe ~ ., 
                 data = train, 
                 method = "knn", 
                 trControl = control_setting_)
knn_time_end <- Sys.time() # time end


## Recursive Partitioning And Regression Trees 
set.seed(11235)
rprt_time_start <- Sys.time() # time start
rprt_fit <- train(classe ~ ., 
                  data = train, 
                  method = "rpart", 
                  trControl = control_setting_)
rprt_time_end <- Sys.time() # time end


#----------------------------------------#
## Parallel core processing ..... begin
#----------------------------------------#

## All subsequent models are then run in parallel
nc <- 16 # number of cores
#cl <- makeCluster(nc) # set cores number (to be used)
cl <- makePSOCKcluster(nc) # set cores number 
registerDoParallel(cl) 


## Multinomial logistic regression
set.seed(11235)
mlr_time_start <- Sys.time() # time start
mlr_fit <- train(classe ~ ., 
                 data = train, 
                 method = "nnet", 
                 trControl = control_setting_,
                 allowParallel = T)
```

```
## # weights:  325
## initial  value 25134.493018 
## iter  10 value 21852.788974
## iter  20 value 21534.610199
## iter  30 value 21076.298589
## iter  40 value 20707.307251
## iter  50 value 20497.477938
## iter  60 value 20217.934443
## iter  70 value 19993.676606
## iter  80 value 19795.367589
## iter  90 value 19369.300715
## iter 100 value 19272.200591
## final  value 19272.200591 
## stopped after 100 iterations
```

```r
mlr_time_end <- Sys.time() # time end


## Naive Bayes
set.seed(11235)
nb_time_start <- Sys.time() # time start
nb_fit <- train(classe ~ ., 
                data = train, 
                method = "nb", 
                trControl = control_setting_,
                allowParallel = T)
nb_time_end <- Sys.time() # time end


## Random Forests
set.seed(11235)
rf_time_start <- Sys.time() # time start
rf_fit <- train(classe ~ ., 
                data = train, 
                method = "rf", 
                trControl = control_setting_,
                allowParallel = T)
rf_time_end <- Sys.time() # time end



## Xgboost ~ eXtreme Gradient Boosting
set.seed(11235)
xgb_time_start <- Sys.time() # time start
xgb_fit <- train(classe ~ ., 
                 data = train, 
                 method = "xgbTree",
                 trControl = control_setting_,
                 allowParallel = T)
```

```
## [18:47:03] WARNING: amalgamation/../src/learner.cc:516: 
## Parameters: { allowParallel } might not be used.
## 
##   This may not be accurate due to some parameters are only used in language bindings but
##   passed down to XGBoost core.  Or some parameters are not used but slip through this
##   verification. Please open an issue if you find above cases.
```

```r
xgb_time_end <- Sys.time() # time end


## When you are done:
on.exit(stopCluster(cl))
env <- foreach:::.foreachGlobals
rm(list=ls(name=env), pos=env)
  

#----------------------------------------#
## Parallel core processing ..... end
#----------------------------------------#
```

Time spent to fit each model are shown below:


```r
# Gather model estimation running time
model.fit.time <- tibble(model = c("k-nearest neighbors", "Recursive Partitioning And Regression Trees", "Multinomial logistic regression", "Naive Bayes", "Random Forests", "Xgboost"),
                            `multi-core estimation` = c(F, F, T, T, T, T),
                            `time in sec` = c(round(difftime(knn_time_end, knn_time_start, units = "secs"),0),
round(difftime(rprt_time_end, rprt_time_start, units = "secs"),0),
round(difftime(mlr_time_end, mlr_time_start, units = "secs"),0),
round(difftime(nb_time_end, nb_time_start, units = "secs"),0),
round(difftime(rf_time_end, rf_time_start, units = "secs"),0),
round(difftime(xgb_time_end, xgb_time_start, units = "secs"),0)))

# show time
model.fit.time %>% 
  kbl() %>% 
  kable_paper() %>%
  scroll_box(width = "100%", height = "100%")
```

<div style="border: 1px solid #ddd; padding: 0px; overflow-y: scroll; height:100%; overflow-x: scroll; width:100%; "><table class=" lightable-paper" style='font-family: "Arial Narrow", arial, helvetica, sans-serif; margin-left: auto; margin-right: auto;'>
 <thead>
  <tr>
   <th style="text-align:left;position: sticky; top:0; background-color: #FFFFFF;"> model </th>
   <th style="text-align:left;position: sticky; top:0; background-color: #FFFFFF;"> multi-core estimation </th>
   <th style="text-align:left;position: sticky; top:0; background-color: #FFFFFF;"> time in sec </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> k-nearest neighbors </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> 71 secs </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Recursive Partitioning And Regression Trees </td>
   <td style="text-align:left;"> FALSE </td>
   <td style="text-align:left;"> 6 secs </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Multinomial logistic regression </td>
   <td style="text-align:left;"> TRUE </td>
   <td style="text-align:left;"> 62 secs </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Naive Bayes </td>
   <td style="text-align:left;"> TRUE </td>
   <td style="text-align:left;"> 49 secs </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Random Forests </td>
   <td style="text-align:left;"> TRUE </td>
   <td style="text-align:left;"> 238 secs </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Xgboost </td>
   <td style="text-align:left;"> TRUE </td>
   <td style="text-align:left;"> 330 secs </td>
  </tr>
</tbody>
</table></div>



### Model benchmark


## Results

