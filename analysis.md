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



## Modeling


## Results

