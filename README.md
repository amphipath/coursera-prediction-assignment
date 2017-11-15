# coursera prediction assignment

### Preliminaries

|Package        |Version    |
|---------------|-----------|
|caret          |6.0-77     |
|e1071          |1.6-8      |
|ElemStatLearn  |2015.6.26  |

`set.seed(5318008)` will be used throughout

### Initial view of data
Looking at the data in excel, majority of the data are NA's or missing, and a huge majority of the columns are completely meaningless except specifically for rows that have `new_window` as `yes`. This probably corresponds to the start of a recording window in which the device snapshots some coordinates that would be too resource-intensive to constantly monitor.

Using the time-tested trial of common sense, it then is most likely that any data point that has the same `num_window` would correspond to the same exercise motion and hence the same `classe`.

To test this theory:

```r
training <- read.csv("plm-training.csv")
theorytest <- unique(data.frame(training$num_window,training$classe))
length(unique(theorytest[,1]))-length(theorytest[,1])

[1] 0
```

No repeated values in `theorytest$num_window` indicates each value of `num_window` corresponds to exactly one value in `classe`. Hey, whaddya know.

Supposing that the testing data was taken from the same source/experiments/exercises that the training data was, then it would be possible to predict with perfect accuracy the `classe` of all of the testing data. 

Just for the hell of it, let's throw some cross validation in:

```r
train_Control <- trainControl(method="repeatedcv",number=20,repeats=5)
perfectModel <- train(classe~num_window,data=training,method="ada")
print(perfectModel)

Random Forest 

19622 samples
    1 predictor
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Bootstrapped (25 reps) 
Summary of sample sizes: 19622, 19622, 19622, 19622, 19622, 19622, ... 
Resampling results:

  Accuracy   Kappa   
  0.9997722  0.999712
```
  
Pretty good as far as accuracy goes.

As much as it would satisfy the purposes of this assignment, it would also create a model with probably one of the worst cases of overfitting and be completely inapplicable as a model to predict on future exercise data. So let's create a model that uses some actual machine learning.

### Trimming the fat

First thing to do if we're to do some actual data analysis is to cut out all the nonsense columns that are related to the windows.

```r
training <- training[training$new_window != "yes",]
keepcolumn <- vector(mode="logical",length=ncol(training))

for(i in 1:ncol(training)) {
    for(j in 1:nrow(training)) {
    if(!is.na(training[j,i]) & training[j,i] != "") {
        keepcolumn[i] <- TRUE
        break
        }
    }
}

training <- training[,keepcolumn]
testing <- testing[,keepcolumn]
```

I don't want to keep the timestamps or windows either, so

```r
training <- training[,-3:-7]
testing <- testing [,-3:-7]
```

### A first model

```r
train_Control <- trainControl(method="repeatedcv",number=10,repeats=3)
imperfectModel <- train(classe~.,data=training[,-1:-2],trControl=train_Control,method="nb")
print(imperfectModel)

Naive Bayes 

19216 samples
   52 predictor
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Cross-Validated (10 fold, repeated 3 times) 
Summary of sample sizes: 17294, 17293, 17295, 17294, 17294, 17295, ... 
Resampling results across tuning parameters:

  usekernel  Accuracy   Kappa    
  FALSE      0.5052036  0.3888115
   TRUE      0.7450556  0.6740567

Tuning parameter 'fL' was held constant at a value of 0
Tuning parameter 'adjust' was held constant at a value of 1
Accuracy was used to select the optimal model using  the largest value.
The final values used for the model were fL = 0, usekernel = TRUE and adjust = 1.
```

An accuracy of 74.5% sets a lower bound of what we should go for.

Trying another model:

```r
imperfectModel2 <- train(classe~.,data=training[,-1:-2],method="gbm",verbose=FALSE)
print(imperfectModel2)
```

Using crossvalidation for gbm proved to make it take obscenely long and hence was omitted this round.

```r
Stochastic Gradient Boosting 

19216 samples
   52 predictor
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Bootstrapped (25 reps) 
Summary of sample sizes: 19216, 19216, 19216, 19216, 19216, 19216, ... 
Resampling results across tuning parameters:

  interaction.depth  n.trees  Accuracy   Kappa    
  1                   50      0.7515771  0.6849533
  1                  100      0.8197656  0.7718241
  1                  150      0.8522448  0.8129752
  2                   50      0.8552716  0.8165568
  2                  100      0.9049832  0.8797286
  2                  150      0.9294126  0.9106541
  3                   50      0.8956938  0.8679085
  3                  100      0.9408040  0.9250706
  3                  150      0.9599428  0.9493060

Tuning parameter 'shrinkage' was held constant at a value of 0.1
Tuning parameter 'n.minobsinnode' was held constant at
 a value of 10
Accuracy was used to select the optimal model using  the largest value.
The final values used for the model were n.trees = 150, interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

We have a winner; I estimate the out-of-sample accuracy to be about 90% to take into account possible overfitting.