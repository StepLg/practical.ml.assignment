# Recognition of common mistakes in weight lifting exercises based on on-body sensors
Stepan Kuntco <steplg@gmail.com>  
Saturday, September 20, 2014  

## Summary
In this project there was attempt to train a multi-label classifier to predict common mistakes in weight-lifting excersizes. Data was originally collected in [Human Activity Recognition project](http://groupware.les.inf.puc-rio.br/har) (Weight Lifting Exercise Dataset).

During analysis 4 models were able to show almost 100% accuracy on test set (which was randomly sampled from all data): Random forests, Polynomial svm, Stochastic Gradient Boosting and Bagged CART. Among them Polynomial SVM has the lowest training time (10 minutes vs 23 min for GBM, 34 min for RF and 41 for Bagged CART).

## Environment



```r
sessionInfo()
```

```
## R version 3.1.1 (2014-07-10)
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## 
## locale:
## [1] LC_COLLATE=English_United States.1252 
## [2] LC_CTYPE=English_United States.1252   
## [3] LC_MONETARY=English_United States.1252
## [4] LC_NUMERIC=C                          
## [5] LC_TIME=English_United States.1252    
## 
## attached base packages:
## [1] splines   parallel  stats     graphics  grDevices utils     datasets 
## [8] methods   base     
## 
## other attached packages:
##  [1] pROC_1.7.3          kernlab_0.9-19      randomForest_4.6-10
##  [4] ipred_0.9-3         gbm_2.1             survival_2.37-7    
##  [7] plyr_1.8.1          rpart_4.1-8         MASS_7.3-33        
## [10] doParallel_1.0.8    iterators_1.0.7     foreach_1.4.2      
## [13] caret_6.0-35        ggplot2_1.0.0       lattice_0.20-29    
## [16] rCharts_0.4.5      
## 
## loaded via a namespace (and not attached):
##  [1] BradleyTerry2_1.0-5 brglm_0.5-9         car_2.0-21         
##  [4] class_7.3-10        codetools_0.2-8     colorspace_1.2-4   
##  [7] digest_0.6.4        evaluate_0.5.5      formatR_1.0        
## [10] grid_3.1.1          gtable_0.1.2        gtools_3.4.1       
## [13] htmltools_0.2.6     knitr_1.6           lava_1.2.6         
## [16] lme4_1.1-7          Matrix_1.1-4        minqa_1.2.3        
## [19] munsell_0.4.2       nlme_3.1-117        nloptr_1.0.4       
## [22] nnet_7.3-8          prodlim_1.4.5       proto_0.3-10       
## [25] Rcpp_0.11.2         reshape2_1.4        RJSONIO_1.3-0      
## [28] rmarkdown_0.3.3     scales_0.2.4        stringr_0.6.2      
## [31] tools_3.1.1         whisker_0.3-2       yaml_2.1.13
```

## Exploratory Analysis
This dataset comes from [Human Activity Recognition project](http://groupware.les.inf.puc-rio.br/har#dataset), "Weight Lifting Exercises Dataset". 


```r
# download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv', 'pml-training.csv')
# download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv', 'pml-testing.csv')

set.train <- read.csv('pml-training.csv', na.strings=c('NA', '#DIV/0!'))
set.validation <- read.csv('pml-testing.csv', na.strings=c('NA', '#DIV/0!'))
```

Set size:

```r
nrow(set.train)
```

```
## [1] 19622
```

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

```r
plot(set.train$classe, xlab='Activity', ylab='number of observations', main='Activities in dataset')
```

![plot of chunk unnamed-chunk-5](./report_files/figure-html/unnamed-chunk-5.png) 

First six variables in dataset seems to be not very important for our training:

```r
head(names(set.train), n=6)
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"
```
 * `X` - this is just order number.
 * `user_name` - participant name, which actually could be used in training, but only in the way of personalization.
 * `raw_timestamp_part_1`, `raw_timestamp_part_2`, `cvtd_timestamp` - various timestamps, which are also can't be used for training by itself. But they could be useful if we would be able to use some previous observations to predict current class, but in our conditions they are useless.
 * `new_window` - factor variable, by nature very similar to timestamps: it's useless unless we can't use previous observations.

```r
set.train.1 <- set.train[, c(-1: -6)]
```
 

There are also a number of columns which are mostly NA. We should remove them to ensure that our training methods work correct:

```r
na.columns <- apply(set.train.1, 2, function(x) sum(is.na(x)) == 0)
set.train.1 <- set.train.1[, na.columns]
```

Number of removed NA columns:

```r
sum(!na.columns)
```

```
## [1] 100
```

Number of variables left to use in training:

```r
ncol(set.train.1) - 1
```

```
## [1] 53
```

There are some highly correlated variables, but not too much, so we can leave them in the set:

```r
cor.matrix <- cor(subset(set.train.1, select=-c(classe)))
colnames(subset(set.train.1, select=-c(classe)))[findCorrelation(cor.matrix, 0.9)]
```

```
## [1] "accel_belt_z"     "roll_belt"        "accel_belt_y"    
## [4] "accel_belt_x"     "gyros_arm_y"      "gyros_forearm_z" 
## [7] "gyros_dumbbell_x"
```

## Training models
### Create train/test sets
The original data set was split into train and test sets with 25% for the test set and save them to separate files.

```r
set.seed(123)
train.ids <- createDataPartition(set.train.1$classe, p = 0.75)[[1]]

training <- set.train.1[train.ids, ]
testing <- set.train.1[-train.ids, ]
saveRDS(training, file='training.RDS')
saveRDS(testing, file='testing.RDS')
```

### Training models
Some models require much time to train, so running tasks inside `training` function in parallel will significantly reduce total train time.

```r
cl <- makeCluster(detectCores())
registerDoParallel(cl)
```

Train and save all models to separate file. Following methods were tried:
 * `lda` - Linear Discriminant Analysis
 * `qda` - Quadratic Discriminant Analysis
 * `rpart` - CART, Recursive Partitioning and Regression Trees
 * `gbm` - Stochastic Gradient Boosting
 * `rf` - Random Forest
 * `treebag` - Bagged CART
 * `svmPoly` - Support Vector Machines with Polynomial Kernel

```r
models.file <- 'models.RDS'
if (file.exists(models.file)) {
  models <- readRDS(models.file)
} else {
  models <- sapply(c('lda', 'qda', 'rpart', 'gbm', 'rf', 'treebag', 'svmPoly'),
                   function(x) {train(classe ~ ., data=training, method=x)}, USE.NAMES=T, simplify=F)
  saveRDS(models, file='models.RDS')
}
```

### Models summary

```r
models.summary <- data.frame(t(
  sapply(models, function(x) {
    confMatrix <- confusionMatrix(testing$classe, predict(x, newdata=testing))
    c(
      'Accuracy.CV'=unname(max(x$results['Accuracy'])),
      'Accuracy.Test'=unname(confMatrix$overall['Accuracy']),
      'Accuracy.Test.lower'=unname(confMatrix$overall['AccuracyLower']),
      'Accuracy.Test.upper'=unname(confMatrix$overall['AccuracyUpper']),
      'time.train.total'=unname(x$times$everything['user.self']))
  })
))


dt <- Datatables$new()
dt$addTable(data.frame(Model=row.names(models.summary), models.summary), bPaginate=F, bSearchable=F, bFilter=F)
dt$print('tbl_models_summary', include_assets=F, cdn=F)
```


<table id = 'tbl_models_summary' class = 'rChart datatables'></table>
<script type="text/javascript" charset="utf-8">
  var chartParamstbl_models_summary = {
 "dom": "tbl_models_summary",
"width":    800,
"height":    400,
"table": {
 "aaData": [
 [
 "lda",
0.71063,
0.70881,
0.69587,
0.7215,
  0.92 
],
[
 "qda",
0.89666,
0.89478,
0.88585,
0.90323,
  0.65 
],
[
 "rpart",
0.55179,
0.48797,
0.47389,
0.50206,
   4.2 
],
[
 "gbm",
0.98445,
0.9845,
0.98064,
0.98777,
 23.41 
],
[
 "rf",
0.99597,
0.99857,
0.99706,
0.99943,
 34.69 
],
[
 "treebag",
0.99149,
0.99592,
0.99371,
0.99751,
  41.5 
],
[
 "svmPoly",
0.98929,
0.99205,
0.98914,
0.99434,
  9.92 
] 
],
"aoColumns": [
 {
 "sTitle": "Model" 
},
{
 "sTitle": "Accuracy.CV" 
},
{
 "sTitle": "Accuracy.Test" 
},
{
 "sTitle": "Accuracy.Test.lower" 
},
{
 "sTitle": "Accuracy.Test.upper" 
},
{
 "sTitle": "time.train.total" 
} 
],
"bPaginate": false,
"bSearchable": false,
"bFilter": false 
},
"id": "tbl_models_summary" 
}
  $('#' + chartParamstbl_models_summary.id).removeClass("rChart")

  $(document).ready(function() {
		drawDataTable(chartParamstbl_models_summary)
	});
  function drawDataTable(chartParams){
    var dTable = $('#' + chartParams.dom).dataTable(
      chartParams.table
    );
    //first use rCharts width
  	$('#'+chartParams.id+"_wrapper").css("width",chartParams.width)  
		$('#'+chartParams.id+"_wrapper").css("width",chartParams.table.width)
    
    //then if specified change to table width
    $('#'+chartParams.id+"_wrapper").css("margin-left", "auto");
    $('#'+chartParams.id+"_wrapper").css("margin-right", "auto");
		dTable.fnAdjustColumnSizing();
  }
		
</script>
`rpart`, `qda` and `lda` has pure accuracy but fast training time. `svmPoly` model has almost 100 accuracy and lowest training time.

### Variables importance for good models

```r
svmPoly.varImp <- varImp(models$svmPoly)$importance
models.varImp <- cbind(gbm=unname(varImp(models$gbm)$importance),
                       rf=unname(varImp(models$rf)$importance),
                       treebag=unname(varImp(models$treebag)$importance),
                       svmPoly=unname(data.frame(rowSums(svmPoly.varImp) / ncol(svmPoly.varImp)))
                       )

dt <- Datatables$new()
dt$addTable(data.frame("Variable name"=row.names(models.varImp), models.varImp))
dt$print('tbl_models_varImp', include_assets=F, cdn=F)
```


<table id = 'tbl_models_varImp' class = 'rChart datatables'></table>
<script type="text/javascript" charset="utf-8">
  var chartParamstbl_models_varImp = {
 "dom": "tbl_models_varImp",
"width":    800,
"height":    400,
"table": {
 "aaData": [
 [
 "num_window",
   100,
   100,
12.152,
44.879 
],
[
 "roll_belt",
84.658,
66.852,
 2.977,
44.524 
],
[
 "pitch_belt",
10.336,
28.198,
1.2649,
10.891 
],
[
 "yaw_belt",
28.126,
33.435,
 3.648,
17.952 
],
[
 "total_accel_belt",
     0,
2.2087,
6.2729,
13.727 
],
[
 "gyros_belt_x",
0.18714,
0.84626,
21.773,
4.9369 
],
[
 "gyros_belt_y",
2.1908,
 1.185,
  10.2,
4.0073 
],
[
 "gyros_belt_z",
9.1545,
5.1602,
28.443,
 16.95 
],
[
 "accel_belt_x",
     0,
1.1261,
15.434,
21.526 
],
[
 "accel_belt_y",
     0,
1.2314,
12.088,
10.552 
],
[
 "accel_belt_z",
1.3439,
9.3005,
2.2629,
30.148 
],
[
 "magnet_belt_x",
2.4899,
6.5306,
5.0027,
24.699 
],
[
 "magnet_belt_y",
 3.437,
7.3341,
2.6237,
 62.67 
],
[
 "magnet_belt_z",
13.984,
7.8184,
2.6031,
49.849 
],
[
 "roll_arm",
4.1767,
5.3291,
     0,
25.998 
],
[
 "pitch_arm",
     0,
2.8502,
2.5406,
41.429 
],
[
 "yaw_arm",
3.9374,
3.5705,
3.6041,
12.373 
],
[
 "total_accel_arm",
0.78584,
0.87885,
10.979,
35.136 
],
[
 "gyros_arm_x",
0.47998,
1.0858,
2.1446,
6.2769 
],
[
 "gyros_arm_y",
2.5124,
1.6095,
5.8954,
 8.562 
],
[
 "gyros_arm_z",
     0,
     0,
0.67843,
3.1336 
],
[
 "accel_arm_x",
     0,
 3.705,
0.37132,
61.429 
],
[
 "accel_arm_y",
     0,
1.4949,
1.5555,
21.252 
],
[
 "accel_arm_z",
0.31646,
0.9776,
1.1975,
26.949 
],
[
 "magnet_arm_x",
 5.468,
4.4387,
10.445,
66.326 
],
[
 "magnet_arm_y",
0.69909,
3.0767,
5.5957,
63.226 
],
[
 "magnet_arm_z",
 3.474,
1.8597,
 1.642,
47.078 
],
[
 "roll_dumbbell",
11.988,
13.023,
12.534,
69.903 
],
[
 "pitch_dumbbell",
0.43747,
  2.45,
 21.75,
61.173 
],
[
 "yaw_dumbbell",
     0,
5.1774,
13.704,
28.967 
],
[
 "total_accel_dumbbell",
1.5581,
9.0309,
21.913,
30.362 
],
[
 "gyros_dumbbell_x",
0.69528,
1.3472,
46.733,
7.7415 
],
[
 "gyros_dumbbell_y",
8.7707,
3.6036,
33.886,
14.243 
],
[
 "gyros_dumbbell_z",
0.33019,
0.46119,
4.2675,
5.2577 
],
[
 "accel_dumbbell_x",
6.3554,
4.3954,
6.0165,
52.363 
],
[
 "accel_dumbbell_y",
8.3311,
12.702,
 11.82,
38.492 
],
[
 "accel_dumbbell_z",
9.0183,
8.4857,
   100,
35.787 
],
[
 "magnet_dumbbell_x",
4.2327,
10.949,
3.4237,
60.219 
],
[
 "magnet_dumbbell_y",
22.446,
 29.27,
49.072,
56.521 
],
[
 "magnet_dumbbell_z",
 28.73,
29.094,
6.5349,
44.528 
],
[
 "roll_forearm",
19.023,
25.028,
 44.56,
27.829 
],
[
 "pitch_forearm",
47.211,
40.364,
9.7933,
80.064 
],
[
 "yaw_forearm",
0.24711,
3.0582,
53.676,
21.966 
],
[
 "total_accel_forearm",
0.41342,
0.79986,
26.991,
27.178 
],
[
 "gyros_forearm_x",
     0,
0.24226,
39.703,
4.9319 
],
[
 "gyros_forearm_y",
     0,
 1.211,
1.1602,
5.2791 
],
[
 "gyros_forearm_z",
0.55462,
0.5832,
15.105,
9.1689 
],
[
 "accel_forearm_x",
11.242,
11.207,
 11.98,
63.775 
],
[
 "accel_forearm_y",
1.0027,
1.5324,
0.78292,
25.239 
],
[
 "accel_forearm_z",
3.8344,
4.6936,
4.8136,
11.172 
],
[
 "magnet_forearm_x",
2.7887,
2.5277,
52.143,
54.583 
],
[
 "magnet_forearm_y",
0.60955,
 3.539,
10.129,
36.147 
],
[
 "magnet_forearm_z",
6.0846,
8.0294,
 7.563,
26.512 
] 
],
"aoColumns": [
 {
 "sTitle": "Variable.name" 
},
{
 "sTitle": "gbm" 
},
{
 "sTitle": "rf" 
},
{
 "sTitle": "treebag" 
},
{
 "sTitle": "svmPoly" 
} 
] 
},
"id": "tbl_models_varImp" 
}
  $('#' + chartParamstbl_models_varImp.id).removeClass("rChart")

  $(document).ready(function() {
		drawDataTable(chartParamstbl_models_varImp)
	});
  function drawDataTable(chartParams){
    var dTable = $('#' + chartParams.dom).dataTable(
      chartParams.table
    );
    //first use rCharts width
  	$('#'+chartParams.id+"_wrapper").css("width",chartParams.width)  
		$('#'+chartParams.id+"_wrapper").css("width",chartParams.table.width)
    
    //then if specified change to table width
    $('#'+chartParams.id+"_wrapper").css("margin-left", "auto");
    $('#'+chartParams.id+"_wrapper").css("margin-right", "auto");
		dTable.fnAdjustColumnSizing();
  }
		
</script>

### Final model coefficients
#### Stochastic Gradient Boosting

```r
models$gbm$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 53 predictors of which 43 had non-zero influence.
```

#### Random forests

```r
models$rf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.22%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4183    1    0    0    1   0.0004779
## B    5 2838    4    1    0   0.0035112
## C    0    7 2560    0    0   0.0027269
## D    0    0    8 2404    0   0.0033167
## E    0    0    0    5 2701   0.0018477
```

#### Bagged CART

```r
models$treebag$finalModel
```

```
## 
## Bagging classification trees with 25 bootstrap replications
```

#### Support Vector Machines with Polynomial Kernel

```r
models$svmPoly$finalModel
```

```
## Support Vector Machine object of class "ksvm" 
## 
## SV type: C-svc  (classification) 
##  parameter : cost C = 0.5 
## 
## Polynomial kernel function. 
##  Hyperparameters : degree =  3  scale =  0.1  offset =  1 
## 
## Number of Support Vectors : 3190 
## 
## Objective Function Value : -27.85 -5.033 -4.2 -1.807 -12.73 -3.236 -3.291 -59.88 -6.717 -12.55 
## Training error : 0.001427
```
