require(caret)
library(doParallel)
cl <- makeCluster(detectCores())
registerDoParallel(cl)


pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}


download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv', 'pml-training.csv')
download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv', 'pml-testing.csv')

set.train <- read.csv('pml-training.csv', na.strings=c('NA', '#DIV/0!'))
set.validation <- read.csv('pml-testing.csv', na.strings=c('NA', '#DIV/0!'))

set.train.1 <- set.train[, c(-1: -6)]

na.columns <- apply( set.train.1 , 2 , function(x) sum(is.na(x)) == 0)
set.train.1 <- set.train.1[, na.columns]

train.ids <- createDataPartition(set.train.1$classe, p = 3/4)[[1]]

training <- set.train.1[train.ids, ]
testing <- set.train.1[-train.ids, ]

###############

fit.LDA <- train(classe ~ ., data=set.train.1, method="lda")
fit.LDA
varImp(fit.LDA)
dotPlot(varImp(fit.LDA), main="LDA: Dotplot of variable importance values")

confusionMatrix(testing$classe, predict(fit.LDA, newdata=testing))
confusionMatrix(testing$classe, predict(fit.LDA, newdata=testing))$overall['Accuracy']
apply(confusionMatrix(testing$classe, predict(fit.LDA, newdata=testing))$table, 2, function(x) x / sum(x))

final.LDA <- predict(fit.LDA, newdata=set.validation)
final.LDA

###############

fit.QDA <- train(classe ~ ., data=set.train.1, method="qda")
fit.QDA
varImp(fit.QDA)
dotPlot(varImp(fit.QDA), main="QDA: Dotplot of variable importance values")

confusionMatrix(testing$classe, predict(fit.QDA, newdata=testing))
confusionMatrix(testing$classe, predict(fit.QDA, newdata=testing))$overall['Accuracy']
apply(confusionMatrix(testing$classe, predict(fit.QDA, newdata=testing))$table, 2, function(x) x / sum(x))

final.QDA <- predict(fit.QDA, newdata=set.validation)
final.QDA

###############

fit.rpart <- train(classe ~ ., data=set.train.1, method="rpart")
fit.rpart
varImp(fit.rpart)
dotPlot(varImp(fit.rpart), main="rpart: Dotplot of variable importance values")

confusionMatrix(testing$classe, predict(fit.rpart, newdata=testing))
confusionMatrix(testing$classe, predict(fit.rpart, newdata=testing))$overall['Accuracy']
apply(confusionMatrix(testing$classe, predict(fit.rpart, newdata=testing))$table, 2, function(x) x / sum(x))

final.rpart <- predict(fit.rpart, newdata=set.validation)
final.rpart

###############

fit.GBM <- train(classe ~ ., data=set.train.1, method="gbm")
fit.GBM
varImp(fit.GBM)
dotPlot(varImp(fit.GBM), main="GBM: Dotplot of variable importance values")

confusionMatrix(testing$classe, predict(fit.GBM, newdata=testing))
confusionMatrix(testing$classe, predict(fit.GBM, newdata=testing))$overall['Accuracy']
apply(confusionMatrix(testing$classe, predict(fit.GBM, newdata=testing))$table, 2, function(x) x / sum(x))

final.GBM <- predict(fit.GBM, newdata=set.validation)
final.GBM
pml_write_files(final.GBM)

###############

fit.rf <- train(classe ~ ., data=set.train.1, method="rf")
fit.rf
varImp(fit.rf)
dotPlot(varImp(fit.rf), main="rf: Dotplot of variable importance values")

confusionMatrix(testing$classe, predict(fit.rf, newdata=testing))
confusionMatrix(testing$classe, predict(fit.rf, newdata=testing))$overall['Accuracy']
apply(confusionMatrix(testing$classe, predict(fit.rf, newdata=testing))$table, 2, function(x) x / sum(x))

final.rf <- predict(fit.rf, newdata=set.validation)
final.rf
pml_write_files(final.rf)

###############

fit.treebag <- train(classe ~ ., data=set.train.1, method="treebag")
confusionMatrix(testing$classe, predict(fit.treebag, newdata=testing))$overall['Accuracy']

###############

fit.svmPoly <- train(classe ~ ., data=set.train.1, method="svmPoly")
confusionMatrix(testing$classe, predict(fit.svmPoly, newdata=testing))$overall['Accuracy']

###############

http://stackoverflow.com/questions/15585501/usage-of-caret-with-gbm-method-for-multiclass-classification
https://www.youtube.com/watch?v=7Jbb2ItbTC4
https://earlglynn.shinyapps.io/ShinyCaret/
https://github.com/EarlGlynn/earlglynn.github.io


model.gbm1 <- train(classe ~ ., data=testing, method='gbm')

confusionMatrix(predict(train(classe ~ ., data=training, method='glm'), testing), testing$classe)


pc <- preProcess(training[, feature.names], 'pca')

model.gbm1 <- train(predict(pc, training[, feature.names]), training$classe, method='gbm')

training.pc <- predict(pc, training[, feature.names])
testing.pc <- predict(pc, testing[, feature.names])

model.gbm1 <- gbm(classe ~ ., data=transform(training.pc, classe=training$classe),
                  distribution='multinomial',
                  n.trees=100, verbose=T, shrinkage=0.1)

confusionMatrix(predict(model.gbm1, testing.pc, n.trees=100, type="response"), testing$classe)
pred <- predict.gbm(model.gbm1, testing.pc, n.trees=100, type="response")
pred1 <- pred
dim(pred1) <- dim(pred1)[c(1, 2)]
pred1 <- as.data.frame(pred1)
summary(as.factor(colnames(pred)[max.col(pred)]))

conf <- confusionMatrix(as.factor(levels(training$classe)[max.col(pred1)]), testing$classe)$table
apply(conf, 2, function(x) x / sum(x))
write.table(data.frame(actula=testing$classe, predicted=as.factor(levels(training$classe)[max.col(pred1)])), file='test1.tsv', quote=F, sep='\t', row.names=F, col.names=T)






