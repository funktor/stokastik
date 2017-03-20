library("mlbench")
library("slam")
#Rcpp::sourceCpp('ClassificationTree.cpp')

data(Sonar)

SonarData <- Sonar[,1:60]

class.labels <- ifelse(Sonar$Class == "R", 0, 1)

dtm <- as.simple_triplet_matrix(SonarData)

train.idx <- sample(1:nrow(SonarData), 0.7*nrow(SonarData))
test.idx <- seq(1, nrow(SonarData))[-train.idx]

train.dtm <- dtm[train.idx,]
test.dtm <- dtm[test.idx,]

test <- function(train.dtm, test.dtm, train.idx, test.idx, class.labels) {
  df.train <- data.frame("i"=train.dtm$i, "j"=train.dtm$j, "v"=train.dtm$v)
  model <- cpp__tree(df.train, class.labels[train.idx])
  
  df.test <- data.frame("i"=test.dtm$i, "j"=test.dtm$j, "v"=test.dtm$v)
  preds <- cpp__test(df.test, model)
  
  preds <- preds[order(preds$Doc),]
  mean(preds$Class==class.labels[test.idx])
}

replicate(10, test(train.dtm, test.dtm, train.idx, test.idx, class.labels))


test2 <- function(train.dtm, test.dtm, train.idx, test.idx, class.labels) {
  df.train <- data.frame("i"=train.dtm$i, "j"=train.dtm$j, "v"=train.dtm$v)
  models <- cpp__adaBoostedTree(df.train, class.labels[train.idx], boostingRounds = 10)
  
  df.test <- data.frame("i"=test.dtm$i, "j"=test.dtm$j, "v"=test.dtm$v)
  preds <- cpp__test(df.test, models)
  
  preds <- preds[order(preds$Doc),]
  mean(preds$Class==class.labels[test.idx])
}

replicate(10, test(train.dtm, test.dtm, train.idx, test.idx, class.labels))
