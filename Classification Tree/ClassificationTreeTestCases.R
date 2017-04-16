library("mlbench")
library("slam")

data(Sonar)

SonarData <- Sonar[,1:60]

class.labels <- ifelse(Sonar$Class == "R", 0, 1)

dtm <- as.simple_triplet_matrix(SonarData)

train.idx <- sample(1:nrow(SonarData), 0.7*nrow(SonarData))
test.idx <- seq(1, nrow(SonarData))[-train.idx]

train.dtm <- dtm[train.idx,]
test.dtm <- dtm[test.idx,]

test.tree <- function(train.dtm, test.dtm, train.idx, test.idx, class.labels) {
  df.train <- data.frame("i"=train.dtm$i, "j"=train.dtm$j, "v"=train.dtm$v)
  models <- cpp__tree(df.train, class.labels[train.idx], cvRounds = 1)
  
  df.test <- data.frame("i"=test.dtm$i, "j"=test.dtm$j, "v"=test.dtm$v)
  preds <- cpp__test(df.test, models)
  
  preds <- unlist(lapply(preds, function(x) as.integer(names(which(x == max(x)))[1])))
  mean(preds == class.labels[test.idx][as.integer(names(preds))])
}

replicate(10, test.tree(train.dtm, test.dtm, train.idx, test.idx, class.labels))


test.adaboost <- function(train.dtm, test.dtm, train.idx, test.idx, class.labels) {
  df.train <- data.frame("i"=train.dtm$i, "j"=train.dtm$j, "v"=train.dtm$v)
  models <- cpp__adaBoostedTree(df.train, class.labels[train.idx], boostingRounds = 5, maxDepth = 4, cvRounds = 1)
  
  df.test <- data.frame("i"=test.dtm$i, "j"=test.dtm$j, "v"=test.dtm$v)
  preds <- cpp__test(df.test, models)
  
  preds <- unlist(lapply(preds, function(x) as.integer(names(which(x == max(x))))))
  mean(preds == class.labels[test.idx][as.integer(names(preds))])
}

replicate(10, test.adaboost(train.dtm, test.dtm, train.idx, test.idx, class.labels))