Rcpp::sourceCpp("../Utilities.cpp")

source("../Newsgroup.R")

fileFeatures <- cpp__fileFeatures(contents, tm::stopwords("en"), 1, 2)

dtm <- cpp__createDTM(fileFeatures)
dtm <- slam::simple_triplet_matrix(i=dtm$i, j=dtm$j, v=dtm$v, nrow=max(dtm$i), ncol=max(dtm$j), 
                                   dimnames=list("Docs"=as.character(1:max(dtm$i)), "Terms"=dtm$Terms))


selected.features <- cpp__mutualInformation(dtm$i, dtm$j, classes = labels, maxFeaturesPerClass = 150)
dtm <- dtm[,selected.features]

#dtm <- tm::weightBin(dtm)
dtm <- tm::as.DocumentTermMatrix(dtm, weighting = function(x) tm::weightTfIdf(x, normalize = T))

train.dtm <- dtm[1:length(train$Contents),]
test.dtm <- dtm[(length(train$Contents)+1):length(contents),]

df.train <- data.frame("i"=train.dtm$i, "j"=train.dtm$j, "v"=train.dtm$v)
models <- cpp__nb(df.train, train.labels, nrow(train.dtm), ncol(train.dtm), lambda = 1, maxIter = 1)

df.test <- data.frame("i"=test.dtm$i, "j"=test.dtm$j, "v"=test.dtm$v)
out <- cpp__nbTest(df.test, models)

preds <- unlist(lapply(out, function(x) as.integer(names(which(x == max(x)))[1])))
x <- mean(preds == test.labels[as.integer(names(preds))])




testFunction <- function(dtm, labels, train.pct=0.25, no.label.train.pct=0.5, lambda=1, maxIter=10) {
  train1.idx <- c()
  train2.idx <- c()
  test.idx <- c()
  
  w <- lapply(unique(labels), function(x) {
    idx <- which(labels == x)
    idx <- sample(idx)
    
    train1.idx <<- c(train1.idx, idx[1:floor(train.pct*length(idx))])
    train2.idx <<- c(train2.idx, idx[(floor(train.pct*length(idx))+1):floor((no.label.train.pct+train.pct)*length(idx))])
    test.idx <<- c(test.idx, idx[(floor((no.label.train.pct+train.pct)*length(idx))+1):length(idx)])
  })
  
  train.dtm <- dtm[c(sort(train1.idx), sort(train2.idx)),]
  train.labels <- c(labels[sort(train1.idx)], rep(-1, length(train2.idx)))
  
  test.dtm <- dtm[sort(test.idx),]
  test.labels <- labels[sort(test.idx)]
  
  df.train <- data.frame("i"=train.dtm$i, "j"=train.dtm$j, "v"=train.dtm$v)
  models <- cpp__nb(df.train, train.labels, nrow(train.dtm), ncol(train.dtm), lambda = lambda, maxIter = maxIter)
  
  df.test <- data.frame("i"=test.dtm$i, "j"=test.dtm$j, "v"=test.dtm$v)
  out <- cpp__nbTest(df.test, models)
  
  preds <- unlist(lapply(out, function(x) as.integer(names(which(x == max(x)))[1])))
  x <- mean(preds == test.labels[as.integer(names(preds))])
  
  train.dtm <- dtm[sort(train1.idx),]
  train.labels <- labels[sort(train1.idx)]
  
  df.train <- data.frame("i"=train.dtm$i, "j"=train.dtm$j, "v"=train.dtm$v)
  models <- cpp__nb(df.train, train.labels, nrow(train.dtm), ncol(train.dtm), lambda = 1, maxIter = 1)
  
  df.test <- data.frame("i"=test.dtm$i, "j"=test.dtm$j, "v"=test.dtm$v)
  out <- cpp__nbTest(df.test, models)
  
  preds <- unlist(lapply(out, function(x) as.integer(names(which(x == max(x)))[1])))
  y <- mean(preds == test.labels[as.integer(names(preds))])
  
  data.frame("x"=x, "y"=y)
}

testFunction(dtm, labels, 0.01, 0.75, 1, 5)
