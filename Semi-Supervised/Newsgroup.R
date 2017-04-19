Rcpp::sourceCpp("../Utilities.cpp")

getContentAndLabels <- function(folder.path) {
  dirs <- list.dirs(path = folder.path)[-1]
  
  file.contents <- c()
  cl.names <- c()
  
  for (dir in dirs) {
    files <- list.files(path = dir, full.names = T, recursive = T)
    
    contents <- lapply(files, function(x) tolower(tm::stripWhitespace(gsub("\\b[a-zA-Z]{1,2}\\b", " ", 
                                                                   gsub("[^a-zA-Z]", " ", 
                                                                        paste(readLines(x), collapse = " "))))))
    
    file.contents <- c(file.contents, contents)
    
    cl.names <- c(cl.names, rep(gsub("^.*\\/(.*)$", "\\1", dir), length(files)))
  }
  
  list("Contents"=file.contents, "ClassNames"=cl.names)
}

train <- getContentAndLabels("/Users/funktor/Downloads/20news-bydate/20news-bydate-train")
test <- getContentAndLabels("/Users/funktor/Downloads/20news-bydate/20news-bydate-test")

unique.class.names <- unique(c(train$ClassNames, test$ClassNames))

train.labels <- as.integer(sapply(train$ClassNames, function(x) which(unique.class.names == x)))
test.labels <- as.integer(sapply(test$ClassNames, function(x) which(unique.class.names == x)))

contents <- c(train$Contents, test$Contents)
labels <- c(train.labels, test.labels)

fileFeatures <- cpp__fileFeatures(contents, tm::stopwords("en"), 1, 2)

dtm <- cpp__createDTM(fileFeatures)
dtm <- slam::simple_triplet_matrix(i=dtm$i, j=dtm$j, v=dtm$v, nrow=max(dtm$i), ncol=max(dtm$j), 
                                   dimnames=list("Docs"=as.character(1:max(dtm$i)), "Terms"=dtm$Terms))


selected.features <- cpp__mutualInformation(dtm$i, dtm$j, classes = labels, maxFeaturesPerClass = 100)
dtm <- dtm[,selected.features]

dtm <- tm::as.DocumentTermMatrix(dtm, weighting = function(x) tm::weightTfIdf(x, normalize = T))

train1.idx <- c()
train2.idx <- c()
test.idx <- c()

w <- lapply(unique(labels), function(x) {
  idx <- which(labels == x)
  idx <- sample(idx)
  
  train1.idx <<- c(train1.idx, idx[1:floor(0.01*length(idx))])
  train2.idx <<- c(train2.idx, idx[(floor(0.01*length(idx))+1):floor(0.75*length(idx))])
  test.idx <<- c(test.idx, idx[(floor(0.75*length(idx))+1):length(idx)])
})

train.dtm <- dtm[c(sort(train1.idx), sort(train2.idx)),]
train.labels <- c(labels[sort(train1.idx)], rep(-1, length(train2.idx)))

test.dtm <- dtm[sort(test.idx),]
test.labels <- labels[sort(test.idx)]

df.train <- data.frame("i"=train.dtm$i, "j"=train.dtm$j, "v"=train.dtm$v)
models <- cpp__nb(df.train, train.labels, nrow(train.dtm), ncol(train.dtm), lambda = 1, maxIter = 10)

df.test <- data.frame("i"=test.dtm$i, "j"=test.dtm$j, "v"=test.dtm$v)
out <- cpp__nbTest(df.test, models)

preds <- unlist(lapply(out, function(x) as.integer(names(which(x == max(x)))[1])))
mean(preds == test.labels)


train.dtm <- dtm[sort(train1.idx),]
train.labels <- labels[sort(train1.idx)]

df.train <- data.frame("i"=train.dtm$i, "j"=train.dtm$j, "v"=train.dtm$v)
models <- cpp__nb(df.train, train.labels, nrow(train.dtm), ncol(train.dtm), lambda = 1)

df.test <- data.frame("i"=test.dtm$i, "j"=test.dtm$j, "v"=test.dtm$v)
out <- cpp__nbTest(df.test, models)

preds <- unlist(lapply(out, function(x) as.integer(names(which(x == max(x)))[1])))
mean(preds == test.labels)


