source("../Newsgroup.R")

vectors <- cpp__generateWordVectors(contents, tm::stopwords("en"), contextSize = 5, negativeSamplesSize = 5, learningRate = 0.01)

similarity <- function(word1, word2, vectors) {
  u <- vectors$InputVectors[[as.character(vectors$Words[word1])]]
  v <- vectors$InputVectors[[as.character(vectors$Words[word2])]]
  
  sumsq <- function(u) sqrt(sum(u^2))
  
  abs(sum(u*v))/(sumsq(u)*sumsq(v))
}

similarity("computer", "graphics")
similarity("atheist", "religion")