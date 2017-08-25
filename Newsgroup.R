getContentAndLabels <- function(folder.path) {
  dirs <- list.dirs(path = folder.path)[-1]
  
  file.contents <- c()
  cl.names <- c()
  
  for (dir in dirs) {
    files <- list.files(path = dir, full.names = T, recursive = T)
    
    contents <- lapply(files, function(x) stringr::str_trim(tolower(tm::stripWhitespace(gsub("\\b[a-zA-Z]{1,2}\\b", " ", 
                                                                                             gsub("[^a-zA-Z]", " ", 
                                                                                                  readLines(x)))))))
    
    file.contents <- c(file.contents, contents)
    
    cl.names <- c(cl.names, rep(gsub("^.*\\/(.*)$", "\\1", dir), length(files)))
  }
  
  list("Contents"=file.contents, "ClassNames"=cl.names)
}

train <- getContentAndLabels("/Users/funktor/Downloads/20news-bydate/20news-bydate-train")
test <- getContentAndLabels("/Users/funktor/Downloads/20news-bydate/20news-bydate-test")

contents <- c(train$Contents, test$Contents)
contents <- lapply(contents, function(x) x[x != ""])

unique.class.names <- unique(c(train$ClassNames, test$ClassNames))

train.labels <- as.integer(sapply(train$ClassNames, function(x) which(unique.class.names == x)))
test.labels <- as.integer(sapply(test$ClassNames, function(x) which(unique.class.names == x)))

labels <- c(train.labels, test.labels)
contents <- lapply(contents, function(x) paste(x, collapse = " "))

model <- cpp__generateGraph(contents, tm::stopwords("en"), contextSize = 5)
model$Words$Word <- as.character(model$Words$Word)

vocabulary <- readLines(con=file("/Users/funktor/Downloads/words.txt", open="r"))

q <- cpp__spellCorrect(model, contents, tm::stopwords("en"), vocabulary, contextSize = 5, similarCounts = 50, maxDepth = 1)