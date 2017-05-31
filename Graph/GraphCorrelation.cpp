// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
#include <fstream>
#include <queue>
#include <stack>
// [[Rcpp::depends(BH)]]
#include <boost/algorithm/string.hpp>
#include <random>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/discrete_distribution.hpp>

using namespace Rcpp;

template<typename T, typename F>
std::vector<F> myMap(std::vector<T> vec, F (*func)(T)) {
  std::vector<F> myOutput;
  
  std::for_each(vec.begin(), vec.end(),
                [&myOutput, &func](const T v){myOutput.push_back(func(v));});
  
  return myOutput;
}

template<typename T>
std::map<T, int> invertedIndex(std::vector<T> vec) {
  std::map<T, int> myOutput;
  
  for (unsigned int i = 0; i < vec.size(); i++) {
    myOutput[vec[i]] = i;
  }
  
  return myOutput;
}

bool BothAreSpaces(char lhs, char rhs) { return (lhs == rhs) && (lhs == ' '); }

std::string toLowerCase(std::string str) {
  boost::algorithm::to_lower(str);
  return str;
}

std::vector<std::string> removeStopWords(const std::vector<std::string> &words, const std::set<std::string> &stopWords) {
  
  std::vector<std::string> updatedWords;
  for (size_t j = 0; j < words.size(); j++) if (stopWords.find(words[j]) == stopWords.end()) updatedWords.push_back(words[j]);
  
  return updatedWords;
}

std::vector<std::string> extractWords(std::string &str) {
  std::string::iterator new_end = std::unique(str.begin(), str.end(), BothAreSpaces);
  str.erase(new_end, str.end());
  
  std::string delimiters(" \r\n\t`~!@#$%^&*()-+=[]{}|\\;:'\",<.>/?");
  std::vector<std::string> words;
  boost::split(words, str, boost::is_any_of(delimiters));
  
  return words;
}


std::vector<std::string> stripWhiteSpaces(const std::vector<std::string> &words) {
  std::vector<std::string> updatedWords;
  for (size_t i = 0; i < words.size(); i++) if (words[i] != "" && words[i] != " ") updatedWords.push_back(words[i]);
  
  return updatedWords;
}

std::vector<std::string> trimWords(std::vector<std::string> &words) {
  for (size_t i = 0; i < words.size(); i++) boost::trim(words[i]);
  
  return words;
}

std::vector<std::vector<std::vector<std::string>>> tokenize(const std::vector<std::vector<std::string>> &textContents, 
                                                            std::vector<std::string> &excludeWords) {
  
  std::vector<std::vector<std::vector<std::string>>> fileFeatures;
  
  excludeWords = myMap(excludeWords, toLowerCase);
  std::set<std::string> stopWords(excludeWords.begin(), excludeWords.end());
  
  for (size_t i = 0; i < textContents.size(); i++) {
    
    std::vector<std::string> contents = textContents[i];
    std::vector<std::vector<std::string>> vec;
    
    int j = 0;
    
    while (j < (int)contents.size()) {
      
      std::vector<std::string> words = extractWords(contents[j]);
      
      words = myMap(words, toLowerCase);
      
      if (!stopWords.empty()) words = removeStopWords(words, stopWords);
      
      words = trimWords(words);
      words = stripWhiteSpaces(words);
      
      vec.push_back(words);
      j++;
    }
    fileFeatures.push_back(vec);
  }
  
  return fileFeatures;
}

struct TrainInstance {
  int givenWordIndex;
  std::set<int> contextWordsIndices;
};

std::vector<TrainInstance> generateSkipGramTrainingInstances(const std::vector<std::vector<std::vector<std::string>>> &tokenizedWords, 
                                                             std::map<std::string, int> &wordIndex,
                                                             const int &contextSize) {
  
  std::vector<TrainInstance> output;
  
  for (int i = 0; i < tokenizedWords.size(); i++) {
    for (int j = 0; j < tokenizedWords[i].size(); j++) {
      if (tokenizedWords[i][j].size() > 1) {
        std::deque<int> leftIndices, rightIndices;
        
        for (int k = 0; k < tokenizedWords[i][j].size(); k++) {
          
          std::string trainWord = tokenizedWords[i][j][k];
          int trainWordIndex = wordIndex[trainWord];
          
          if (k > contextSize) {
            leftIndices.pop_front();
            leftIndices.push_back(wordIndex[tokenizedWords[i][j][k-1]]);
          }
          else if (k > 0) leftIndices.push_back(wordIndex[tokenizedWords[i][j][k-1]]);
          
          int maxEl = std::min((double)contextSize, (double)tokenizedWords[i][j].size()-1);
          
          if (k == 0) for (int m = 1; m <= maxEl; m++) rightIndices.push_back(wordIndex[tokenizedWords[i][j][m]]);
          
          else if (k + contextSize < tokenizedWords[i][j].size()) {
            rightIndices.pop_front();
            rightIndices.push_back(wordIndex[tokenizedWords[i][j][k + contextSize]]);
          }
          else rightIndices.pop_front();
          
          std::set<int> contextIndices;
          
          for (auto p = leftIndices.begin(); p != leftIndices.end(); ++p) contextIndices.insert(*p);
          for (auto p = rightIndices.begin(); p != rightIndices.end(); ++p) contextIndices.insert(*p);
          
          TrainInstance inst;
          inst.givenWordIndex = trainWordIndex;
          inst.contextWordsIndices = contextIndices;
          
          output.push_back(inst);
        }
      }
    }
  }
  
  return output;
}

std::map<std::string, int> getWordIndices(const std::vector<std::vector<std::vector<std::string>>> &tokenizedWords) {
  std::set<std::string> wordSet;
  
  for (size_t i = 0; i < tokenizedWords.size(); i++) {
    for (size_t j = 0; j < tokenizedWords[i].size(); j++) wordSet.insert(tokenizedWords[i][j].begin(), tokenizedWords[i][j].end());
  }
  
  std::vector<std::string> uniqueWords(wordSet.begin(), wordSet.end());
  std::map<std::string, int> wordIndex = invertedIndex(uniqueWords);
  
  return wordIndex;
}

std::unordered_map<int, std::unordered_map<int, double>> train(const std::vector<TrainInstance> &trainInstances) {
  std::unordered_map<int, std::unordered_map<int, double>> nodes;
  
  for (size_t i = 0; i < trainInstances.size(); i++) {
    TrainInstance instance = trainInstances[i];
    
    int currWordIndex = instance.givenWordIndex;
    std::set<int> contextWordsIndices = instance.contextWordsIndices;
    
    for (auto p = contextWordsIndices.begin(); p != contextWordsIndices.end(); ++p) {
      nodes[currWordIndex][*p]++;
      nodes[*p][currWordIndex]++;
    }
  }
  
  std::unordered_map<int, double> neighborsSum;
  
  for (auto p = nodes.begin(); p != nodes.end();++p) {
    std::unordered_map<int, double> x = p->second;
    
    double sum = 0;
    for (auto q = x.begin(); q != x.end(); ++q) sum += q->second;
    neighborsSum[p->first] = sum;
  }
  
  for (auto p = nodes.begin(); p != nodes.end();++p) {
    std::unordered_map<int, double> x = p->second;
    
    for (auto q = x.begin(); q != x.end(); ++q) nodes[p->first][q->first] /= (neighborsSum[p->first]+neighborsSum[q->first]-nodes[p->first][q->first]);
  }
  
  return nodes;
}

// [[Rcpp::export]]
List cpp__generateGraph(std::vector<std::vector<std::string>> textContents, 
                        std::vector<std::string> excludeWords, int contextSize=5) {
  
  std::vector<std::vector<std::vector<std::string>>> tokenizedWords = tokenize(textContents, excludeWords);
  
  std::map<std::string, int> wordIndex = getWordIndices(tokenizedWords);

  std::vector<TrainInstance> trainInstances = generateSkipGramTrainingInstances(tokenizedWords, wordIndex, contextSize);
  
  std::unordered_map<int, std::unordered_map<int, double>> nodes = train(trainInstances);
  
  std::vector<int> fromNodeIdx, toNodeIdx;
  std::vector<double> probs;
  
  for (auto p = nodes.begin(); p != nodes.end(); ++p) {
    std::unordered_map<int, double> x = p->second;
    
    for (auto q = x.begin(); q != x.end(); ++q) {
      fromNodeIdx.push_back(p->first);
      toNodeIdx.push_back(q->first);
      probs.push_back(q->second);
    }
  }
  
  std::vector<std::string> words;
  std::vector<int> indices;
  
  for (auto p = wordIndex.begin(); p != wordIndex.end(); ++p) {
    words.push_back(p->first);
    indices.push_back(p->second);
  }
  
  return List::create(_["Words"]=DataFrame::create(_["Word"]=words, _["Index"]=indices), 
                      _["Nodes"]=DataFrame::create(_["NodeA"]=fromNodeIdx, _["NodeB"]=toNodeIdx, _["Prob"]=probs));
}

double bfsSearch(std::unordered_map<int, std::unordered_map<int, double>> &nodes, 
                 const int &index1, const int &index2) {
  
  std::set<int> visitedNodes;
  
  std::queue<int> nodeQueue;
  std::queue<double> scoreQueue;
  
  int currIdx = index1;
  
  nodeQueue.push(currIdx);
  scoreQueue.push(1.0);
  
  double maxScore = 0;
  
  while(!nodeQueue.empty()) {
    int frontIdx = nodeQueue.front();
    double frontScore = scoreQueue.front();
    
    visitedNodes.insert(frontIdx);
    
    std::unordered_map<int, double> neighbors = nodes[frontIdx];
    
    nodeQueue.pop();
    scoreQueue.pop();
    
    for (auto p = neighbors.begin(); p != neighbors.end(); ++p) {
      double score = frontScore*p->second;
      
      if (p->first == index2 && score > maxScore) maxScore = score;
      
      if (p->first != index2 && visitedNodes.find(p->first) == visitedNodes.end() && score > maxScore) {
        nodeQueue.push(p->first);
        scoreQueue.push(score);
      }
    }
  }
  
  return maxScore;
}

struct comparator {
  bool operator()(const std::pair<int, double> &a, const std::pair<int, double> &b) {
    return a.second > b.second;
  }
};

std::map<int, double> mostContextual(std::unordered_map<int, std::unordered_map<int, double>> &nodes, 
                                     const int &index, const int &count, const int &maxDepth) {
  
  std::set<int> visitedNodes;
  int currIdx = index;
  
  std::priority_queue<std::pair<int, double>, std::vector<std::pair<int, double>>, comparator> minHeap;

  std::queue<int> nodeQueue, depthQueue;
  std::queue<double> scoreQueue;

  nodeQueue.push(currIdx);
  scoreQueue.push(1.0);
  depthQueue.push(0);

  while(!nodeQueue.empty()) {
    
    int frontIdx = nodeQueue.front();
    int frontDepth = depthQueue.front();
    double frontScore = scoreQueue.front();
    
    visitedNodes.insert(frontIdx);
    
    std::unordered_map<int, double> neighbors = nodes[frontIdx];
    
    nodeQueue.pop();
    scoreQueue.pop();
    depthQueue.pop();
    
    for (auto p = neighbors.begin(); p != neighbors.end(); ++p) {
      double score = frontScore*p->second;

      if (visitedNodes.find(p->first) == visitedNodes.end() && frontDepth < maxDepth) {
        
        if ((int)minHeap.size() >= count && minHeap.top().second < score) minHeap.pop();
        
        if (minHeap.empty() || (int)minHeap.size() < count) {
          nodeQueue.push(p->first);
          scoreQueue.push(score);
          depthQueue.push(frontDepth+1);
          
          minHeap.push(std::make_pair(p->first, score));
        }
      }
    }
  }
  
  std::map<int, double> output;
  
  while(!minHeap.empty()) {
    std::pair<int, double> tops = minHeap.top();
    output[tops.first] = tops.second;
    minHeap.pop();
  }
  
  return output;
}

std::map<int, double> mostContextualMultiple(std::unordered_map<int, std::unordered_map<int, double>> &nodes, 
                                             const std::set<int> &indices, const int &count, const int &maxDepth) {
  
  std::map<int, double> combined, output;
  std::vector<std::pair<int, double>> collect;
  
  for (auto p = indices.begin(); p != indices.end(); ++p) {
    std::map<int, double> contextual = mostContextual(nodes, *p, 1000, maxDepth);
    
    for (auto q = contextual.begin(); q != contextual.end(); ++q) {
      if (combined.find(q->first) != combined.end()) combined[q->first] += (1+q->second);
      else combined[q->first] = (1+q->second);
    }
  }
  
  for (auto p = combined.begin(); p != combined.end(); ++p) collect.push_back(std::make_pair(p->first, p->second));
  
  std::sort(collect.begin(), collect.end(), [](const std::pair<int, double> &a, const std::pair<int, double> &b){return a.second > b.second;});
  
  std::vector<std::pair<int, double>> subCollect(collect.begin(), collect.begin()+count);
  
  for (size_t i = 0; i < subCollect.size(); i++) output[subCollect[i].first] = subCollect[i].second;
  
  return output;
}

void convertModelDataFrame(List model, std::unordered_map<int, std::unordered_map<int, double>> &nodes, 
                           std::map<std::string, int> &wordIndex,
                           std::map<int, std::string> &invertedWordIndex) {
  
  DataFrame wordsDF = as<DataFrame>(model["Words"]);
  
  std::vector<std::string> word = as<std::vector<std::string>>(wordsDF["Word"]);
  std::vector<int> index = as<std::vector<int>>(wordsDF["Index"]);
  
  for (size_t i = 0; i < word.size(); i++) {
    wordIndex[word[i]] = index[i];
    invertedWordIndex[index[i]] = word[i];
  }
  
  DataFrame nodesDF = as<DataFrame>(model["Nodes"]);
  
  std::vector<int> idx1 = as<std::vector<int>>(nodesDF["NodeA"]);
  std::vector<int> idx2 = as<std::vector<int>>(nodesDF["NodeB"]);
  std::vector<double> probs = as<std::vector<double>>(nodesDF["Prob"]);
  
  for (size_t i = 0; i < idx1.size(); i++) nodes[idx1[i]][idx2[i]] = probs[i];
}

// [[Rcpp::export]]
double cpp__computeSimilarity(List model, std::string word1, std::string word2) {
  
  std::unordered_map<int, std::unordered_map<int, double>> nodes;
  std::map<std::string, int> wordIndex;
  std::map<int, std::string> invertedWordIndex;
  
  convertModelDataFrame(model, nodes, wordIndex, invertedWordIndex);
  
  return bfsSearch(nodes, wordIndex[word1], wordIndex[word2]);
}

// [[Rcpp::export]]
std::map<std::string, double> cpp__getMostSimilar(List model, std::string target, 
                                                  int similarCounts=10, int maxDepth=1) {
  
  std::map<std::string, double> myOut;
  
  std::unordered_map<int, std::unordered_map<int, double>> nodes;
  std::map<std::string, int> wordIndex;
  std::map<int, std::string> invertedWordIndex;
  
  convertModelDataFrame(model, nodes, wordIndex, invertedWordIndex);
  
  std::map<int, double> out = mostContextual(nodes, wordIndex[target], similarCounts, maxDepth);

  for (auto p = out.begin(); p != out.end(); ++p) myOut[invertedWordIndex[p->first]] = p->second;
  
  return myOut;
}

// [[Rcpp::export]]
std::map<std::string, double> cpp__getMostSimilarMultiple(List model, std::vector<std::string> targets, 
                                                          int similarCounts=10, int maxDepth=1) {
  
  std::map<std::string, double> myOut;
  
  std::unordered_map<int, std::unordered_map<int, double>> nodes;
  std::map<std::string, int> wordIndex;
  std::map<int, std::string> invertedWordIndex;
  
  convertModelDataFrame(model, nodes, wordIndex, invertedWordIndex);
  
  std::set<int> ids;
  for (size_t i = 0; i < targets.size(); i++) ids.insert(wordIndex[targets[i]]);
  
  std::map<int, double> out = mostContextualMultiple(nodes, ids, similarCounts, maxDepth);
  
  for (auto p = out.begin(); p != out.end(); ++p) myOut[invertedWordIndex[p->first]] = p->second;
  
  return myOut;
}