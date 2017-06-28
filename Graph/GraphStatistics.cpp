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
std::unordered_map<T, int> invertedIndex(std::vector<T> vec) {
  std::unordered_map<T, int> myOutput;
  
  for (unsigned int i = 0; i < vec.size(); i++) {
    myOutput[vec[i]] = i;
  }
  
  return myOutput;
}

size_t levenshteinDistance(const std::string &s1, const std::string &s2) {
  
  const size_t m(s1.size());
  const size_t n(s2.size());
  
  if ( m==0 ) return n;
  if ( n==0 ) return m;
  
  size_t *costs = new size_t[n + 1];
  
  for( size_t k=0; k<=n; k++ ) costs[k] = k;
  
  size_t i = 0;
  for ( std::string::const_iterator it1 = s1.begin(); it1 != s1.end(); ++it1, ++i ) {
    
    costs[0] = i+1;
    size_t corner = i;
    size_t j = 0;
    
    for ( std::string::const_iterator it2 = s2.begin(); it2 != s2.end(); ++it2, ++j ) {
      size_t upper = costs[j+1];
      
      if( *it1 == *it2 ) {
        costs[j+1] = corner;
      }
      else {
        size_t t = upper < corner ? upper : corner;
        costs[j+1] = (costs[j] < t ? costs[j] : t) + 1;
      }
      corner = upper;
    }
  }
  
  size_t result = costs[n];
  delete [] costs;
  
  return result;
}

struct TrainInstance {
  int givenWordIndex;
  std::set<int> contextWordsIndices;
};

std::vector<TrainInstance> generateSkipGramTrainingInstances(const std::vector<std::vector<std::string>> &documentWords, 
                                                             std::unordered_map<std::string, int> &wordIndex, const int &contextSize,
                                                             const std::vector<std::string> &vocabulary, const bool &onlyNonVocab) {
  
  std::vector<TrainInstance> output;
  std::set<std::string> vocab(vocabulary.begin(), vocabulary.end());
  
  for (int i = 0; i < documentWords.size(); i++) {
    if (documentWords[i].size() > 1) {
      std::deque<int> leftIndices, rightIndices;
      
      for (int k = 0; k < documentWords[i].size(); k++) {
        
        std::string trainWord = documentWords[i][k];
        int trainWordIndex = wordIndex[trainWord];
        
        if (k > contextSize) {
          leftIndices.pop_front();
          leftIndices.push_back(wordIndex[documentWords[i][k-1]]);
        }
        else if (k > 0) leftIndices.push_back(wordIndex[documentWords[i][k-1]]);
        
        int maxEl = std::min((double)contextSize, (double)documentWords[i].size()-1);
        
        if (k == 0) for (int m = 1; m <= maxEl; m++) rightIndices.push_back(wordIndex[documentWords[i][m]]);
        
        else if (k + contextSize < documentWords[i].size()) {
          rightIndices.pop_front();
          rightIndices.push_back(wordIndex[documentWords[i][k + contextSize]]);
        }
        else rightIndices.pop_front();
        
        std::set<int> contextIndices;
        
        for (auto p = leftIndices.begin(); p != leftIndices.end(); ++p) contextIndices.insert(*p);
        for (auto p = rightIndices.begin(); p != rightIndices.end(); ++p) contextIndices.insert(*p);
        
        if ((onlyNonVocab && vocab.find(trainWord) == vocab.end()) || !onlyNonVocab) {
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

std::unordered_map<std::string, int> getWordIndices(const std::vector<std::vector<std::string>> &documentWords) {
  std::set<std::string> wordSet;
  
  for (size_t i = 0; i < documentWords.size(); i++) wordSet.insert(documentWords[i].begin(), documentWords[i].end());
  
  std::vector<std::string> uniqueWords(wordSet.begin(), wordSet.end());
  std::unordered_map<std::string, int> wordIndex = invertedIndex(uniqueWords);
  
  return wordIndex;
}

std::unordered_map<int, std::unordered_map<int, double>> train(const std::vector<TrainInstance> &trainInstances) {
  std::unordered_map<int, std::unordered_map<int, double>> nodes;
  
  for (size_t i = 0; i < trainInstances.size(); i++) {
    TrainInstance instance = trainInstances[i];
    
    int currWordIndex = instance.givenWordIndex;
    std::set<int> contextWordsIndices = instance.contextWordsIndices;
    
    for (auto p = contextWordsIndices.begin(); p != contextWordsIndices.end(); ++p) nodes[*p][currWordIndex]++;
  }
  
  for (auto p = nodes.begin(); p != nodes.end();++p) {
    std::unordered_map<int, double> x = p->second;
    
    double sum = 0;
    for (auto q = x.begin(); q != x.end(); ++q) sum += q->second;
    for (auto q = x.begin(); q != x.end(); ++q) nodes[p->first][q->first] /= sum;
  }
  
  return nodes;
}

// [[Rcpp::export]]
List cpp__generateGraph(std::vector<std::vector<std::string>> documentWords, int contextSize=5) {
  
  std::unordered_map<std::string, int> wordIndex = getWordIndices(documentWords);
  
  std::vector<TrainInstance> trainInstances = generateSkipGramTrainingInstances(documentWords, wordIndex, 
                                                                                contextSize, std::vector<std::string>(), false);
  
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

struct comparator {
  bool operator()(const std::pair<int, double> &a, const std::pair<int, double> &b) {
    return a.second > b.second;
  }
};

struct NodeQueue {
  int nodeIdx, nodeDepth;
  double nodeScore;
};

std::unordered_map<int, double> mostContextual(std::unordered_map<int, std::unordered_map<int, double>> &nodes, 
                                               const int &index, const int &count, const int &maxDepth) {
  
  int currIdx = index;
  std::set<int> visitedNodes;
  
  std::priority_queue<std::pair<int, double>, std::vector<std::pair<int, double>>, comparator> minHeap;
  
  std::queue<NodeQueue> nodeQueue;

  nodeQueue.push({currIdx, 0, 1.0});
  
  while(!nodeQueue.empty()) {
    
    NodeQueue frontNode = nodeQueue.front();
    
    if (frontNode.nodeDepth >= maxDepth) break;
    
    visitedNodes.insert(frontNode.nodeIdx);
    
    std::unordered_map<int, double> neighbors = nodes[frontNode.nodeIdx];
    
    nodeQueue.pop();
    
    for (auto p = neighbors.begin(); p != neighbors.end(); ++p) {
      double score = frontNode.nodeScore*p->second;
      
      if (visitedNodes.find(p->first) == visitedNodes.end()) {
        
        if ((int)minHeap.size() >= count && minHeap.top().second < score) minHeap.pop();
        
        if (minHeap.empty() || (int)minHeap.size() < count) {
          nodeQueue.push({p->first, frontNode.nodeDepth+1, score});
          minHeap.push(std::make_pair(p->first, score));
        }
      }
    }
  }
  
  std::unordered_map<int, double> output;
  
  while(!minHeap.empty()) {
    std::pair<int, double> tops = minHeap.top();
    output[tops.first] = tops.second;
    minHeap.pop();
  }
  
  return output;
}

std::unordered_map<int, double> mostContextualMultiple(std::unordered_map<int, std::unordered_map<int, double>> &nodes, 
                                                       const std::set<int> &indices, const int &count, const int &maxDepth) {
  
  std::unordered_map<int, double> combined, output;
  std::vector<std::pair<int, double>> collect;
  
  for (auto p = indices.begin(); p != indices.end(); ++p) {
    std::unordered_map<int, double> contextual = mostContextual(nodes, *p, 1000, maxDepth);
    
    for (auto q = contextual.begin(); q != contextual.end(); ++q) {
      if (combined.find(q->first) != combined.end()) combined[q->first] += (1+q->second);
      else combined[q->first] = (1+q->second);
    }
  }
  
  std::priority_queue<std::pair<int, double>, std::vector<std::pair<int, double>>, comparator> minHeap;
  
  for (auto p = combined.begin(); p != combined.end(); ++p) {
    if ((int)minHeap.size() >= count && minHeap.top().second < p->second) minHeap.pop();
    if (minHeap.empty() || (int)minHeap.size() < count) minHeap.push(std::make_pair(p->first, p->second));
  }
  
  while(!minHeap.empty()) {
    std::pair<int, double> tops = minHeap.top();
    output[tops.first] = tops.second;
    minHeap.pop();
  }
  
  return output;
}

struct ModelData {
  std::unordered_map<int, std::unordered_map<int, double>> nodes;
  std::unordered_map<std::string, int> wordIndex;
  std::unordered_map<int, std::string> invertedWordIndex;
};

ModelData convertModelDataFrame(const List &model) {
  
  ModelData modelData;
  
  DataFrame wordsDF = as<DataFrame>(model["Words"]);
  
  std::vector<std::string> word = as<std::vector<std::string>>(wordsDF["Word"]);
  std::vector<int> index = as<std::vector<int>>(wordsDF["Index"]);
  
  for (size_t i = 0; i < word.size(); i++) {
    modelData.wordIndex[word[i]] = index[i];
    modelData.invertedWordIndex[index[i]] = word[i];
  }
  
  DataFrame nodesDF = as<DataFrame>(model["Nodes"]);
  
  std::vector<int> idx1 = as<std::vector<int>>(nodesDF["NodeA"]);
  std::vector<int> idx2 = as<std::vector<int>>(nodesDF["NodeB"]);
  std::vector<double> probs = as<std::vector<double>>(nodesDF["Prob"]);
  
  for (size_t i = 0; i < idx1.size(); i++) modelData.nodes[idx1[i]][idx2[i]] = probs[i];
  
  return modelData;
}

// [[Rcpp::export]]
std::unordered_map<std::string, double> cpp__getMostSimilar(List model, std::string target, 
                                                            int similarCounts=10, int maxDepth=1) {
  
  std::unordered_map<std::string, double> myOut;
  
  ModelData modelData = convertModelDataFrame(model);
  
  if (modelData.wordIndex.find(target) != modelData.wordIndex.end()) {
    std::unordered_map<int, double> out = mostContextual(modelData.nodes, modelData.wordIndex[target], similarCounts, maxDepth);
    
    for (auto p = out.begin(); p != out.end(); ++p) myOut[modelData.invertedWordIndex[p->first]] = p->second;
  }
  else myOut[""]=0.0;
  
  return myOut;
}

// [[Rcpp::export]]
std::unordered_map<std::string, double> cpp__getMostSimilarMultiple(List model, std::vector<std::string> targets, 
                                                                    int similarCounts=10, int maxDepth=1) {
  
  std::unordered_map<std::string, double> myOut;
  
  ModelData modelData = convertModelDataFrame(model);
  
  std::set<int> ids;
  for (size_t i = 0; i < targets.size(); i++) if (modelData.wordIndex.find(targets[i]) != modelData.wordIndex.end()) ids.insert(modelData.wordIndex[targets[i]]);
  
  if (ids.size() > 0) {
    std::unordered_map<int, double> out = mostContextualMultiple(modelData.nodes, ids, similarCounts, maxDepth);
    
    for (auto p = out.begin(); p != out.end(); ++p) myOut[modelData.invertedWordIndex[p->first]] = p->second;
  }
  else myOut[""]=0.0;
  
  return myOut;
}

// [[Rcpp::export]]
List cpp__spellCorrect(List model, std::vector<std::vector<std::string>> documentWords, 
                       std::vector<std::string> vocabulary, 
                       int contextSize=5, int similarCounts=50, int maxDepth=2) {
  
  ModelData modelData = convertModelDataFrame(model);
  
  std::vector<TrainInstance> nonVocab = generateSkipGramTrainingInstances(documentWords, modelData.wordIndex, 
                                                                          contextSize, vocabulary, true);
  
  std::unordered_map<std::string, std::string> cache;
  
  std::vector<std::string> incorrect, correct;
  std::set<std::string> vocab(vocabulary.begin(), vocabulary.end());
  
  for (size_t i = 0; i < nonVocab.size(); i++) {
    TrainInstance instance = nonVocab[i];
    
    std::string incorrectWord = modelData.invertedWordIndex[instance.givenWordIndex];
    std::string correctWord = incorrectWord;
    
    if (cache.find(incorrectWord) != cache.end()) correctWord = cache[incorrectWord];
    else {
      std::set<int> contextIds = instance.contextWordsIndices;
      
      if (contextIds.size() > 0) {
        std::unordered_map<int, double> out = mostContextualMultiple(modelData.nodes, contextIds, similarCounts, maxDepth);
        
        std::vector<std::pair<int, double>> temp;
        for (auto p = out.begin(); p != out.end(); ++p) temp.push_back(std::make_pair(p->first, p->second));
        
        std::sort(temp.begin(), temp.end(), [](const std::pair<int, double> &a, const std::pair<int, double> &b){return a.second > b.second;});
        
        int minDistance = 3;
        
        for (size_t j = 0; j < temp.size(); j++) {
          std::string possibleWord = modelData.invertedWordIndex[temp[j].first];
          
          int d = (int)levenshteinDistance(incorrectWord, possibleWord);
          
          if (d > 0 && d < minDistance && (double)d/(double)incorrectWord.size() <= 0.33 
                && (vocab.find(possibleWord) != vocab.end() 
                      || modelData.nodes[modelData.wordIndex[possibleWord]].size() > modelData.nodes[modelData.wordIndex[incorrectWord]].size())) {
                      
            minDistance = d;
            correctWord = possibleWord;
          }
        }
        
        cache[incorrectWord] = correctWord;
        
        if (correctWord != incorrectWord) {
          incorrect.push_back(incorrectWord);
          correct.push_back(correctWord);
        }
      }
    }
  }
  
  for (size_t i = 0; i < documentWords.size(); i++) {
    std::vector<std::string> words = documentWords[i];
    std::vector<std::string> updatedFileContent;
    
    for (size_t j = 0; j < words.size(); j++) {
      std::string word = words[j];
      
      if (cache.find(word) != cache.end()) words[j] = cache[word];
    }
    documentWords[i] = words;
  }
  
  return List::create(_["Corrections"]=DataFrame::create(_["Incorrect"]=incorrect, _["Correct"]=correct), _["UpdatedContents"]=documentWords);
}