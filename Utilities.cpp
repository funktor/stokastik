// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
#include <fstream>
#include <queue>
// [[Rcpp::depends(BH)]]
#include <boost/algorithm/string.hpp>
#include <string>
#include <algorithm>

using namespace Rcpp;

template<typename T>
std::map<T, int> classCounts(std::vector<T> classes) {
  std::map<T, int> counts;
  
  std::for_each(classes.begin(), classes.end(), [&counts](const T cl){counts[cl]++;});
  
  return counts;
}

template<typename T, typename F>
std::vector<F> myMap(std::vector<T> vec, F (*func)(T)) {
  std::vector<F> myOutput;
  
  std::for_each(vec.begin(), vec.end(),
                [&myOutput, &func](const T v){myOutput.push_back(func(v));});
  
  return myOutput;
}

template<typename T>
std::vector<T> myFilter(std::vector<T> vec, bool (*func)(T)) {
  std::vector<T> myOutput;
  
  std::for_each(vec.begin(), vec.end(),
                [&myOutput, &func](const T v){if (func(v)) {myOutput.push_back(func(v));}});
  
  return myOutput;
}

template<typename T>
std::map<T, int> invertedIndex(std::vector<T> vec) {
  std::map<T, int> myOutput;
  
  for (unsigned int i = 0; i < vec.size(); i++) {
    myOutput[vec[i]] = i+1;
  }
  
  return myOutput;
}

struct comparator {
  bool operator()(const std::pair<int, double> &a, const std::pair<int, double> &b) {
    return a.second > b.second;
  }
};

bool BothAreSpaces(char lhs, char rhs) { return (lhs == rhs) && (lhs == ' '); }

std::map<int, std::map<int, int>> featureClassCounts(std::vector<int> rows, std::vector<int> cols, std::vector<int> classes) {
  
  std::map<int, std::map<int, int>> myOutput;
  
  for (size_t i = 0; i < cols.size(); i++) {
    int featureIndex = cols[i];
    int cl = classes[rows[i]-1];
    
    myOutput[featureIndex][cl]++;
  }
  
  return myOutput;
}

std::set<int> entropyMeasures(std::vector<int> rows, std::vector<int> cols, std::vector<int> classes,
                              double (*func) (const double &, const double &, const double &, const double &),
                              const int &maxFeaturesPerClass) {
  
  std::set<int> myOutput;
  std::map<int, std::map<int, int>> fcCounts = featureClassCounts(rows, cols, classes);
  
  std::map<int, int> clCounts = classCounts(classes);
  std::map<int, int> fCounts = classCounts(cols);
  
  int totalDocs = 0;
  std::for_each(clCounts.begin(), clCounts.end(), [&totalDocs](std::pair<const int, int>& m){totalDocs += m.second;});
  
  std::map<int, std::priority_queue<std::pair<int, double>, std::vector<std::pair<int, double>>, comparator>> classMinHeaps;
  
  for (auto it = fCounts.begin(); it != fCounts.end(); ++it) {
    int featureIndex = it->first;
    int featureCount = it->second;
    
    std::map<int, int> p = fcCounts[featureIndex];
    
    
    for (auto it2 = p.begin(); it2 != p.end(); ++it2) {
      int cl = it2->first;
      int classWithFeatureCount = it2->second;
      
      int classCount = clCounts[cl];
      
      double value = func((double)classWithFeatureCount, (double)classCount, (double)featureCount, (double)totalDocs);
      
      if (classMinHeaps[cl].empty() || (int)classMinHeaps[cl].size() < maxFeaturesPerClass) classMinHeaps[cl].push(std::make_pair(featureIndex, value));
      
      else if (value > classMinHeaps[cl].top().second) {
        classMinHeaps[cl].pop();
        classMinHeaps[cl].push(std::make_pair(featureIndex, value));
      }
    }
  }
  
  for (auto it = classMinHeaps.begin(); it != classMinHeaps.end(); ++it) {
    std::priority_queue<std::pair<int, double>, std::vector<std::pair<int, double>>, comparator> q = it->second;
    
    while(!q.empty()) {
      std::pair<int, double> tops = q.top();
      myOutput.insert(tops.first);
      q.pop();
    }
  }
  
  return myOutput;
}


// [[Rcpp::export]]
std::set<int> cpp__mutualInformation(std::vector<int> rows, std::vector<int> cols,  
                                     std::vector<int> classes, int maxFeaturesPerClass) {
  return entropyMeasures(rows, cols, classes,
                         [](const double & classWithFeatureCount, const double & classCount, const double & featureCount, const double & totalDocs)
                         {
                           
                           double classProb = classCount/totalDocs;
                           double featureProb = featureCount/totalDocs;
                           
                           double classWithFeatureProb = classWithFeatureCount/totalDocs;
                           double noClassWithFeatureProb = (featureCount - classWithFeatureCount)/totalDocs;
                           double classWithoutFeatureProb = (classCount - classWithFeatureCount)/totalDocs;
                           double noClassWithoutFeatureProb = (totalDocs - featureCount - classCount + classWithFeatureCount)/totalDocs;
                           
                           return classWithFeatureProb*log2(classWithFeatureProb/(classProb*featureProb))
                             + ((noClassWithFeatureProb != 0) ? noClassWithFeatureProb*log2(noClassWithFeatureProb/((1-classProb)*featureProb)) : 0)  +
                               + ((classWithoutFeatureProb != 0) ? classWithoutFeatureProb*log2(classWithoutFeatureProb/(classProb*(1-featureProb))) : 0) +
                               + ((noClassWithoutFeatureProb != 0) ? noClassWithoutFeatureProb*log2(noClassWithoutFeatureProb/((1-classProb)*(1-featureProb))) : 0);
                         }, maxFeaturesPerClass);
}

/**
* Return mormal N-grams.
* (a, b, c, d) with 2-grams will return (ab, bc, cd)
*/

std::vector<std::string> getNgrams(std::vector<std::string> &strs, int minLength, int maxLength) {
  
  std::vector<std::string> out;
  std::map<int, std::vector<std::string>> ngrams;
  
  for (int len = 1; len <= maxLength; len++) {
    if (len == 1) ngrams[len].insert(ngrams[len].end(), strs.begin(), strs.end());
    else for (int i = len-1; i < (int) strs.size(); i++) ngrams[len].push_back(ngrams[len-1][i-len+1] + " " + strs[i]);
  }
  
  for (auto it = ngrams.begin(); it != ngrams.end(); ++it) 
    if (it->first >= minLength) out.insert(out.end(), (it->second).begin(), (it->second).end());
    
    return out;
}

/**
* Return all possible combinations of N-grams.
* (a, b, c, d) with 2-grams will return (ab, ac, ad, bc, bd, cd)
*/

std::vector<std::string> getNgramsAll(std::vector<std::string> &strs, int minLength, int maxLength) {
  
  std::vector<std::string> out;
  std::map<int, std::vector<std::string>> ngrams;
  
  std::map<int, int> cnt;
  
  for (int len = 1; len <= maxLength; len++) {
    if (len == 1) {
      ngrams[len].insert(ngrams[len].end(), strs.begin(), strs.end());
      for (int i = 1; i < (int)strs.size(); i++) cnt[i] = i;
    }
    else {
      std::vector<int> m;
      
      for (int i = len-1; i < (int) strs.size(); i++) {
        for (int j = 0; j < cnt[i]; j++) ngrams[len].push_back(ngrams[len-1][j] + " " + strs[i]);
        m.push_back((int)ngrams[len].size());
      }
      
      for (int i = len; i < (int)strs.size(); i++) cnt[i] = m[i-len];
    }
    
  }
  
  for (auto it = ngrams.begin(); it != ngrams.end(); ++it) if (it->first >= minLength) out.insert(out.end(), (it->second).begin(), (it->second).end());
  
  return out;
}

std::vector<std::string> ngramTokenize(std::vector<std::string> &words, int minNgram, int maxNgram, int allCombinations = 0) {
  
  std::vector<std::string> ngrams;
  
  if (allCombinations == 1) 
    ngrams = getNgramsAll(words, minNgram, std::min(maxNgram, (int)words.size()));
  else 
    ngrams = getNgrams(words, minNgram, std::min(maxNgram, (int)words.size()));
  
  return ngrams;
}

std::vector<std::string> getTitles(std::vector<std::string> &words) {
  
  std::vector<std::string> titles;
  std::vector<int> starts, ends;
  
  for (int j = 0; j < (int)words.size(); j++) if (std::isupper(words[j][0])) starts.push_back(j);
  
  if ((int)starts.size() > 1) {
    std::string str = "";
    
    for (int j = 1; j <= (int)starts.size(); j++) {
      if (j < (int)starts.size() && starts[j] == (starts[j-1]+1)) str += words[starts[j-1]] + "__";
      else if (str != ""){
        str += words[starts[j-1]];
        titles.push_back(str);
        str = "";
      }
    }
  }
  
  return titles;
}

std::string toLowerCase(std::string str) {
  boost::algorithm::to_lower(str);
  return str;
}

std::vector<std::string> removeStopWords(std::vector<std::string> &words, std::set<std::string> &stopWords) {
  
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


std::vector<std::string> stripWhiteSpaces(std::vector<std::string> &words) {
  std::vector<std::string> updatedWords;
  
  for (size_t i = 0; i < words.size(); i++) if (words[i] != "" && words[i] != " ") updatedWords.push_back(words[i]);
  
  return updatedWords;
}

std::vector<std::string> trimWords(std::vector<std::string> &words) {
  for (size_t i = 0; i < words.size(); i++) boost::trim(words[i]);
  
  return words;
}

/**
* For each file content, it returns the extracted features. Gets titles, removes
* stopwords, removes any unwanted whitespaces and constructs the N-grams for each file.
* 
* Takes input the file contents, the stopwords and the min and max ngram size.
*/

// [[Rcpp::export]]
std::vector<std::vector<std::vector<std::string>>> cpp__fileFeatures(std::vector<std::vector<std::string>> fileContents, 
                                                                     std::vector<std::string> excludeWords, 
                                                                     int minNgrams=1, int maxNgrams=2) {
  
  std::vector<std::vector<std::vector<std::string>>> fileFeatures;
  
  excludeWords = myMap(excludeWords, toLowerCase);
  
  std::set<std::string> stopWords(excludeWords.begin(), excludeWords.end());
  
  for (size_t i = 0; i < fileContents.size(); i++) {
    std::vector<std::string> contents = fileContents[i];
    
    std::vector<std::vector<std::string>> vec;
    
    int j = 0;
    
    while (j < (int)contents.size()) {
      
      std::vector<std::string> words = extractWords(contents[j]);
      //std::vector<std::string> titles = getTitles(words);
      
      //if ((int)titles.size() > 0) contents.insert(contents.end(), titles.begin(), titles.end());
      
      words = myMap(words, toLowerCase);
      
      if (!stopWords.empty()) words = removeStopWords(words, stopWords);
      
      words = ngramTokenize(words, minNgrams, maxNgrams);
      
      words = trimWords(words);
      words = stripWhiteSpaces(words);
      
      vec.push_back(words);
      j++;
    }
    
    fileFeatures.push_back(vec);
  }
  
  return fileFeatures;
}

/**
* Creates the DTM from the extracted features from each file. Returns the DTM in 
* sparse matrix format (i, j, v, docs, terms).
* 
* Takes as input features extracted per file
*/

// [[Rcpp::export]]
List cpp__createDTM(std::vector<std::vector<std::vector<std::string>>> fileFeatures) {
  
  std::vector<int> rows, cols, docs;
  std::vector<double> vals;
  std::set<std::string> features;
  
  std::map<int, std::map<std::string, double>> rowFeatureMap;
  
  for (size_t i = 0; i < fileFeatures.size(); i++) {
    std::vector<std::vector<std::string>> contents = fileFeatures[i];
    
    for (size_t j = 0; j < contents.size(); j++) {
      std::vector<std::string> words = contents[j];
      
      features.insert(words.begin(), words.end());
      
      for (size_t k = 0; k < words.size(); k++) rowFeatureMap[i+1][words[k]]++;
    }
  }
  
  std::vector<std::string> terms(features.begin(), features.end());
  
  std::map<std::string, int> featureIndices = invertedIndex(terms);
  
  int k = 1;
  for (auto p = rowFeatureMap.begin(); p != rowFeatureMap.end(); ++p) {
    
    std::map<std::string, double> fVMap = p->second;
    std::vector<int> indices;
    std::vector<double> v;
    
    docs.push_back(p->first);
    
    for (auto q = fVMap.begin(); q != fVMap.end(); ++q) {
      indices.push_back(featureIndices[q->first]);
      v.push_back(q->second);
    }
    
    for (size_t i = 0; i < indices.size(); i++) {
      rows.push_back(p->first);
      cols.push_back(indices[i]);
      vals.push_back(v[i]);
    }
    
    k++;
  }
  
  return List::create(_["i"]=rows, _["j"]=cols, _["v"]=vals, _["Terms"]=features, _["Docs"]=docs);
}

/**
* Word transition frequencies are normalized to obtain probabilities
*/

void getNormalizedProbs(std::unordered_map<std::string, std::unordered_map<std::string, double>> &inp) {
  for (auto p = inp.begin(); p != inp.end(); ++p) {
    std::unordered_map<std::string, double> x = p->second;
    
    double sum = 0;
    for (auto q = x.begin(); q != x.end(); ++q) sum += q->second;
    for (auto q = x.begin(); q != x.end(); ++q) inp[p->first][q->first]=(q->second)/sum;
  }
}

/**
* Compute word transition frequencies
*/

void wordTransitionFrequencies(const std::vector<std::vector<std::string>> &contents, 
                               std::unordered_map<std::string, std::unordered_map<std::string, double>> &forward, 
                               std::unordered_map<std::string, std::unordered_map<std::string, double>> &backward) {
  
  for (size_t i = 0; i < contents.size(); i++) {
    if ((int)contents[i].size() > 0) {
      std::vector<std::string> words = contents[i];
      
      words = myMap(words, toLowerCase);
      
      std::vector<int> textIdx;
      
      for (size_t j = 0; j < words.size(); j++) {
        std::string x = words[j];
        
        bool flag = true;
        for (size_t m = 0; m < x.size(); m++) if (!isalpha(x[m])) {flag=false;break;}
        if (flag) textIdx.push_back(j);
      }
      
      if ((int)textIdx.size() > 1) {
        for (int y = 0; y <= (int)textIdx.size(); y++) {
          
          std::string currWord, prevWord;
          
          if (y == (int)textIdx.size()) currWord = "__END__";
          else currWord = words[textIdx[y]];
          
          if (y == 0) prevWord = "__BEGIN__";
          else prevWord = words[textIdx[y-1]];
          
          forward[prevWord][currWord]++;
          backward[currWord][prevWord]++;
        }
      }
    }
  }
}

/**
* Levenshtein Distance
*/

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

/**
* Convert word transition probability maps into vector format for transporting into R
*/

void getVectorsFromMap(std::unordered_map<std::string, std::unordered_map<std::string, double>> &myMap, 
                       std::vector<std::string> &vec1, std::vector<std::string> &vec2, std::vector<double> &vec3) {
  
  for (auto p = myMap.begin(); p != myMap.end(); ++p) {
    std::unordered_map<std::string, double> x = p->second;
    
    for (auto q = x.begin(); q != x.end(); ++q) {
      vec1.push_back(p->first);
      vec2.push_back(q->first);
      vec3.push_back(q->second);
    }
  }
}

// [[Rcpp::export]]
List cpp__getWordTransitionProbabilities(std::vector<std::vector<std::vector<std::string>>> fileFeatures) {
  
  std::unordered_map<std::string, std::unordered_map<std::string, double>> wordTransFreqFwd, wordTransFreqBwd;
  
  for (size_t i = 0; i < fileFeatures.size(); i++) wordTransitionFrequencies(fileFeatures[i], wordTransFreqFwd, wordTransFreqBwd);
  
  getNormalizedProbs(wordTransFreqFwd);
  getNormalizedProbs(wordTransFreqBwd);
  
  std::vector<std::string> currWd1, currWd2, nextWd, prevWd;
  std::vector<double> freq1, freq2;
  
  getVectorsFromMap(wordTransFreqFwd, currWd1, nextWd, freq1);
  getVectorsFromMap(wordTransFreqBwd, currWd2, prevWd, freq2);
  
  return List::create(_["NEXT"]=List::create(_["x"]=currWd1, _["y"]=nextWd, _["v"]=freq1), 
                      _["PREV"]=List::create(_["x"]=currWd2, _["y"]=prevWd, _["v"]=freq2));
}

/**
* Compute the mean word transition probabilities from each word.
* meanP(A) = (P(A->W1) + P(A->W2) +...+ P(A->WN))/N
*/

double getMeanTransProb(std::unordered_map<std::string, double> &inp) {
  double sum = 0;
  for (auto p = inp.begin(); p != inp.end(); ++p) sum += p->second;
  
  return sum/(double)inp.size();
}

/**
* Compute the possible components of a string where blank spaces are mistakenly
* omitted. For e.g. "mortgagerider" is broken down to "mortgage rider"
*/

std::string spaceCorrectedWord(std::string incorrectWord, const std::set<std::string> &vocab, 
                               std::unordered_map<std::string, std::unordered_map<std::string, double>> &wordTransFreqFwd) {
  
  std::map<int, std::map<int, bool>> indicator;
  std::map<int, std::map<int, std::vector<std::string>>> components;
  
  for (int length = 1; length <= (int)incorrectWord.size(); length++) {
    for (int i = 0; i <= (int)incorrectWord.size()-length; i++) {
      int start = i;
      int end = start+length-1;
      
      indicator[start][end]=false;
      
      std::string subIncorrectWord=incorrectWord.substr(start, length);
      
      if (vocab.find(subIncorrectWord) != vocab.end()) {
        indicator[start][end]=true;
        components[start][end].push_back(subIncorrectWord);
      }
      else if (length > 1) {
        double maxFreq = std::numeric_limits<double>::min();
        int split = -1;
        
        for (int j = start; j < end; j++) {
          bool a = indicator[start][j];
          bool b = indicator[j+1][end];
          
          if (a && b) {
            std::string x = components[start][j].back();
            std::string y = components[j+1][end].front();
            
            bool c = wordTransFreqFwd.find(x) != wordTransFreqFwd.end() && wordTransFreqFwd[x].find(y) != wordTransFreqFwd[x].end();
            
            if (c) {
              indicator[start][end]=true;
              
              if (wordTransFreqFwd[x][y] >= maxFreq) {
                maxFreq = wordTransFreqFwd[x][y];
                split = j;
              }
            }  
          }
        }
        
        if (indicator[start][end]) {
          components[start][end].insert(components[start][end].end(), components[start][split].begin(), components[start][split].end());
          components[start][end].insert(components[start][end].end(), components[split+1][end].begin(), components[split+1][end].end());
        }
      }
    }
  }
  
  std::vector<std::string> allComponents = components[0][(int)incorrectWord.size()-1];
  
  std::string out = "";
  
  if ((int)allComponents.size() > 0) {
    for (int i = 0; i < (int)allComponents.size()-1; i++) out += allComponents[i] + " ";
    out += allComponents[(int)allComponents.size()-1];
  }
  else out = incorrectWord;
  
  return out;
}

// [[Rcpp::export]]
List cpp__spellCorrection(std::vector<std::vector<std::vector<std::string>>> fileFeatures, 
                          std::vector<std::string> currWd1, std::vector<std::string> currWd2, 
                          std::vector<std::string> nextWd, std::vector<std::string> prevWd, 
                          std::vector<double> freq1, std::vector<double> freq2,
                          std::vector<std::string> vocabulary) {
  
  vocabulary = myMap(vocabulary, toLowerCase);
  
  std::set<std::string> vocab(vocabulary.begin(), vocabulary.end());
  
  std::vector<std::string> wordA, wordB;
  std::unordered_map<std::string, std::unordered_map<std::string, int>> distanceCache;
  std::unordered_map<std::string, std::string> spaceCorrectedCache;
  
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>> correctedCache;
  
  std::unordered_map<std::string, std::unordered_map<std::string, double>> wordTransFreqFwd, wordTransFreqBwd;
  std::unordered_map<std::string, double> meanWordTransFreqFwd, meanWordTransFreqBwd;
  
  for (size_t i = 0; i < currWd1.size(); i++) wordTransFreqFwd[currWd1[i]][nextWd[i]] = freq1[i];
  for (size_t i = 0; i < currWd2.size(); i++) wordTransFreqBwd[currWd2[i]][prevWd[i]] = freq2[i];
  
  for (auto p = wordTransFreqFwd.begin(); p != wordTransFreqFwd.end(); ++p) meanWordTransFreqFwd[p->first] = getMeanTransProb(p->second);
  for (auto p = wordTransFreqBwd.begin(); p != wordTransFreqBwd.end(); ++p) meanWordTransFreqBwd[p->first] = getMeanTransProb(p->second);
  
  for (size_t i = 0; i < fileFeatures.size(); i++) {
    
    std::vector<std::vector<std::string>> contents = fileFeatures[i];
    
    for (size_t j = 0; j < contents.size(); j++) {
      if ((int)contents[j].size() > 0) {
        
        std::vector<std::string> words = contents[j];
        std::vector<int> textIdx;
        
        words = myMap(words, toLowerCase);
        
        for (size_t k = 0; k < words.size(); k++) {
          std::string x = words[k];
          
          bool flag = true;
          for (size_t m = 0; m < x.size(); m++) if (!isalpha(x[m])) {flag=false;break;}
          if (flag) textIdx.push_back(k);
        }
        
        if ((int)textIdx.size() > 1) {
          for (int y = 0; y < (int)textIdx.size(); y++) {
            
            std::string prevWord="", nextWord="", currWord=words[textIdx[y]];
            
            if ((int)currWord.size() <= 4 || vocab.find(currWord) != vocab.end()) continue;
            
            if (y == 0) prevWord = "__BEGIN__";
            else prevWord = words[textIdx[y-1]];
            
            if (correctedCache.find(prevWord) != correctedCache.end() 
                  && correctedCache[prevWord].find(currWord) != correctedCache[prevWord].end()) {
              
              wordA.push_back(currWord);
              wordB.push_back(correctedCache[prevWord][currWord]);
              words[textIdx[y]]=correctedCache[prevWord][currWord];
              
              continue;
            }
            
            std::string spaceCorrected = currWord;
            
            if (spaceCorrectedCache.find(currWord) != spaceCorrectedCache.end()) spaceCorrected = spaceCorrectedCache[currWord];
            else {
              spaceCorrected = spaceCorrectedWord(currWord, vocab, wordTransFreqFwd);
              spaceCorrectedCache[currWord] = spaceCorrected;
            }
            
            if (spaceCorrected != currWord) {
              wordA.push_back(currWord);
              wordB.push_back(spaceCorrected);
              
              correctedCache[prevWord][currWord] = spaceCorrected;
              words[textIdx[y]]=spaceCorrected;
              
              continue;
            }
            
            bool a = wordTransFreqFwd.find(prevWord) != wordTransFreqFwd.end() 
              && (wordTransFreqFwd[prevWord].find(currWord) == wordTransFreqFwd[prevWord].end() 
                    || wordTransFreqFwd[prevWord][currWord] < meanWordTransFreqFwd[prevWord]);
                    
                    if (y == (int)textIdx.size()-1) nextWord = "__END__";
                    else nextWord = words[textIdx[y+1]];
                    
                    bool b = wordTransFreqBwd.find(nextWord) != wordTransFreqBwd.end() 
                      && (wordTransFreqBwd[nextWord].find(currWord) == wordTransFreqBwd[nextWord].end() 
                            || wordTransFreqBwd[nextWord][currWord] < meanWordTransFreqBwd[nextWord]);
                            
                            if (a && b) {
                              std::unordered_map<std::string, double> prevWordsProbs = wordTransFreqBwd[nextWord];
                              std::unordered_map<std::string, double> nextWordsProbs = wordTransFreqFwd[prevWord];
                              
                              std::unordered_map<std::string, double> commWordsProbs;
                              
                              for (auto p = prevWordsProbs.begin(); p != prevWordsProbs.end(); ++p) {
                                if (nextWordsProbs.find(p->first) != nextWordsProbs.end()) commWordsProbs[p->first]=(p->second+nextWordsProbs[p->first])/2;
                              }
                              
                              double thres = std::min(meanWordTransFreqFwd[prevWord], meanWordTransFreqBwd[nextWord]);
                              double maxFreq = thres;
                              std::string correctedWord = "";
                              
                              for (auto p = commWordsProbs.begin(); p != commWordsProbs.end(); ++p) {
                                std::string possibleWord = p->first;
                                double frequency = p->second;
                                
                                if (frequency > maxFreq) {
                                  int d;
                                  
                                  if (distanceCache.find(currWord) != distanceCache.end() 
                                        && distanceCache[currWord].find(possibleWord) != distanceCache[currWord].end()) {
                                    d = distanceCache[currWord][possibleWord];
                                  }
                                  else {
                                    d = (int)levenshteinDistance(currWord, possibleWord);
                                    distanceCache[currWord][possibleWord] = d;
                                  }
                                  
                                  if (d == 1 || d == 2) {
                                    maxFreq = frequency;
                                    correctedWord = possibleWord;
                                  }
                                }
                              }
                              
                              if (maxFreq > thres) {
                                spaceCorrected = correctedWord;
                                
                                if (spaceCorrectedCache.find(correctedWord) != spaceCorrectedCache.end()) spaceCorrected = spaceCorrectedCache[correctedWord];
                                else {
                                  spaceCorrected = spaceCorrectedWord(correctedWord, vocab, wordTransFreqFwd);
                                  spaceCorrectedCache[correctedWord] = spaceCorrected;
                                }
                                
                                wordA.push_back(currWord);
                                wordB.push_back(spaceCorrected);
                                
                                correctedCache[prevWord][currWord] = spaceCorrected;
                                words[textIdx[y]]=spaceCorrected;
                              }
                            }           
          }
        }
        
        fileFeatures[i][j] = words;
      }
    }
  }
  
  
  
  return List::create(_["corrections"]=DataFrame::create(_["x"]=wordA, _["y"]=wordB), _["features"]=fileFeatures);
}

// [[Rcpp::export]]
DataFrame cpp__appendTestDocs(std::vector<int> trainRows, std::vector<int> trainCols, std::vector<int> trainVals, 
                              std::vector<int> testRows, std::vector<int> testCols, std::vector<int> testVals, int trainSize) {
  
  std::vector<int> outRows, outCols, outVals;
  
  outRows = trainRows;
  outCols = trainCols;
  outVals = trainVals;
  
  for (size_t i = 0; i < testRows.size(); i++) {
    outRows.push_back(trainSize+testRows[i]);
    outCols.push_back(testCols[i]);
    outVals.push_back(testVals[i]);
  }
  
  return DataFrame::create(_["i"]=outRows, _["j"]=outCols, _["v"]=outVals);
}


// [[Rcpp::export]]
List cpp__appendSTM(std::vector<int> trainRows, std::vector<int> trainCols, std::vector<int> trainVals, std::vector<std::string> trainTerms,
                    std::vector<int> testRows, std::vector<int> testCols, std::vector<int> testVals, std::vector<std::string> testTerms){
  
  std::vector<int> outRows, outCols, outVals;
  std::vector<std::string> outTerms;
  const int ONE = 1;
  
  outRows = trainRows;
  outVals = trainVals;
  
  auto result = std::max_element(trainRows.begin(), trainRows.end());
  const int curSize = *result;
  
  for (auto i : testRows){
    outRows.push_back(curSize + i);
    outVals.push_back(ONE);
  }
  
  
  std::set_union(trainTerms.begin(), trainTerms.end(), testTerms.begin(), testTerms.end(), std::back_inserter(outTerms));
  std::map<std::string, int> featureToIndexMap = invertedIndex(outTerms);
  for (const auto &j: trainCols){
    std::string term = trainTerms[j-1];
    int index = featureToIndexMap[term];
    outCols.push_back(index);
  }
  
  for (auto k: testCols){
    std::string term = testTerms[k-1];
    int index = featureToIndexMap[term];
    outCols.push_back(index);
  }
  
  return List::create(_["i"]=outRows, _["j"]=outCols, _["v"]=outVals, _["terms"]=outTerms);
  
}