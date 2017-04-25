// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
#include <fstream>
#include <string>
#include <algorithm>

using namespace Rcpp;

typedef std::unordered_map<int, std::unordered_map<int, double>> Int_Int_Double;
typedef std::unordered_map<int, double> Int_Double;
typedef std::unordered_map<int, int> Int_Int;


Int_Int_Double computePosteriorProbabilities(Int_Int_Double &priorClassFeatureCounts, Int_Double &priorClassCounts, 
                                             Int_Int_Double &rowColValMap, const std::set<int> &classLabels,
                                             const double &numDocs, const double &numFeatures) {
  
  Int_Int_Double posteriorProbs;
  
  Int_Double classFeatureProbsSum;
  
  int numClasses = (int)classLabels.size();
  
  for (auto p = priorClassFeatureCounts.begin(); p != priorClassFeatureCounts.end(); ++p) {
    Int_Double x = p->second;
    for (auto q = x.begin(); q != x.end(); ++q) classFeatureProbsSum[p->first] += q->second;
  }
  
  for (auto p = rowColValMap.begin(); p != rowColValMap.end(); ++p) {
    Int_Double colValMap = p->second;
    
    for (auto q = classLabels.begin(); q != classLabels.end(); ++q) {
      
      double classProb = (1+priorClassCounts[*q])/((double)numClasses + (double)numDocs);
      posteriorProbs[p->first][*q] += log(classProb);
      
      double constant = (double)classFeatureProbsSum[*q] + (double)numFeatures;
      
      double featureProb = 0;
      
      for (auto r = colValMap.begin(); r != colValMap.end(); ++r) {
        if (priorClassFeatureCounts[*q].find(r->first) != priorClassFeatureCounts[*q].end()) 
          featureProb += log(1+priorClassFeatureCounts[*q][r->first])-log(constant);
        
        else featureProb -= log(constant);
      }
      
      posteriorProbs[p->first][*q] += featureProb;
    }
    
    Int_Double x = posteriorProbs[p->first];
    
    double maxVal = -std::numeric_limits<double>::max();
    
    for (auto q = x.begin(); q != x.end(); ++q) if (q->second > maxVal) maxVal = q->second;
    
    for (auto q = x.begin(); q != x.end(); ++q) posteriorProbs[p->first][q->first] = exp((q->second)-maxVal);
  }
  
  for (auto p = posteriorProbs.begin(); p != posteriorProbs.end(); ++p) {
    Int_Double x = p->second;

    double sum = 0;
    for (auto q = x.begin(); q != x.end(); ++q) sum += q->second;

    for (auto q = x.begin(); q != x.end(); ++q) posteriorProbs[p->first][q->first] = (q->second)/sum;
  }
  
  return posteriorProbs;
}

Int_Int_Double computePriorClassFeatureCounts(Int_Int_Double &rowClassProbs, Int_Int_Double &rowColValMap,
                                              Int_Int &rowClassMap, const double &lambda) {
  
  Int_Int_Double priorCounts;
  
  for (auto p = rowColValMap.begin(); p != rowColValMap.end(); ++p) {
    int row = p->first;
    
    Int_Double colValMap = p->second;
    Int_Double classProbs = rowClassProbs[row];
    
    for (auto q = classProbs.begin(); q != classProbs.end(); ++q) {
      for (auto r = colValMap.begin(); r != colValMap.end(); ++r) {
        if (rowClassMap[row] == -1) priorCounts[q->first][r->first] += lambda*(r->second)*(q->second);
        else priorCounts[q->first][r->first] += (r->second)*(q->second);
      }
    }
  }
  
  return priorCounts;
}

Int_Double computeClassCounts(Int_Int_Double &rowClassProbs, Int_Int &rowClassMap, 
                              const double &lambda) {
  
  Int_Double classPriorCounts;
  
  for (auto p = rowClassProbs.begin(); p != rowClassProbs.end(); ++p) {
    Int_Double classProbs = p->second;
    
    for (auto q = classProbs.begin(); q != classProbs.end(); ++q) {
      if (rowClassMap[p->first] == -1) classPriorCounts[q->first] += lambda*(q->second);
      else classPriorCounts[q->first] += (q->second);
    }
  }
  
  return classPriorCounts;
}

// [[Rcpp::export]]
List cpp__nb(DataFrame inputSparseMatrix, std::vector<int> classLabels, 
             int numDocs, int numFeatures, double lambda=0.5, int maxIter=5) {
  
  std::vector<int> rows = inputSparseMatrix["i"];
  std::vector<int> cols = inputSparseMatrix["j"];
  std::vector<double> vals = inputSparseMatrix["v"];
  
  std::set<int> uniqueRows(rows.begin(), rows.end());
  std::set<int> uniqueLabels(classLabels.begin(), classLabels.end());
  
  if (uniqueLabels.find(-1) != uniqueLabels.end()) uniqueLabels.erase(uniqueLabels.find(-1));
  
  Int_Int_Double rowColValMap, unlabelledDocsRowColValMap, labelledDocsRowColValMap;
  for (size_t i = 0; i < rows.size(); i++) rowColValMap[rows[i]][cols[i]] = vals[i];

  Int_Int rowClassMap;
  for (auto p = uniqueRows.begin(); p != uniqueRows.end(); ++p) rowClassMap[*p] = classLabels[*p-1];
  
  for (auto p = rowColValMap.begin(); p != rowColValMap.end(); ++p) {
    if (rowClassMap[p->first] == -1) unlabelledDocsRowColValMap[p->first] = p->second;
    else labelledDocsRowColValMap[p->first] = p->second;
  }
  

  Int_Int_Double rowClassProbs;

  for (auto p = uniqueRows.begin(); p != uniqueRows.end(); ++p) if (rowClassMap[*p] != -1) rowClassProbs[*p][rowClassMap[*p]] = 1;

  Int_Double priorClassCounts = computeClassCounts(rowClassProbs, rowClassMap, lambda);
  Int_Int_Double priorClassFeatureCounts = computePriorClassFeatureCounts(rowClassProbs, rowColValMap, rowClassMap, lambda);

  if (unlabelledDocsRowColValMap.size() > 0) {
    
    double counter = 0;
    
    while(counter < maxIter) {
      
      Int_Int_Double unlabelledDocsRowClassProbs = computePosteriorProbabilities(priorClassFeatureCounts, priorClassCounts, 
                                                                                 unlabelledDocsRowColValMap, uniqueLabels,
                                                                                 (double)numDocs, (double)numFeatures);
      
      for (auto p = uniqueRows.begin(); p != uniqueRows.end(); ++p) if (rowClassMap[*p] == -1) rowClassProbs[*p] = unlabelledDocsRowClassProbs[*p];
      
      priorClassCounts = computeClassCounts(rowClassProbs, rowClassMap, lambda);
      priorClassFeatureCounts = computePriorClassFeatureCounts(rowClassProbs, rowColValMap, rowClassMap, lambda);
      
      counter++;
    }
  }
  
  
  std::vector<int> clLabels1, clLabels2, featureLabels2;
  std::vector<double> clCounts1, clFeatureCounts2;
  
  for (auto p = priorClassCounts.begin(); p != priorClassCounts.end(); ++p) {
    clLabels1.push_back(p->first);
    clCounts1.push_back(p->second);
  }
  
  for (auto p = priorClassFeatureCounts.begin(); p != priorClassFeatureCounts.end(); ++p) {
    Int_Double x = p->second;
    for (auto q = x.begin(); q != x.end(); ++q) {
      clLabels2.push_back(p->first);
      featureLabels2.push_back(q->first);
      clFeatureCounts2.push_back(q->second);
    }
  }
  
  return List::create(_["ClassProbs"]=DataFrame::create(_["Class"]=clLabels1, _["Count"]=clCounts1),
                      _["ClassFeatureProbs"]=DataFrame::create(_["Class"]=clLabels2, 
                                              _["Feature"]=featureLabels2, 
                                              _["Count"]=clFeatureCounts2));
}

// [[Rcpp::export]]
Int_Int_Double cpp__nbTest(DataFrame inputSparseMatrix, List model) {
  
  std::vector<int> rows = inputSparseMatrix["i"];
  std::vector<int> cols = inputSparseMatrix["j"];
  std::vector<double> vals = inputSparseMatrix["v"];
  
  DataFrame clCounts = as<DataFrame>(model["ClassProbs"]);
  DataFrame clFeatureCounts = as<DataFrame>(model["ClassFeatureProbs"]);
  
  Int_Int_Double rowColValMap;
  for (size_t i = 0; i < rows.size(); i++) rowColValMap[rows[i]][cols[i]] = vals[i];
  
  Int_Double priorClassCounts;
  Int_Int_Double priorClassFeatureCounts;
  
  std::vector<int> clLabels1 = as<std::vector<int>>(clCounts["Class"]);
  std::vector<double> clCounts1 = as<std::vector<double>>(clCounts["Count"]);
  
  std::set<int> uniqueLabels(clLabels1.begin(), clLabels1.end());
  
  std::vector<int> clLabels2 = as<std::vector<int>>(clFeatureCounts["Class"]);
  std::vector<int> featureLabels2 = as<std::vector<int>>(clFeatureCounts["Feature"]);
  std::vector<double> clFeatureCounts2 = as<std::vector<double>>(clFeatureCounts["Count"]);
  
  std::set<int> featureSet(featureLabels2.begin(), featureLabels2.end());
  
  double numDocs = std::accumulate(clCounts1.begin(), clCounts1.end(), 0.0, 
                                   [](double &sum, const double &cnt){return sum+cnt;});
  
  double numFeatures = (double)featureSet.size();
  
  for (size_t i = 0; i < clLabels1.size(); i++) priorClassCounts[clLabels1[i]] = clCounts1[i];
  for (size_t i = 0; i < clLabels2.size(); i++) priorClassFeatureCounts[clLabels2[i]][featureLabels2[i]] = clFeatureCounts2[i];
  
  return computePosteriorProbabilities(priorClassFeatureCounts, priorClassCounts, rowColValMap, uniqueLabels, 
                                       numDocs, numFeatures);
}