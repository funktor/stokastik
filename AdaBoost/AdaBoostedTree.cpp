// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
#include <fstream>
#include <queue>
#include <stack>
#include <random>

using namespace Rcpp;

typedef std::unordered_map<int, std::unordered_map<double, std::set<int>>> DataFormat;

struct Node {
  int featureIndex, depth;
  double featureDecisionVal;
  
  std::unordered_map<int, double> classProbs;
  std::set<int> rows, leftRows, rightRows;
  
  Node* left;
  Node* right;
  
  bool operator==(const Node* x){
    return this->featureDecisionVal == x->featureDecisionVal 
    && this->featureIndex == x->featureIndex 
    && this->rows.size() == x->rows.size();
  }
};

struct InputMetaData {
  std::unordered_map<int, std::unordered_map<double, std::set<int>>> colValAllRowsMap;
  std::unordered_map<int, std::set<double>> colSortedVals;
  std::unordered_map<int, std::set<int>> rowCols;
  std::unordered_map<int, std::unordered_map<int, double>> rowColValMap;
  std::unordered_map<int, double> instanceWeights;
  std::unordered_map<int, int> rowClassMap;
  std::set<int> classLabels;
};

std::unordered_map<int, double> computeClassProbs(const std::set<int> &rows, InputMetaData &metaData) {
  
  std::unordered_map<int, double> classProbs;
  
  for (auto p = metaData.classLabels.begin(); p != metaData.classLabels.end(); ++p) classProbs[*p] = 0;
  
  double weightSum = 0;
  
  for (auto p = rows.begin(); p != rows.end(); ++p) {
    classProbs[metaData.rowClassMap[*p]] += metaData.instanceWeights[*p];
    weightSum += metaData.instanceWeights[*p];
  }
  
  for (auto p = metaData.classLabels.begin(); p != metaData.classLabels.end(); ++p) classProbs[*p] /= weightSum;
  
  return classProbs;
}

double giniImpurity(const std::set<int> &rows, InputMetaData &metaData) {
  
  std::unordered_map<int, double> classProbs = computeClassProbs(rows, metaData);
  
  double impurity = 0;
  
  for (auto p = classProbs.begin(); p != classProbs.end(); ++p) impurity += (p->second)*(1-p->second);
  
  return impurity;
}

double entropy(const std::set<int> &rows, InputMetaData &metaData) {
  
  std::unordered_map<int, double> classProbs = computeClassProbs(rows, metaData);
  
  double entropy = 0;
  
  for (auto p = classProbs.begin(); p != classProbs.end(); ++p) entropy += -(p->second)*log2(p->second);
  
  return entropy;
}

std::pair<int, double> bestClass(std::unordered_map<int, double> &classProbs) {
  
  double maxProb = std::numeric_limits<double>::min();
  int bestClass = -1;
  
  for (auto p = classProbs.begin(); p != classProbs.end(); ++p) {
    if (p->second > maxProb) {
      maxProb = p->second;
      bestClass = p->first;
    }
  }
  
  return std::make_pair(bestClass, maxProb);
}

Node* getLeafNode(const std::set<int> &rows, InputMetaData &metaData) {
  
  Node *node = new Node();
  
  node->featureIndex = -1;
  node->featureDecisionVal = -1;
  
  node->classProbs = computeClassProbs(rows, metaData);
  
  node->rows = rows;
  
  node->left = NULL;
  node->right = NULL;
  
  return node;
}

Node* createNode(const std::set<int> &currRows,
                 InputMetaData &metaData,
                 double (*costFuntion) (const std::set<int> &, InputMetaData &),
                 const bool &maxDepthReached) {
  
  Node *node = new Node();
  
  std::unordered_map<int, double> classProbs = computeClassProbs(currRows, metaData);
  
  if ((int)classProbs.size() == 1 || maxDepthReached) node = getLeafNode(currRows, metaData);
  
  else {
    double maxFeatureGain = 0;
    double featureDecisionVal = -1;
    int featureIdx = -1;
    
    double totalCost=costFuntion(currRows, metaData);
    
    std::set<int> leftRows, rightRows;
    
    std::set<int> currCols;
    
    for (auto p = currRows.begin(); p != currRows.end(); ++p) {
      std::unordered_map<int, double> colValMap = metaData.rowColValMap[*p];
      for (auto q = colValMap.begin(); q != colValMap.end(); ++q) currCols.insert(q->first);
    }
    
    std::vector<int> colsVec(currCols.begin(), currCols.end());
    
    std::random_shuffle(colsVec.begin(), colsVec.end());
    
    std::set<int> thisCols(colsVec.begin(), colsVec.begin()+0.05*colsVec.size());
    
    for (auto p = thisCols.begin(); p != thisCols.end(); ++p) {
      
      int feature = *p;
      
      std::set<int> currVals;
      std::set<double> sortedVals = metaData.colSortedVals[*p];
      
      std::set<int> allRows = metaData.colValAllRowsMap[feature][*sortedVals.rbegin()];
      std::set<int> thisRows, missingRows;
      
      std::set_intersection(currRows.begin(), currRows.end(), allRows.begin(), allRows.end(),
                            std::inserter(thisRows, thisRows.end()));
      
      std::set_difference(currRows.begin(), currRows.end(), thisRows.begin(), thisRows.end(),
                          std::inserter(missingRows, missingRows.end()));
      
      if (missingRows.size() > 0) sortedVals.insert(0);
      
      double maxGain = 0;
      double decisionVal = -1;
      
      for (auto q = sortedVals.begin(); q != sortedVals.end(); ++q) {
        
        std::set<int> leftCostRows, rightCostRows;
        
        if (*q > 0) {
          std::set<int> leftR = metaData.colValAllRowsMap[feature][*q];
          
          std::set_intersection(currRows.begin(), currRows.end(), leftR.begin(), leftR.end(),
                                std::inserter(leftCostRows, leftCostRows.end()));
        }
        
        leftCostRows.insert(missingRows.begin(), missingRows.end());
        
        double leftCost = costFuntion(leftCostRows, metaData);
        
        std::set_difference(currRows.begin(), currRows.end(), leftCostRows.begin(), leftCostRows.end(), 
                            std::inserter(rightCostRows, rightCostRows.end()));
        
        double rightCost = costFuntion(rightCostRows, metaData);
        
        double w1 = (double)leftCostRows.size()/(double)currRows.size();
        double w2 = (double)rightCostRows.size()/(double)currRows.size();
        
        double gain = totalCost-(w1*leftCost+w2*rightCost);
        
        if (gain > maxGain) {
          maxGain = gain;
          decisionVal = *q;
        }
      }
      
      if (maxGain > maxFeatureGain && decisionVal < *sortedVals.rbegin()) {
        maxFeatureGain = maxGain;
        featureDecisionVal = decisionVal;
        featureIdx = feature;
      }
    }
    
    if (featureIdx != -1) {
      
      std::set<double> sortedVals = metaData.colSortedVals[featureIdx];
      
      std::set<int> allRows = metaData.colValAllRowsMap[featureIdx][*sortedVals.rbegin()];
      std::set<int> thisRows, missingRows;
      
      std::set_intersection(currRows.begin(), currRows.end(), allRows.begin(), allRows.end(),
                            std::inserter(thisRows, thisRows.end()));
      
      std::set_difference(currRows.begin(), currRows.end(), thisRows.begin(), thisRows.end(),
                          std::inserter(missingRows, missingRows.end()));
      
      std::set<int> leftR = metaData.colValAllRowsMap[featureIdx][featureDecisionVal];
      
      std::set_intersection(currRows.begin(), currRows.end(), leftR.begin(), leftR.end(),
                            std::inserter(leftRows, leftRows.end()));
      
      leftRows.insert(missingRows.begin(), missingRows.end());
      
      std::set_difference(currRows.begin(), currRows.end(), leftRows.begin(), leftRows.end(), 
                          std::inserter(rightRows, rightRows.end()));
      
      
      
      node->featureIndex = featureIdx;
      node->featureDecisionVal = featureDecisionVal;
      
      node->classProbs = computeClassProbs(currRows, metaData);
      
      node->rows = currRows;
      
      node->leftRows = leftRows;
      node->rightRows = rightRows;
    }
    
    else node = getLeafNode(currRows, metaData);
  }
  
  return node;
}

Node* constructTree(const std::set<int> &rows,
                    InputMetaData &metaData, 
                    double (*costFuntion) (const std::set<int> &, InputMetaData &),
                    const int &maxDepth) {
  
  std::queue<Node*> nodeQ;
  
  Node *node = createNode(rows, metaData, costFuntion, false);
  node->depth = 0;
  
  Node* root = node;
  
  nodeQ.push(node);
  
  while(!nodeQ.empty()) {
    node = nodeQ.front();
    nodeQ.pop();
    
    if (node->featureDecisionVal != -1) {
      bool maxDepthReached = (node->depth == maxDepth-1);
      
      node->left = createNode(node->leftRows, metaData, costFuntion, maxDepthReached);
      node->right = createNode(node->rightRows, metaData, costFuntion, maxDepthReached);
      
      node->left->depth = node->depth + 1;
      node->right->depth = node->depth + 1;
      
      nodeQ.push(node->left);
      nodeQ.push(node->right);
    }
  }
  
  return root;
}

DataFrame transformTreeIntoDF(Node* &node, DataFrame &leafNodeClassProbs) {
  
  std::vector<int> nodeIndex, leftNodeIndex, rightNodeIndex, featureIndex, leafLabels, leafIndices;
  std::vector<double> featureDecisionVal, leafLabelProbs;
  
  std::queue<Node*> nodeQ;
  nodeQ.push(node);
  
  int index = 0;
  int lastIndex = 0;
  
  while (!nodeQ.empty()) {
    Node* n = nodeQ.front();
    
    nodeIndex.push_back(index);
    featureIndex.push_back(n->featureIndex);
    featureDecisionVal.push_back(n->featureDecisionVal);
    
    if (n->left != NULL) {
      leftNodeIndex.push_back(++lastIndex);
      nodeQ.push(n->left);
      
      rightNodeIndex.push_back(++lastIndex);
      nodeQ.push(n->right);
    }
    
    else {
      leftNodeIndex.push_back(-1);
      rightNodeIndex.push_back(-1);
      
      std::unordered_map<int, double> predProbs = n->classProbs;
      
      for (auto p = predProbs.begin(); p != predProbs.end(); ++p) {
        leafLabels.push_back(p->first);
        leafIndices.push_back(index);
        leafLabelProbs.push_back(p->second);
      }
    }
    
    nodeQ.pop();
    index++;
  }
  
  leafNodeClassProbs = DataFrame::create(_["LeafIndex"]=leafIndices, _["LeafLabel"]=leafLabels, _["LeafLabelProb"]=leafLabelProbs);
  
  return DataFrame::create(_["NodeIndex"]=nodeIndex, _["LeftNodeIndex"]=leftNodeIndex, 
                           _["RightNodeIndex"]=rightNodeIndex, _["FeatureIndex"]=featureIndex, 
                           _["FeatureDecisionVal"]=featureDecisionVal);
}

std::unordered_map<int, double> treePredict(Node* &node, 
                                            std::unordered_map<int, double> &colValMap) {
  
  if (node->featureIndex == -1) return node->classProbs;
  
  else {
    int decisionFeature = node->featureIndex;
    double decisionVal = node->featureDecisionVal;
    
    if (colValMap[decisionFeature] <= decisionVal) return treePredict(node->left, colValMap);
    else return treePredict(node->right, colValMap);
  }
}

// [[Rcpp::export]]
List cpp__adaBoostedTree(DataFrame inputSparseMatrix, std::vector<int> classLabels, 
                         int boostingRounds = 5, int maxDepth=100) {
  
  std::vector<DataFrame> trees;
  std::vector<double> treeWeights;
  
  std::vector<DataFrame> leafNodeClassProbs;
  
  InputMetaData metaData;
  
  std::vector<int> rows = inputSparseMatrix["i"];
  std::vector<int> cols = inputSparseMatrix["j"];
  std::vector<double> vals = inputSparseMatrix["v"];
  
  std::set<int> uniqueRows(rows.begin(), rows.end());
  std::set<int> uniqueCols(cols.begin(), cols.end());
  std::set<int> uniqueLabels(classLabels.begin(), classLabels.end());
  
  metaData.classLabels = uniqueLabels;
  
  for (size_t i = 0; i < rows.size(); i++) metaData.colSortedVals[cols[i]].insert(vals[i]);
  for (size_t i = 0; i < rows.size(); i++) metaData.colValAllRowsMap[cols[i]][vals[i]].insert(rows[i]);
  for (size_t i = 0; i < rows.size(); i++) metaData.rowCols[rows[i]].insert(cols[i]);
  
  for (auto p = uniqueCols.begin(); p != uniqueCols.end(); ++p) {
    std::set<double> sortedVals = metaData.colSortedVals[*p];
    std::vector<double> sortedValsVec(sortedVals.begin(), sortedVals.end());
    
    for (size_t i = 1; i < sortedValsVec.size(); i++) 
      metaData.colValAllRowsMap[*p][sortedValsVec[i]].insert(metaData.colValAllRowsMap[*p][sortedValsVec[i-1]].begin(), 
                                                             metaData.colValAllRowsMap[*p][sortedValsVec[i-1]].end());
  }
  
  for (size_t i = 0; i < rows.size(); i++) metaData.rowColValMap[rows[i]][cols[i]] = vals[i];
  for (auto p = uniqueRows.begin(); p != uniqueRows.end(); ++p) metaData.rowClassMap[*p] = classLabels[*p-1];
  
  for (auto p = uniqueRows.begin(); p != uniqueRows.end(); ++p) metaData.instanceWeights[*p]=1/(double)uniqueRows.size();
  
  int iterNum = 1;
  
  while(iterNum <= boostingRounds) {
    
    Node* tree = constructTree(uniqueRows, metaData, giniImpurity, maxDepth);
    
    double err = 0;
    
    std::set<int> errorRows;
    
    for (auto p = uniqueRows.begin(); p != uniqueRows.end(); ++p) {
      std::unordered_map<int, double> predClassProbs = treePredict(tree, metaData.rowColValMap[*p]);
      std::pair<int, double> best = bestClass(predClassProbs);
      
      if (best.first != metaData.rowClassMap[*p]) {
        errorRows.insert(*p);
        err += metaData.instanceWeights[*p];
      }
    }
    
    if (err > 0.5 || err <= 0) break;
    
    double treeWt = log((1-err)/err)+log(uniqueLabels.size()-1);
    
    DataFrame predClassProbs;
    
    trees.push_back(transformTreeIntoDF(tree, predClassProbs));
    treeWeights.push_back(treeWt);
    
    leafNodeClassProbs.push_back(predClassProbs);
    
    double sumWeights = 0;
    
    for (auto p = uniqueRows.begin(); p != uniqueRows.end(); ++p) {
      if (errorRows.find(*p) != errorRows.end()) metaData.instanceWeights[*p] *= exp(treeWt);
      sumWeights += metaData.instanceWeights[*p];
    }
    
    for (auto p = uniqueRows.begin(); p != uniqueRows.end(); ++p) metaData.instanceWeights[*p] /= sumWeights;
    
    iterNum++;
  }
  
  double sumTreeWeights = 0;
  
  for (size_t i = 0; i < treeWeights.size(); i++) sumTreeWeights += treeWeights[i];
  for (size_t i = 0; i < treeWeights.size(); i++) treeWeights[i] /= sumTreeWeights;
  
  return List::create(_["Trees"]=trees, _["TreeWeights"]=treeWeights, _["TreeClassProbs"]=leafNodeClassProbs);
}

struct NodeDF {
  int index, leftIndex, rightIndex, featureIndex;
  double decisionVal;
};


std::unordered_map<int, double> dfPredict(std::unordered_map<int, NodeDF> &nodeDFMap,
                                          std::unordered_map<int, double> &colValMap,
                                          std::unordered_map<int, std::unordered_map<int, double>> &predClassProbs) {
  
  int index = 0;
  
  while(true) {
    if (nodeDFMap[index].featureIndex == -1) return predClassProbs[index];
    else {
      if (colValMap[nodeDFMap[index].featureIndex] <= nodeDFMap[index].decisionVal) index = nodeDFMap[index].leftIndex;
      else index = nodeDFMap[index].rightIndex;
    }
  }
}

// [[Rcpp::export]]
std::map<int, std::map<int, double>> cpp__test(DataFrame inputSparseMatrix, List modelsMetaData) {
  
  std::map<int, std::map<int, double>> output;
  
  std::vector<int> rows = inputSparseMatrix["i"];
  std::vector<int> cols = inputSparseMatrix["j"];
  std::vector<double> vals = inputSparseMatrix["v"];
  
  std::unordered_map<int, std::unordered_map<int, double>> rowColValMap;
  
  for (size_t i = 0; i < rows.size(); i++) rowColValMap[rows[i]][cols[i]] = vals[i];
  
  std::vector<DataFrame> models = modelsMetaData["Trees"];
  std::vector<double> modelWeights = modelsMetaData["TreeWeights"];
  std::vector<DataFrame> leafNodeClassProbs = modelsMetaData["TreeClassProbs"];
  
  std::map<int, std::map<int, double>> predProbs;
  
  for (size_t i = 0; i < models.size(); i++) {
    DataFrame model = models[i];
    
    std::vector<int> nodeIndex = model["NodeIndex"];
    std::vector<int> leftNodeIndex = model["LeftNodeIndex"];
    std::vector<int> rightNodeIndex = model["RightNodeIndex"];
    std::vector<int> featureIndex = model["FeatureIndex"];
    std::vector<double> featureDecisionVal = model["FeatureDecisionVal"];
    
    DataFrame leafNodeClassProb = leafNodeClassProbs[i];
    
    std::vector<int> leafIndex = leafNodeClassProb["LeafIndex"];
    std::vector<int> leafLabel = leafNodeClassProb["LeafLabel"];
    std::vector<double> leafLabelProb = leafNodeClassProb["LeafLabelProb"];
    
    std::unordered_map<int, std::unordered_map<int, double>> predClassProbs;
    
    for (size_t j = 0; j < leafIndex.size(); j++) predClassProbs[leafIndex[j]][leafLabel[j]] = leafLabelProb[j];
    
    std::unordered_map<int, NodeDF> nodeDFMap;
    
    for (size_t j = 0; j < nodeIndex.size(); j++) {
      NodeDF df;
      
      df.index = nodeIndex[j];
      df.decisionVal = featureDecisionVal[j];
      df.featureIndex = featureIndex[j];
      df.leftIndex = leftNodeIndex[j];
      df.rightIndex = rightNodeIndex[j];
      
      nodeDFMap[nodeIndex[j]] = df;
    }
    
    for (auto p = rowColValMap.begin(); p != rowColValMap.end(); ++p) {
      std::unordered_map<int, double> out = dfPredict(nodeDFMap, p->second, predClassProbs);
      std::pair<int, double> best = bestClass(out);
      
      for (auto q = out.begin(); q != out.end(); ++q) {
        if (q->first == best.first) predProbs[p->first][q->first] += modelWeights[i];
        else predProbs[p->first][q->first] += 0;
      }
    }
  }
  
  for (auto p = predProbs.begin(); p != predProbs.end(); ++p) {
    std::map<int, double> classProbs = p->second;
    
    output[p->first]=classProbs;
  }
  
  return output;
}