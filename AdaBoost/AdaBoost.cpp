// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
#include <fstream>
#include <queue>
#include <stack>
#include <random>

using namespace Rcpp;

typedef std::unordered_map<int, std::unordered_map<double, std::set<int>>> DataFormat;

struct Node {
  int featureIndex, numLeavesBranch;
  double featureDecisionVal, resubErrorBranch;
  
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
  
  for (auto p = rows.begin(); p != rows.end(); ++p) classProbs[metaData.rowClassMap[*p]]++;
  
  for (auto p = metaData.classLabels.begin(); p != metaData.classLabels.end(); ++p) classProbs[*p] /= (double)rows.size();
  
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
                 double (*costFuntion) (const std::set<int> &, InputMetaData &)) {
  
  Node *node = new Node();
  
  std::unordered_map<int, double> classProbs = computeClassProbs(currRows, metaData);
  std::pair<int, double> best = bestClass(classProbs);
  
  if (best.second >= 0.9) node = getLeafNode(currRows, metaData);
  
  else {
    double maxFeatureGain = 0;
    double featureDecisionVal = -1;
    int featureIdx = -1;
    
    double totalCost=costFuntion(currRows, metaData);
    
    std::set<int> leftRows, rightRows;
    
    std::set<int> thisCols;
    
    for (auto p = currRows.begin(); p != currRows.end(); ++p) 
      thisCols.insert(metaData.rowCols[*p].begin(), metaData.rowCols[*p].end());
    
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
                    double (*costFuntion) (const std::set<int> &, InputMetaData &)) {
  
  std::queue<Node*> nodeQ;
  
  Node *node = createNode(rows, metaData, costFuntion);
  Node* root = node;
  
  nodeQ.push(node);
  
  while(!nodeQ.empty()) {
    node = nodeQ.front();
    nodeQ.pop();
    
    if (node->featureDecisionVal != -1) {
      node->left = createNode(node->leftRows, metaData, costFuntion);
      node->right = createNode(node->rightRows, metaData, costFuntion);
      
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

double observedErrors(Node* &node, std::set<int> &rows, InputMetaData &metaData) {
  
  double errors = 0;
  
  for (auto p = rows.begin(); p != rows.end(); ++p) {
    std::unordered_map<int, double> colValMap = metaData.rowColValMap[*p];
    
    std::unordered_map<int, double> preds = treePredict(node, colValMap);
    std::pair<int, double> best = bestClass(preds);
    
    if (best.first != metaData.rowClassMap[*p]) errors += metaData.instanceWeights[*p];
  }
  
  return errors;
}

Node* redPrune(Node* &node, InputMetaData &metaData, Node* &maxPruneNode) {
  
  if (node->left == NULL) return node;
  else if (node == maxPruneNode) return getLeafNode(node->rows, metaData);
  
  else {
    node->left = redPrune(node->left, metaData, maxPruneNode);
    node->right = redPrune(node->right, metaData, maxPruneNode);
    
    return node;
  }
}

double resbstitutionError(Node* &node, 
                          InputMetaData &metaData, int totalRows) {

  std::set<int> rows = node->rows;
  std::set<int> labels = metaData.classLabels;
  
  std::unordered_map<int, double> classProbs = computeClassProbs(rows, metaData);
  std::pair<int, double> best = bestClass(classProbs);
  
  double errors = 0;
  
  for (auto p = rows.begin(); p != rows.end(); ++p) errors += metaData.instanceWeights[*p]*(1-best.second);
  
  return errors;
}

Node* resubstitutionErrorBranches(Node* &node, InputMetaData &metaData, int totalRows) {
  
  if (node->left == NULL) node->resubErrorBranch = resbstitutionError(node, metaData, totalRows);
  
  else {
    Node* n1 = resubstitutionErrorBranches(node->left, metaData, totalRows);
    Node* n2 = resubstitutionErrorBranches(node->right, metaData, totalRows);
    
    node->resubErrorBranch = n1->resubErrorBranch + n2->resubErrorBranch;
  }
  
  return node;
}

Node* nodeNumLeaves(Node* &node) {
  
  if (node->left == NULL) node->numLeavesBranch = 1;
  else node->numLeavesBranch = nodeNumLeaves(node->left)->numLeavesBranch + nodeNumLeaves(node->right)->numLeavesBranch;
  
  return node;
}

std::pair<Node*, double> getMinComplexityNode(Node* node, InputMetaData &metaData) {
  
  Node* n = node;
  std::queue<Node*> nodeQ;
  
  nodeQ.push(n);
  
  double minComplexity = std::numeric_limits<double>::max();
  Node* minComplexityNode = new Node();
  
  while(!nodeQ.empty()) {
    n = nodeQ.front();
    nodeQ.pop();
    
    double a = resbstitutionError(n, metaData, (int)node->rows.size());
    double b = n->resubErrorBranch;
    int c = n->numLeavesBranch;
    
    double g = (a-b)/((double)c-1);
    
    if (g < minComplexity || (g == minComplexity && minComplexityNode->numLeavesBranch > c)) {
      minComplexity = g;
      minComplexityNode = n;
    }
    
    if (n->left != NULL && n->left->left != NULL) nodeQ.push(n->left);
    if (n->right != NULL && n->right->left != NULL) nodeQ.push(n->right);
  }
  
  return std::make_pair(minComplexityNode, minComplexity);
}

std::map<std::string, std::set<int>> trainValidationPartition(const std::set<int> &allRows) {
  
  std::map<std::string, std::set<int>> output;
  
  std::vector<int> rowsVec(allRows.begin(), allRows.end());
  std::random_shuffle(rowsVec.begin(), rowsVec.end());
  
  int numValidationRows = std::floor(0.3*rowsVec.size());
  
  std::set<int> validationRows(rowsVec.begin(), rowsVec.begin()+numValidationRows);
  std::set<int> trainRows;
  
  std::set_difference(allRows.begin(), allRows.end(), validationRows.begin(), validationRows.end(),
                      std::inserter(trainRows, trainRows.begin()));
  
  output["train"]=trainRows;
  output["validation"]=validationRows;
  
  return output;
}

Node* costComplexityPrune(const std::set<int> &allRows, InputMetaData &metaData) {
  
  std::map<std::string, std::set<int>> partition = trainValidationPartition(allRows);
  
  Node* tree = constructTree(partition["train"], metaData, giniImpurity);
  
  tree = resubstitutionErrorBranches(tree, metaData, (int)tree->rows.size());
  tree = nodeNumLeaves(tree);
  
  double minErr = observedErrors(tree, partition["validation"], metaData);
  double alpha = 0;
  
  while(tree->left != NULL) {
    std::pair<Node*, double> minComplexityNode = getMinComplexityNode(tree, metaData);
    tree = redPrune(tree, metaData, minComplexityNode.first);
    
    tree = resubstitutionErrorBranches(tree, metaData, (int)tree->rows.size());
    tree = nodeNumLeaves(tree);
    
    double err = observedErrors(tree, partition["validation"], metaData);
    
    if (err < minErr) {
      minErr = err;
      alpha = minComplexityNode.second;
    }
  }
  
  Node* fullTree = constructTree(allRows, metaData, giniImpurity);
  
  fullTree = resubstitutionErrorBranches(fullTree, metaData, (int)fullTree->rows.size());
  fullTree = nodeNumLeaves(fullTree);
  
  while(fullTree->left != NULL) {
    std::pair<Node*, double> minComplexityNode = getMinComplexityNode(fullTree, metaData);
    
    fullTree = redPrune(fullTree, metaData, minComplexityNode.first);
    
    fullTree = resubstitutionErrorBranches(fullTree, metaData, (int)fullTree->rows.size());
    fullTree = nodeNumLeaves(fullTree);
    
    if (minComplexityNode.second >= alpha) break;
  }
  
  return fullTree;
}

// [[Rcpp::export]]
List cpp__adaBoostedTree(DataFrame inputSparseMatrix, std::vector<int> classLabels, 
                         int boostingRounds = 5) {
  
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

    Node* node = costComplexityPrune(uniqueRows, metaData);
    double err = observedErrors(node, uniqueRows, metaData);
    
    if (err > 0.5 || err <= 0) break;
    
    double treeWt = log((1-err)/err)+log(uniqueLabels.size()-1);
    
    DataFrame predClassProbs;
    
    trees.push_back(transformTreeIntoDF(node, predClassProbs));
    treeWeights.push_back(treeWt);
    
    leafNodeClassProbs.push_back(predClassProbs);
    
    double sum = 0;
    
    for (auto p = uniqueRows.begin(); p != uniqueRows.end(); ++p) {
      std::unordered_map<int, double> predClassProbs = treePredict(node, metaData.rowColValMap[*p]);
      std::pair<int, double> best = bestClass(predClassProbs);
      
      if (best.first != metaData.rowClassMap[*p]) metaData.instanceWeights[*p] *= exp(treeWt);
      
      sum += metaData.instanceWeights[*p];
    }
    
    for (auto p = uniqueRows.begin(); p != uniqueRows.end(); ++p) metaData.instanceWeights[*p] /= sum;
    
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
DataFrame cpp__test(DataFrame inputSparseMatrix, List modelsMetaData) {

  std::vector<int> doc, topClass;
  std::vector<double> topClassProb;

  std::vector<int> rows = inputSparseMatrix["i"];
  std::vector<int> cols = inputSparseMatrix["j"];
  std::vector<double> vals = inputSparseMatrix["v"];
  
  std::unordered_map<int, std::unordered_map<int, double>> rowColValMap;
  
  for (size_t i = 0; i < rows.size(); i++) rowColValMap[rows[i]][cols[i]] = vals[i];

  std::vector<DataFrame> models = modelsMetaData["Trees"];
  std::vector<double> modelWeights = modelsMetaData["TreeWeights"];
  std::vector<DataFrame> leafNodeClassProbs = modelsMetaData["TreeClassProbs"];

  std::unordered_map<int, std::unordered_map<int, double>> predProbs;

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
      
      predProbs[p->first][best.first] += modelWeights[i];
    }
  }

  for (auto p = predProbs.begin(); p != predProbs.end(); ++p) {
    std::unordered_map<int, double> classProbs = p->second;
    std::pair<int, double> best = bestClass(classProbs);

    doc.push_back(p->first);
    topClass.push_back(best.first);
    topClassProb.push_back(best.second);
  }

  return DataFrame::create(_["Doc"]=doc, _["Class"]=topClass, _["Probability"]=topClassProb);
}