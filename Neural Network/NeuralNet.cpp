// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>

using namespace Rcpp;

struct NeuralNet {
  std::map<int, std::map<int, double>> input;
  std::vector<int> inputClassLabels;
  std::map<int, std::map<int, std::map<int, double>>> weights, prevWeights;
  std::map<int, double> biases;
  std::vector<int> layerNumNeurons;
  std::map<int, std::map<int, double>> outputs;
};

void feedForward(NeuralNet &nnet, const int &randomDoc) {
  
  int numLayers = (int)nnet.layerNumNeurons.size();
  
  std::map<int, double> inputDoc = nnet.input[randomDoc];
  for (auto q = inputDoc.begin(); q != inputDoc.end(); ++q) nnet.outputs[1][q->first] = q->second;
  
  for (int layer = 2; layer <= numLayers; layer++) {
    
    double layerSumOutputs = 0;
    
    for (int j = 1; j <= nnet.layerNumNeurons[layer-1]; j++) {
      double sum = 0;
      
      if (layer == 2) for (auto q = inputDoc.begin(); q != inputDoc.end(); ++q) sum += nnet.weights[layer-1][q->first][j]*nnet.outputs[layer-1][q->first];
      else for (int k = 1; k <= nnet.layerNumNeurons[layer-2]; k++) sum += nnet.weights[layer-1][k][j]*nnet.outputs[layer-1][k];
      
      sum += nnet.biases[layer];
      
      nnet.outputs[layer][j] = exp(sum);
      layerSumOutputs += nnet.outputs[layer][j];
    }
    for (int j = 1; j <= nnet.layerNumNeurons[layer-1]; j++) nnet.outputs[layer][j] /= layerSumOutputs;
  }
}

void crossEntropyLossBP(NeuralNet &nnet, std::vector<double> &errors, const double &learningRate, const double &momentumParameter, const int &randomDoc) {
  
  int numLayers = (int)nnet.layerNumNeurons.size();
  
  std::map<int, double> outputDiffs = nnet.outputs[numLayers];
  
  std::map<int, double> inputDoc = nnet.input[randomDoc];
  int actualClass = nnet.inputClassLabels[randomDoc-1];
  outputDiffs[actualClass] -= 1;
  
  double error = 0;
  
  for (auto p = nnet.outputs[numLayers].begin(); p != nnet.outputs[numLayers].end(); ++p) {
    if (p->first == actualClass) error += -log(p->second);
    else error += -log(1-p->second);
  }
  
  errors.push_back(error);
  
  std::map<int, std::map<int, std::map<int, double>>> gradient, prevGrad;
  std::map<int, std::map<int, double>> sumPrevGrad;
  
  for (int j = 1; j <= nnet.layerNumNeurons[numLayers-1]; j++) sumPrevGrad[numLayers][j] = outputDiffs[j];
  
  for (int layer = numLayers-1; layer >= 2; layer--) {
    
    for (int j = 1; j <= nnet.layerNumNeurons[layer-1]; j++) {
      for (int k = 1; k <= nnet.layerNumNeurons[layer]; k++) {
        
        gradient[layer][j][k] = sumPrevGrad[layer+1][k]*nnet.outputs[layer][j];
        
        prevGrad[layer][j][k] = sumPrevGrad[layer+1][k]*nnet.weights[layer][j][k];
        
        nnet.weights[layer][j][k] -= (learningRate*gradient[layer][j][k] + momentumParameter*(nnet.prevWeights[layer][j][k] - nnet.weights[layer][j][k]));
        
        sumPrevGrad[layer][j] += prevGrad[layer][j][k];
      }
    }
  }
  
  int layer = 1;
  
  for (auto p = inputDoc.begin(); p != inputDoc.end(); ++p) {
    int j = p->first;
    
    for (int k = 1; k <= nnet.layerNumNeurons[layer]; k++) {
      
      gradient[layer][j][k] = sumPrevGrad[layer+1][k]*nnet.outputs[layer][j];
      
      nnet.weights[layer][j][k] -= (learningRate*gradient[layer][j][k] + momentumParameter*(nnet.prevWeights[layer][j][k] - nnet.weights[layer][j][k]));
    }
  }
  
  nnet.prevWeights = nnet.weights;
}

// [[Rcpp::export]]
List cpp__nnet(std::vector<int> rows, std::vector<int> cols, std::vector<double> vals, 
               std::vector<int> classes, std::vector<int> layerNumNeurons, 
               double learningRate = 0.1, int maxNumIter=10, double momentumParameter=0.9) {
  
  NeuralNet nnet;
  
  for (size_t i = 0; i < rows.size(); i++) nnet.input[rows[i]][cols[i]] = vals[i];
  nnet.layerNumNeurons = layerNumNeurons;
  
  std::vector<double> errors;
  
  int numLayers = (int)layerNumNeurons.size();
  
  std::set<int> docSet(rows.begin(), rows.end());
  std::vector<int> docs(docSet.begin(), docSet.end());
  
  for (int i = 1; i <= (numLayers-1); i++) {
    for (int j = 1; j <= layerNumNeurons[i-1]; j++) {
      for (int k = 1; k <= layerNumNeurons[i]; k++) {
        nnet.weights[i][j][k] = (double)rand()/(double)(RAND_MAX);
        nnet.prevWeights[i][j][k] = 0;
      }
    }
  }
  
  for (int i = 2; i <= numLayers; i++) nnet.biases[i] = (double)rand()/(double)(RAND_MAX);
  
  int numIter = 0;
  while (numIter < maxNumIter) {
    
    int randomRow = docs[rand() % (int)docs.size()];
    feedForward(nnet, randomRow);
    
    crossEntropyLossBP(nnet, errors, learningRate, momentumParameter, randomRow);
    
    numIter++;
  }
  
  std::vector<int> weightLayer, weightFromNeuron, weightToNeuron;
  std::vector<int> biasLayer;
  std::vector<double> weightVal, biasVal;
  
  for (auto a2 = nnet.weights.begin(); a2 != nnet.weights.end(); ++a2) {
    int layer = a2->first;
    std::map<int, std::map<int, double>> b2 = a2->second;
    
    for (auto a3 = b2.begin(); a3 != b2.end(); ++a3) {
      int fromNeuron = a3->first;
      std::map<int, double> b3 = a3->second;
      
      for (auto a4 = b3.begin(); a4 != b3.end(); ++a4) {
        int toNeuron = a4->first;
        double b4 = a4->second;
        
        weightLayer.push_back(layer);
        weightFromNeuron.push_back(fromNeuron);
        weightToNeuron.push_back(toNeuron);
        weightVal.push_back(b4);
      }
    }
  }
  
  for (auto a2 = nnet.biases.begin(); a2 != nnet.biases.end(); ++a2) {
    int layer = a2->first;
    double b2 = a2->second;
    
    biasLayer.push_back(layer);
    biasVal.push_back(b2);
  }
  
  return List::create(_["weights"]=DataFrame::create(_["layer"]=weightLayer, _["fromN"]=weightFromNeuron, _["toN"]=weightToNeuron, _["val"]=weightVal), 
                      _["biases"]=DataFrame::create(_["layer"]=biasLayer, _["val"]=biasVal), 
                        _["errors"]=errors, _["layers"]=layerNumNeurons);
}

// [[Rcpp::export]]
std::map<int, std::map<int, double>> cpp__nnetPredict(std::vector<int> rows, std::vector<int> cols, std::vector<double> vals, 
                                                      std::vector<int> wLayer, std::vector<int> wFromNeuron, std::vector<int> wToNeuron, std::vector<double> wVal,
                                                      std::vector<int> bLayer, std::vector<double> bVal,
                                                      std::vector<int> layerNumNeurons) {
  
  std::map<int, std::map<int, double>> myOut;
  
  NeuralNet nnet;
  
  for (size_t i = 0; i < rows.size(); i++) nnet.input[rows[i]][cols[i]] = vals[i];
  
  int numLayers = (int)layerNumNeurons.size();
  
  for (size_t i = 0; i < wLayer.size(); i++) nnet.weights[wLayer[i]][wFromNeuron[i]][wToNeuron[i]] = wVal[i];
  for (size_t i = 0; i < bLayer.size(); i++) nnet.biases[bLayer[i]] = bVal[i];
  
  for (auto p = nnet.input.begin(); p != nnet.input.end(); ++p) {
    int row = p->first;
    std::map<int, double> colValMap = p->second;
    
    feedForward(nnet, row);
    
    myOut[row] = nnet.outputs[numLayers];
  }
  
  return myOut;
}