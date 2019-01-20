#ifndef FEATURETRANSFORMER_H
#define FEATURETRANSFORMER_H

#include <iostream> 
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <boost/algorithm/string.hpp>
#include <regex>
#include <string>
#include <algorithm>

struct SparseMat {
    std::vector<int> row;
    std::vector<int> col;
    std::vector<double> data;
};

typedef std::vector<std::string> str_vec;
typedef std::unordered_map<std::string, int> str_int_map;
typedef std::unordered_map<std::string, double> str_dbl_map;
typedef std::vector<std::vector<std::string>> str_2_vec;

class FeatureTransformer {
    private:
        int min_ngram;
        int max_ngram;
        int vocab_size;
        str_vec stop_words;
    
        str_2_vec tokenize (str_vec &sentences);
        str_vec ngrams (str_vec &tokens);
        str_2_vec ngram_tokenize (str_2_vec &tokenized_sents);
        str_vec mutual_info_select (str_2_vec &tokenized_sents, str_vec &labels);
        str_dbl_map compute_idf (str_2_vec &tokenized_sents);
        SparseMat compute_tf_idf (str_2_vec &tokenized_sents);
        
    
    public:
        str_vec vocab;
        str_int_map vocab_inv_index;
        str_dbl_map idf_scores;
    
        FeatureTransformer();
        FeatureTransformer(str_vec x_stop_words, int x_vocab_size, int x_min_ngram, int x_max_ngram);
        ~FeatureTransformer();
        void fit(str_vec sentences, str_vec labels);
        SparseMat transform(str_vec sentences);
};

#endif
