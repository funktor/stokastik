#include "FeatureTransformer.h"

struct comparator {
  bool operator()(const std::pair<std::string, double> &a, const std::pair<std::string, double> &b) {
    return a.second > b.second;
  }
};

FeatureTransformer::FeatureTransformer () {}

FeatureTransformer::FeatureTransformer (str_vec x_stop_words, int x_vocab_size, int x_min_ngram, int x_max_ngram) {
    this->stop_words = x_stop_words;
    this->vocab_size = x_vocab_size;
    this->min_ngram = std::max(1, x_min_ngram);
    this->max_ngram = std::max(x_min_ngram, x_max_ngram);
}

FeatureTransformer::~FeatureTransformer () {}

str_2_vec FeatureTransformer::tokenize (str_vec &sentences) {
    std::transform(sentences.begin(), sentences.end(), sentences.begin(), [](std::string x) {return std::regex_replace(x, std::regex("<[^<]+?>|[^\\w\\-]+"), " ");});
    std::transform(sentences.begin(), sentences.end(), sentences.begin(), [](std::string x) {boost::algorithm::to_lower(x); return x;});
    
    std::unordered_set<std::string> stops(this->stop_words.begin(), this->stop_words.end());
    
    str_2_vec tokenized_sents;
    std::string delimiters(" ");
    
    for (size_t i = 0; i < sentences.size(); i++) {
        str_vec words;
        boost::split(words, sentences[i], boost::is_any_of(delimiters));
        std::transform(words.begin(), words.end(), words.begin(), [](std::string x) {boost::algorithm::trim(x); return x;});
        words.erase(std::remove_if(words.begin(), words.end(), [&stops](std::string x){return stops.find(x) != stops.end();}), words.end());
        tokenized_sents.push_back(words);
    }
    
    return tokenized_sents;
}

str_vec FeatureTransformer::ngrams (str_vec &tokens) {
    str_vec ngram_out;
    
    for (int i = this->min_ngram; i <= this->max_ngram; i++) {
        for (int j = 0; j <= static_cast<int> (tokens.size())-i; j++) {
            str_vec temp(tokens.begin() + j, tokens.begin() + j + i);
            std::string x = boost::algorithm::join(temp, " ");
            boost::algorithm::trim(x);
            ngram_out.push_back(x);
        }
    }
    return ngram_out;
}

str_2_vec FeatureTransformer::ngram_tokenize (str_2_vec &tokenized_sents) {
    str_2_vec ngrams_out;
    
    for (size_t i = 0; i < tokenized_sents.size(); i++) {
        str_vec tokens = tokenized_sents[i];
        str_vec ngram_out = this->ngrams(tokens);
        ngrams_out.push_back(ngram_out);
    }
    return ngrams_out;
}

str_vec FeatureTransformer::mutual_info_select (str_2_vec &tokenized_sents, str_vec &labels) {
    std::unordered_map<std::string, double> token_cnts, label_cnts, mi_values;
    std::unordered_map<std::string, std::unordered_map<std::string, double>> label_token_cnts;
    
    std::priority_queue<std::pair<std::string, double>, std::vector<std::pair<std::string, double>>, comparator> mi_max_heap;
        
    int total = static_cast<int> (tokenized_sents.size());
    
    for (size_t i = 0; i < tokenized_sents.size(); i++) {
        str_vec tokens = tokenized_sents[i];
        std::unordered_set<std::string> unique_tokens(tokens.begin(), tokens.end());
        
        std::string label = labels[i];
        
        std::for_each(unique_tokens.begin(), unique_tokens.end(), [&token_cnts](std::string token){token_cnts[token]++;});
        label_cnts[label]++;
        std::for_each(unique_tokens.begin(), unique_tokens.end(), [&label_token_cnts, &label](std::string token){label_token_cnts[label][token]++;});
    }
    
    for (auto it = label_token_cnts.begin(); it != label_token_cnts.end(); ++it) {
        std::string label = it->first;
        std::unordered_map<std::string, double> t_counts = it->second;
        
        for (auto it2 = t_counts.begin(); it2 != t_counts.end(); ++it2) {
            std::string token = it2->first;
            double val = it2->second;

            double x11 = val;
            double x10 = label_cnts[label] - val;
            double x01 = token_cnts[token] - val;
            double x00 = total - (x11 + x10 + x01);

            double x1 = label_cnts[label];
            double x0 = total - x1;

            double y1 = token_cnts[token];
            double y0 = total - y1;

            double p = x11/total;
            double q = x10/total;
            double r = x01/total;
            double s = x00/total;

            double u = x1/total;
            double v = x0/total;
            double w = y1/total;
            double z = y0/total;

            double a1 = (p != 0)? p*log2(p/(u*w)) : 0;
            double a2 = (q != 0)? q*log2(q/(u*z)) : 0;
            double a3 = (r != 0)? r*log2(r/(v*w)) : 0;
            double a4 = (s != 0)? s*log2(s/(v*z)) : 0;

            double mi = a1 + a2 + a3 + a4;
            mi_values[token] = std::max(mi_values[token], mi);
        }
    }
    
    for (auto it = mi_values.begin(); it != mi_values.end(); ++it) {
        std::string token = it->first;
        double mi_value = it->second;
        
        if (static_cast<int> (mi_max_heap.size()) < this->vocab_size) {
            mi_max_heap.push(std::make_pair(token, mi_value));
        }
        else {
            if (mi_value > mi_max_heap.top().second) {
                mi_max_heap.pop();
                mi_max_heap.push(std::make_pair(token, mi_value));
            }
        }
    }
    
    str_vec imp_features;
    while (!mi_max_heap.empty()) {
        std::pair<std::string, double> x = mi_max_heap.top();
        imp_features.push_back(x.first);
        mi_max_heap.pop();
    }
    
    return imp_features;
}

str_dbl_map FeatureTransformer::compute_idf (str_2_vec &tokenized_sents) {
    std::unordered_map<std::string, double> token_idfs;
    int n = static_cast<int> (tokenized_sents.size());
    std::unordered_set<std::string> c_vocab(this->vocab.begin(), this->vocab.end());
    
    for (size_t i = 0; i < tokenized_sents.size(); i++) {
        str_vec tokens = tokenized_sents[i];
        tokens.erase(std::remove_if(tokens.begin(), tokens.end(), [&c_vocab](std::string x){return c_vocab.find(x) == c_vocab.end();}), tokens.end());
        
        std::unordered_set<std::string> unique_tokens(tokens.begin(), tokens.end());
        std::for_each(unique_tokens.begin(), unique_tokens.end(), [&token_idfs](std::string token){token_idfs[token]++;});
    }
    
    std::for_each(token_idfs.begin(), token_idfs.end(), [&n](std::pair<const std::string, double> &x) {x.second = log(static_cast<double> (n)/x.second);});
    
    return token_idfs;
}

SparseMat FeatureTransformer::compute_tf_idf (str_2_vec &tokenized_sents) {
    std::unordered_map<int, std::unordered_map<std::string, double>> token_counts;
    std::unordered_set<std::string> c_vocab(this->vocab.begin(), this->vocab.end());
    
    for (size_t i = 0; i < tokenized_sents.size(); i++) {
        for (size_t j = 0; j < tokenized_sents[i].size(); j++) {
            std::string token = tokenized_sents[i][j];
            if (c_vocab.find(token) != c_vocab.end()) {
                token_counts[i][token]++;
            }
        }
    }
    
    SparseMat out;
    
    for (auto it = token_counts.begin(); it != token_counts.end(); ++it) {
        int doc_index = it->first;
        std::unordered_map<std::string, double> t_cnt = it->second;
        
        for (auto it2 = t_cnt.begin(); it2 != t_cnt.end(); ++it2) {
            std::string token = it2->first;
            double tf = it2->second;
            
            out.row.push_back(doc_index);
            out.col.push_back(this->vocab_inv_index[token]);
            out.data.push_back(this->idf_scores[token]*tf);
        }
    }
    return out;
}

void FeatureTransformer::fit (str_vec sentences, str_vec labels) {
    str_2_vec tokenized_sents = this->tokenize(sentences);
    str_2_vec ngrams = this->ngram_tokenize(tokenized_sents);
    
    this->vocab = this->mutual_info_select(ngrams, labels);
    std::sort(this->vocab.begin(), this->vocab.end());
    
    for (size_t i = 0; i < this->vocab.size(); i++) {
        this->vocab_inv_index[this->vocab[i]] = static_cast<int> (i);
    }
    
    this->idf_scores = this->compute_idf(ngrams);
}

SparseMat FeatureTransformer::transform (str_vec sentences) {
    str_2_vec tokenized_sents = this->tokenize(sentences);
    str_2_vec ngrams = this->ngram_tokenize(tokenized_sents);
    
    return this->compute_tf_idf(ngrams);
}