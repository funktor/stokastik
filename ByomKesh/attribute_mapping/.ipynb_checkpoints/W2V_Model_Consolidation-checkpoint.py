import glob

final_vectors = {}
final_idf_scores, partition_cnt = {}, {}

batch_size = 5*10**6

for f in glob.glob("wv_models/*.pkl"):
    print(f)
    wv = utils.load_data_pkl(f)
    vocab = wv.feature_extractor.vocabulary_
    idf_scores = wv.feature_extractor.idf_
    dfs = {k:((batch_size+1)/np.exp(idf_scores[v]-1))-1 for k, v in vocab.items()}
    
    for k, v in dfs.items():
        if k not in final_idf_scores:
            final_idf_scores[k] = 0
        final_idf_scores[k] += v
    
    for k in wv.wv_model.wv.vocab:
        if k not in partition_cnt:
            partition_cnt[k] = 0
        partition_cnt[k] += 1

        if k not in final_vectors:
            final_vectors[k] = np.copy(wv.wv_model.wv[k])
        else:
            final_vectors[k] += np.copy(wv.wv_model.wv[k])

num_docs = 45*10**6

final_vectors = {k:v/partition_cnt[k] for k, v in final_vectors.items()}
final_vectors = {k:' '.join(v.astype(str)) for k, v in final_vectors.items()}
final_idf_scores = {k:np.log((num_docs+1)/(v+1))+1 for k, v in final_idf_scores.items()}

with open('wv_text_models/vectors.txt', 'w') as f:
    out = [k + ' ' + v for k, v in final_vectors.items()]
    f.write("\n".join(out))
    
with open('wv_text_models/idf_scores.txt', 'w') as f:
    out = [k + ' ' + str(v) for k, v in final_idf_scores.items()]
    f.write("\n".join(out))