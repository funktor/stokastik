from collections import defaultdict
import numpy as np, random, re, math, pickle
import data_reader as dr

def lcs(str1, str2):
    cached = defaultdict(dict)
    tokens1, tokens2 = dr.get_tokens(str1), dr.get_tokens(str2)
    
    for i in range(-1, len(tokens1)):
        for j in range(-1, len(tokens2)):
            if i == -1 or j == -1:
                cached[i][j] = [[]]
            else:
                if tokens1[i] == tokens2[j]:
                    out = [x + [(tokens1[i], i)] for x in cached[i - 1][j - 1]]
                else:
                    a, b = cached[i - 1][j], cached[i][j - 1]
                    if len(a[0]) == len(b[0]):
                        out = a + b if a[len(a) - 1] != b[len(b) - 1] else a
                    else:
                        out = a if len(a[0]) > len(b[0]) else b
                        
                cached[i][j] = out
                
    longest = cached[len(tokens1) - 1][len(tokens2) - 1][0]
    
    if len(longest) > 0:
        start, end = longest[0][1], longest[-1][1]
        return ' '.join(tokens1[start:end+1])
    
    return ''

def get_sequence(labels):
    seq_label, last = dict(), 0
    
    for idx in range(len(labels)):
        if labels[idx] != 'O':
            if labels[idx][0] == 'B' or labels[idx][0] == 'S':
                last = idx
            seq_label[last] = idx
    
    return seq_label

def get_classification_score(test_labels, pred_labels):
    tp, fp, fn = defaultdict(float), defaultdict(float), defaultdict(float)
    support = defaultdict(float)
    
    n = len(test_labels)

    for idx in range(n):
        true_label, pred_label = test_labels[idx], pred_labels[idx]
        true_seq, pred_seq = get_sequence(true_label), get_sequence(pred_label)
        
        for start, end in true_seq.items():
            true_tag = true_label[start][2:]
            pred_tag = pred_label[start][2:]
            
            support[true_tag] += 1
            
            if start in pred_seq and pred_seq[start] == end and pred_tag == true_tag:
                tp[true_tag] += 1
            else:
                fn[true_tag] += 1
                
        for start, end in pred_seq.items():
            true_tag = true_label[start][2:]
            pred_tag = pred_label[start][2:]
            
            if start not in pred_seq or pred_seq[start] != end or pred_tag != true_tag:
                fp[pred_tag] += 1

    precision, recall, f1_score = defaultdict(float), defaultdict(float), defaultdict(float)

    tot_precision, tot_recall, tot_f1 = 0.0, 0.0, 0.0
    sum_sup = 0.0

    for label, sup in support.items():
        precision[label] = float(tp[label])/(tp[label] + fp[label]) if label in tp else 0.0
        recall[label] = float(tp[label])/(tp[label] + fn[label]) if label in tp else 0.0

        f1_score[label] = 2 * float(precision[label] * recall[label])/(precision[label] + recall[label]) if precision[label] + recall[label] != 0 else 0.0

        tot_f1 += sup * f1_score[label]
        tot_precision += sup * precision[label]
        tot_recall += sup * recall[label]

        sum_sup += sup

    for label, sup in support.items():
        print label, precision[label], recall[label], f1_score[label], sup
        
    return tot_precision/float(sum_sup), tot_recall/float(sum_sup), tot_f1/float(sum_sup), sum_sup

def get_accuracy(test_labels, pred_labels):
    n = len(test_labels)
    
    correct = np.sum([test_labels[idx] == pred_labels[idx] for idx in range(n)])
    return float(correct)/n

def get_transition_probs(labels, tag_transformer):
    tags = [tag for tag, idx in tag_transformer.items()]
    n = len(tags)
    start_cnts, transition_cnts = np.zeros(n), np.zeros((n, n))
    
    for label in labels:
        for idx in range(len(label)):
            if idx == 0:
                start_cnts[tag_transformer[label[idx]]] += 1
            else:
                transition_cnts[tag_transformer[label[idx-1]], tag_transformer[label[idx]]] += 1
                
    start_cnts = start_cnts/np.sum(start_cnts)
    transition_cnts = (transition_cnts.T/np.sum(transition_cnts, axis=1)).T
    
    return start_cnts, transition_cnts

def viterbi_decoding(emission_probs, transition_probs, start_probs, tag_inverse_transformer):
    n, m = emission_probs.shape
    viterbi_state = np.zeros((n, m))
    
    for i in range(n):
        if i == 0:
            viterbi_state[i] = emission_probs[i] + start_probs
        else:
            viterbi_state[i] = np.max(np.add.outer(viterbi_state[i-1], emission_probs[i]) + transition_probs, axis=0)
    
    output_states = np.zeros(n)
    
    for i in reversed(range(n)):
        if i == n-1:
            output_states[i] = np.argmax(viterbi_state[i])
        else:
            nxt_state = int(output_states[i+1])
            output_states[i] = np.argmax(viterbi_state[i] + emission_probs[i+1,nxt_state] + transition_probs[:,nxt_state])
            
    output_states = [tag_inverse_transformer[int(x)] for x in output_states]
    
    return output_states