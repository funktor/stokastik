import pandas as pd
import numpy as np
import os, re, math, json, nltk, copy
import Utilities as utils
import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from attribute_extraction.SuffixTree import SuffixTree

class BaseNormalizer():
    def __init__(self):
        pass
    
    def normalize(self, out_vals):
        pass