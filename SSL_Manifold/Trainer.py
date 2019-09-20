import os, sys, ast
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import os, re, math, random
import shutil, copy

from scipy.sparse import hstack
from scipy.sparse import coo_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, classification_report
from sklearn.gaussian_process.kernels import RBF

import numpy as np, os, re, math, random
import pandas as pd
from scipy import sparse
from gensim.models import Word2Vec, FastText
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import DNNetwork as dnet
from DNNetwork import DNN
import Constants as cnt
import Utilities as utils
from classifiers.LaplacianMRClassifier import LaplacianMRClassifier
from feature_transformers.PCA_Transformer import PCAFeatures
from feature_transformers.W2V_Transformer import W2VFeatures 
from feature_transformers.DNN_Transformer import DNNFeatures
from feature_transformers.GLV_Transformer import GLVFeatures 


class Trainer():
    def __init__(self, user_inputs):
        self.x_train_l = None
        self.x_train_u = None
        self.x_test = None
        self.class_labels_train = None
        self.class_labels_test = None
        self.best_estimator = None
        self.model = None
        self.size_data=0
        self.label_transformer = None
        self.title_features = None
        self.description_features = None
        self.pca_titles = None
        self.pca_descriptions = None
        self.dnn_tokenizer = None
        self.dnn = None
        self.feature_set = None
        
        self.attribute = user_inputs['attribute'] if 'attribute' in user_inputs else 'fit'
        self.num_l = user_inputs['num_l'] if 'num_l' in user_inputs else 1000
        self.num_u = user_inputs['num_u'] if 'num_u' in user_inputs else 500
        
        self.train_data_file_l = os.path.join(cnt.DATA_PATH, self.attribute + '_' + str(self.num_l) + '.csv')
        self.train_data_file_u = os.path.join(cnt.DATA_PATH, self.attribute + '_' + str(self.num_u) + '.csv')
        self.test_data_file = os.path.join(cnt.DATA_PATH, self.attribute + '_test.csv')
        
        self.num_features_title = user_inputs['num_features_title'] if 'num_features_title' in user_inputs else None
        self.num_features_desc = user_inputs['num_features_desc'] if 'num_features_desc' in user_inputs else None
        
        self.none_trainable = user_inputs['none_trainable'] if 'none_trainable' in user_inputs else True
        self.filter_labels_less_than = user_inputs['filter_labels_less_than'] if 'filter_labels_less_than' in user_inputs else 0
        
        self.use_pca = user_inputs['use_pca'] if 'use_pca' in user_inputs else False
        self.use_wv_embeds = user_inputs['use_wv_embeds'] if 'use_wv_embeds' in user_inputs else False
        self.use_dnn = user_inputs['use_dnn'] if 'use_dnn' in user_inputs else False
        self.use_glv = user_inputs['use_glv'] if 'use_glv' in user_inputs else False
        
        self.wv_embed_type = user_inputs['wv_embed_type'] if 'wv_embed_type' in user_inputs else 'W2V'
        self.max_words = user_inputs['max_words'] if 'max_words' in user_inputs else 200
        self.batch_size = user_inputs['batch_size'] if 'batch_size' in user_inputs else 64
        self.epochs = user_inputs['epochs'] if 'epochs' in user_inputs else 5
        
        self.dnn_model_path = os.path.join(cnt.PERSISTENCE_PATH, self.attribute + '_' + str(self.num_l) + '_DNN_MODEL.pkl')
        self.glove_path = os.path.join(cnt.PERSISTENCE_PATH, 'glove.6B.300d.txt')
        
        self.feature_title_path = os.path.join(cnt.PERSISTENCE_PATH, self.attribute + '_' + str(self.num_l) + '_TITLE.pkl')
        self.feature_desc_path = os.path.join(cnt.PERSISTENCE_PATH, self.attribute + '_' + str(self.num_l) + '_DESC.pkl')
        self.label_transformer_path = os.path.join(cnt.PERSISTENCE_PATH, self.attribute + '_' + str(self.num_l) + '_LABEL_TRANSFORMER.pkl')
        
        self.embed_dim_title = user_inputs['embed_dim_title'] if 'embed_dim_title' in user_inputs else 128
        self.embed_dim_desc = user_inputs['embed_dim_desc'] if 'embed_dim_desc' in user_inputs else 128
        
        self.use_unlabelled_features = user_inputs['use_unlabelled_features'] if 'use_unlabelled_features' in user_inputs else True
        self.use_unlabelled_training = user_inputs['use_unlabelled_training'] if 'use_unlabelled_training' in user_inputs else True
        
        
    def __read_data_file(self, file_path, stage='labeled'):
        df = pd.read_csv(file_path)

        df['Title'].fillna('', inplace=True)
        df['Long Description'].fillna('', inplace=True)
        df['Short Description'].fillna('', inplace=True)
        df['Manual Curation Value'].replace(to_replace='none', value='None')
        df['Manual Curation Value'].fillna('None', inplace=True)
        
        if self.none_trainable is False and stage == 'labeled':
            df = df.loc[df['Manual Curation Value'] != 'None']
        
        if self.filter_labels_less_than > 0 and stage == 'labeled':
            df = df.groupby('Manual Curation Value').filter(lambda x : len(x)>self.filter_labels_less_than)

        df['Short Description'].apply(str)
        df['Long Description'].apply(str)
        df['Title'].apply(str)

        description_corpus = df['Short Description'] + " " + df['Long Description']
        title_corpus = df['Title']
        
        class_labels_corpus = df['Manual Curation Value'].apply(lambda x: x.strip().split('__'))
        
        return title_corpus, description_corpus, class_labels_corpus
    
    
    def create_train_test_data(self):
        print("Reading datasets")
        title_corpus, description_corpus, class_labels_train = self.__read_data_file(self.train_data_file_l, stage='labeled')
        title_corpus_un, description_corpus_un, _ = self.__read_data_file(self.train_data_file_u, stage='unlabeled')
        titles_test, descriptions_test, class_labels_test = self.__read_data_file(self.test_data_file, stage='labeled')
        
        print("Getting class labels")
        self.label_transformer = MultiLabelBinarizer()
        class_labels_full = pd.concat([class_labels_train, class_labels_test], axis=0)
        self.label_transformer.fit(class_labels_full)
        
        utils.save_data_pkl(self.label_transformer, self.label_transformer_path)
        
        self.class_names = self.label_transformer.classes_
        
        title_corpus_l = list(title_corpus)
        description_corpus_l = list(description_corpus)
        
        title_corpus_u = list(title_corpus_un)
        description_corpus_u = list(description_corpus_un)
        
        title_corpus_test = list(titles_test)
        description_corpus_test = list(descriptions_test)
        
        if os.path.exists(self.feature_title_path) is False:
            if self.num_features_title is not None:
                print("Getting title features MI")
                self.title_features = utils.get_features_mi(title_corpus_l, list(class_labels_train), self.num_features_title, 1, 1)

            if self.num_features_desc is not None:
                print("Getting desc features MI")
                self.description_features = utils.get_features_mi(description_corpus_l, list(class_labels_train), self.num_features_desc, 1, 1)

            if self.use_dnn:
                print("Training DNN network")
                features = list(set(self.title_features + self.description_features))

                self.feature_tf = DNNFeatures(model_path=self.dnn_model_path, embed_dim=self.embed_dim_title, 
                                              max_words=self.max_words, batch_size=self.batch_size, 
                                              num_epochs=self.epochs, features=list(set(self.title_features + self.description_features)))

                class_labels = np.asarray(class_labels_train)

                corpus = [title_corpus_l[i] + " " + description_corpus_l[i] for i in range(len(title_corpus_l))]

                print("Training DNN")
                self.feature_tf.fit(corpus, class_labels)
                self.feature_tf_title, self.feature_tf_desc = copy.deepcopy(self.feature_tf), copy.deepcopy(self.feature_tf)

            else:
                if self.use_pca:
                    print("Training PCA")
                    self.feature_tf_title = PCAFeatures(num_components=self.embed_dim_title, features=self.title_features)
                    self.feature_tf_desc = PCAFeatures(num_components=self.embed_dim_desc, features=self.description_features)
                
                elif self.use_glv:
                    print("Loading Glove")
                    self.feature_tf_title = GLVFeatures(num_components=300, features=self.title_features, glove_path=self.glove_path)
                    self.feature_tf_desc = GLVFeatures(num_components=300, features=self.description_features, glove_path=self.glove_path)

                else:
                    print("Training word2vec")
                    self.feature_tf_title = W2VFeatures(num_components=self.embed_dim_title, features=self.title_features, wv_type=self.wv_embed_type)
                    self.feature_tf_desc = W2VFeatures(num_components=self.embed_dim_desc, features=self.description_features, wv_type=self.wv_embed_type)

                if self.use_unlabelled_features:
                    self.feature_tf_title.fit(title_corpus_l + title_corpus_u)
                    self.feature_tf_desc.fit(description_corpus_l + description_corpus_u)

                else:
                    self.feature_tf_title.fit(title_corpus_l)
                    self.feature_tf_desc.fit(description_corpus_l)

            utils.save_data_pkl(self.feature_tf_title, self.feature_title_path)
            utils.save_data_pkl(self.feature_tf_desc, self.feature_desc_path)
        
        else:
            self.feature_tf_title = utils.load_data_pkl(self.feature_title_path)
            self.feature_tf_desc = utils.load_data_pkl(self.feature_desc_path)

        x_train_title_l = self.feature_tf_title.transform(title_corpus_l)
        x_train_desc_l = self.feature_tf_desc.transform(description_corpus_l)

        self.x_train_l = np.hstack((x_train_title_l, x_train_desc_l))
        
        if self.use_unlabelled_training:
            x_train_title_u = self.feature_tf_title.transform(title_corpus_u)
            x_train_desc_u = self.feature_tf_desc.transform(description_corpus_u)

            self.x_train_u = np.hstack((x_train_title_u, x_train_desc_u))

        x_test_title = self.feature_tf_title.transform(title_corpus_test)
        x_test_desc = self.feature_tf_desc.transform(description_corpus_test)

        self.x_test = np.hstack((x_test_title, x_test_desc))
        
        self.class_labels_train = self.label_transformer.transform(np.asarray(class_labels_train))
        self.class_labels_test = self.label_transformer.transform(np.asarray(class_labels_test))
        

    def train(self):
        self.create_train_test_data()
        
if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()