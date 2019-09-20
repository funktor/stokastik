import os, sys

VECTOR_MODEL='wv'
BASE_PATH=os.environ['BASE_PATH']
DATA_PATH=os.path.join(BASE_PATH, "data", "deep_matching")
PERSISTENCE_PATH=os.path.join(BASE_PATH, "persistence", "deep_matching")
OUTPUT_PATH=os.path.join(BASE_PATH, "outputs", "color_extraction")

TRAIN_DATA_FILE_PATH=os.path.join(DATA_PATH, "cia_data.csv")
TEST_DATA_FILE_PATH=os.path.join(DATA_PATH, "cia_data_test.csv")
MODEL_PATH=os.path.join(PERSISTENCE_PATH, "model.h5")
MAX_WORDS=70
MAX_CHARS=10
BATCH_SIZE=256
WORD_VECTOR_DIM=128
CHAR_VECTOR_DIM=128
NUM_EPOCHS=50
USE_VALIDATION=False

FAST_TEXT_PATH_WORD=os.path.join(PERSISTENCE_PATH, "fasttext_word.pkl")
FAST_TEXT_PATH_CHAR=os.path.join(PERSISTENCE_PATH, "fasttext_char.pkl")
WORD_VECT_PATH_WORD=os.path.join(PERSISTENCE_PATH, "word2vec_word.pkl")
WORD_VECT_PATH_CHAR=os.path.join(PERSISTENCE_PATH, "word2vec_char.pkl")

USE_NUM_GPUS=4
TF_TRAIN_SUMMARY_PATH=os.path.join(OUTPUT_PATH, "model_summary", "train")
TF_TEST_SUMMARY_PATH=os.path.join(OUTPUT_PATH, "model_summary", "test")
LOAD_SAVED_GRAPH = False