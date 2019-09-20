import os

BASE_PATH=os.environ['BASE_PATH']
ATTRIBUTE_NAME=os.environ['ATTRIBUTE']

DATA_PATH=os.path.join(BASE_PATH, "data", "attribute_extraction", ATTRIBUTE_NAME)
PERSISTENCE_PATH=os.path.join(BASE_PATH, "persistence", "attribute_extraction", ATTRIBUTE_NAME)
OUTPUT_PATH=os.path.join(BASE_PATH, "outputs", "attribute_extraction", ATTRIBUTE_NAME)

IMAGE_ARRAY_PATH=os.path.join(DATA_PATH, "image_array.h5")
TEXT_ARRAY_PATH=os.path.join(DATA_PATH, "text_array.h5")

URLS_LIST_PATH=os.path.join(DATA_PATH, "urls_list.txt")
LABELS_PATH=os.path.join(PERSISTENCE_PATH, "labels.pkl")
TRANSFORMED_LABELS_PATH=os.path.join(PERSISTENCE_PATH, "transformed_labels.pkl")

TRAIN_INDICES_PATH=os.path.join(PERSISTENCE_PATH, "train_indices.pkl")
TEST_INDICES_PATH=os.path.join(PERSISTENCE_PATH, "test_indices.pkl")
INPUT_FILE_PATH=os.path.join(DATA_PATH, "input_data.csv")

DOWNLOADED_IMAGES_PATH=os.path.join(DATA_PATH, "downloaded_images")
OUTPUT_FILE_PATH=os.path.join(DATA_PATH, "out_urls.csv")
ENCODER_PATH=os.path.join(PERSISTENCE_PATH, "encoder.pkl")
MODEL_PATH=os.path.join(PERSISTENCE_PATH, "model.h5")
PREDS_PATH=os.path.join(OUTPUT_PATH, "predictions")
CAMS_PATH=os.path.join(OUTPUT_PATH, "cams")
TOKENS_PATH=os.path.join(DATA_PATH, "word_tokens.npy")

BATCH_SIZE=64
NUM_EPOCHS=200
IMAGE_SIZE=128
MAX_FEATURES=500

VECTOR_MODEL='fasttext'
MAX_WORDS=480
WORD_VECTOR_DIM=128

FAST_TEXT_PATH_WORD=os.path.join(PERSISTENCE_PATH, "fasttext_word.pkl")
FAST_TEXT_PATH_CHAR=os.path.join(PERSISTENCE_PATH, "fasttext_char.pkl")
WORD_VECT_PATH_WORD=os.path.join(PERSISTENCE_PATH, "word2vec_word.pkl")
WORD_VECT_PATH_CHAR=os.path.join(PERSISTENCE_PATH, "word2vec_char.pkl")

INPUT_TENSOR_PATH=os.path.join(PERSISTENCE_PATH, "input_tensor.pkl")
TENSOR_TOKENIZER_PATH=os.path.join(PERSISTENCE_PATH, "tensor_tokenizer.pkl")
VOCAB_SIZE_PATH=os.path.join(PERSISTENCE_PATH, "vocab_size.pkl")

USE_NUM_GPUS=1
TF_TRAIN_SUMMARY_PATH=os.path.join(OUTPUT_PATH, "model_summary", "train")
TF_TEST_SUMMARY_PATH=os.path.join(OUTPUT_PATH, "model_summary", "test")
IMAGE_NUM_FILTERS = [32, 64, 128, 256, 512]
TEXT_NUM_FILTERS = [32, 64, 128, 256, 512]
LOAD_SAVED_GRAPH = False
USE_TRANSFER_LEARNING_IMAGE=True
SAVE_BEST_LOSS_MODEL=False
IS_MULTILABEL=True