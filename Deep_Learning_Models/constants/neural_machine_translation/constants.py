import os

BASE_PATH=os.environ['BASE_PATH']

DATA_PATH=os.path.join(BASE_PATH, "data", "attribute_extraction", "color_category")
PERSISTENCE_PATH=os.path.join(BASE_PATH, "persistence", "neural_machine_translation")
OUTPUT_PATH=os.path.join(BASE_PATH, "outputs", "neural_machine_translation")


DATASET_FILEPATH=os.path.join(DATA_PATH, "out_urls.csv")
NUM_EXAMPLES=20000
BATCH_SIZE=64

ENCODER_EMB_DIM=128
DECODER_EMB_DIM=128

ENCODER_UNITS=128
DECODER_UNITS=128

NUM_EPOCHS=50
MODEL_PATH=os.path.join(PERSISTENCE_PATH, "nmt_model.h5")

SRC_TENSOR_TRAIN=os.path.join(PERSISTENCE_PATH, "src_tensor_train.pkl")
TRG_TENSOR_TRAIN=os.path.join(PERSISTENCE_PATH, "trg_tensor_train.pkl")

SRC_TENSOR_VALID=os.path.join(PERSISTENCE_PATH, "src_tensor_valid.pkl")
TRG_TENSOR_VALID=os.path.join(PERSISTENCE_PATH, "trg_tensor_valid.pkl")

SRC_TENSOR=os.path.join(PERSISTENCE_PATH, "src_tensor.pkl")
TRG_TENSOR=os.path.join(PERSISTENCE_PATH, "trg_tensor.pkl")

SRC_LANG=os.path.join(PERSISTENCE_PATH, "src_lang.pkl")
TRG_LANG=os.path.join(PERSISTENCE_PATH, "trg_lang.pkl")