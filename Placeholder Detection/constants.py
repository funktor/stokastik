import os
base_dir = os.path.dirname(os.path.realpath(__file__))

IMAGE_SIZE=64
SIAMESE_BATCH_SIZE=64
TRAINING_SAMPLES_PER_EPOCH=25600
VALIDATION_SAMPLES_PER_EPOCH=12800
TESTING_SAMPLES_PER_EPOCH=12800
NUM_SAMPLES_PER_PT=1500
EMBEDDING_SIZE=128
EMBEDDING_BATCH_SIZE=512
PLACEHOLDER_THRESHOLD=0.01
SIAMESE_NUM_EPOCHS=200
PHASH_MATCHING_THRESHOLD=15

TRAIN_IMAGE_DATA_FILE="train_image_data.npy"
TRAIN_URL_PT_MAP_FILE="train_url_pt_map.pkl"
TRAIN_PT_URL_MAP_FILE="train_pt_url_map.pkl"
TRAIN_PLACEHOLDERS_FILE="training_placeholders.npy"

TEST_IMAGE_DATA_FILE="test_image_data.npy"
TEST_URL_PT_MAP_FILE="test_url_pt_map.pkl"
TEST_PT_URL_MAP_FILE="test_pt_url_map.pkl"
TEST_PLACEHOLDERS_FILE="testing_placeholders.npy"

VALID_IMAGE_DATA_FILE="valid_image_data.npy"
VALID_URL_PT_MAP_FILE="valid_url_pt_map.pkl"
VALID_PT_URL_MAP_FILE="valid_pt_url_map.pkl"

DATA_DIR=os.path.join(base_dir, 'data')
SIAMESE_MODEL_PATH=os.path.join(DATA_DIR, 'siamese_model.h5')
SIAMESE_BEST_MODEL_PATH=os.path.join(DATA_DIR, 'siamese_current_best.h5')
USE_VGG=False
USE_BEST_VAL_LOSS_MODEL=True
VOTING_PTS_PATH='voting_pts.pkl'
VOTING_KD_TREE_PATH='voting_kd_tree.pkl'
KD_TREE_LEAF_SIZE=20
WMT_PAYLOAD={'odnHeight': 300,'odnWidth': 300,'odnBg': 'FFFFFF'}
WMT_HOSTS=["i5.walmartimages.com", "ll-us-i5.wal.co"]
WMT_MAIN_HOST="i5.walmartimages.com"
HYNDL_HOST="images.hayneedle.com"
HYNDL_PAYLOAD={"is" : "300,300"}

TAGGED_PLACEHOLDER_IMAGES_PATH=os.path.join(DATA_DIR, 'tagged_placeholder_images')
PRODUCT_IMAGES_PATH=os.path.join(DATA_DIR, 'product_images')
TEST_PLACEHOLDER_IMAGES_PATH=os.path.join(DATA_DIR, 'generated_placeholder_images')
HASH_LIST_PATH='hash_list.npy'

