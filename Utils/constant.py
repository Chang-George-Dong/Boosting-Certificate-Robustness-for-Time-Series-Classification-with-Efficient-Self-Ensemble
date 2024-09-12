from CODE.Utils.package import *

# from CODE.Utils.utils import setup_logging, silent
from CODE.Utils.augmentation import Augmentation as aug

from CODE.Models.TS_2_V.main import Classifier_TS2V
from CODE.Models.InceptionTime import Classifier_INCEPTION
from CODE.Models.LSTM_FCN import Classifier_LSTMFCN
from CODE.Models.MACNN import Classifier_MACNN
from CODE.Models.ResNet import Classifier_ResNet18


# Set up logging
setup_logging()
# logger = logging

if not os.getcwd() == HOME_LOC:
    logging.warning("Home path not equal to work path, changing!")
    os.chdir(HOME_LOC)

sys.path.append(HOME_LOC)


DATASET_PATH = os.path.join(HOME_LOC, "Dataset", "UCRArchive_2018")
ATTACK_OUTPUT_PATH = os.path.join(HOME_LOC, "OUTPUT", "attack")
TRAIN_OUTPUT_PATH = os.path.join(HOME_LOC, "OUTPUT", "train")
EVAL_OUTPUT_PATH = os.path.join(HOME_LOC, "OUTPUT", "eval")
MODEL_NAME = "MODEL_INFO.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Aug_Methods_Dict = aug.get_method()
AUG_METHODS_DICT = Aug_Methods_Dict

BASE_MODEL_DICT = {
    "TS2V": Classifier_TS2V,
    "Inception": Classifier_INCEPTION,
    "LstmFCN": Classifier_LSTMFCN,
    "MACNN": Classifier_MACNN,
    "ResNet": Classifier_ResNet18,
}

epochs = 1000
batch_size = 100

sigmas = [0, 0.1, 0.2, 0.4, 0.8, 1.6]

selected_dataset = [
    "ChlorineConcentration",
    "SyntheticControl",
    "CBF",
    "CricketX",
    "CricketY",
    "CricketZ",
]

DATASETS = json.load(open(os.path.join(HOME_LOC, "Config", "dataset.json")))
