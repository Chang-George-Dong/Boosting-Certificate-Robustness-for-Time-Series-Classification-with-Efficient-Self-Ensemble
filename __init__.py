__version__ = "3.0.0"


###########################################################
# Warning, Do not change the order of following packages. #
###########################################################

# try:
from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *

# from CODE.Utils.constant import UNIVARIATE_DATASET_NAMES as datasets

from CODE.Utils.augmentation import Augmentation

# from CODE.Attack.attacker import Attack
# from CODE.Train.trainer import Trainer

# except ModuleNotFoundError as e:
#     logging.ERROR(e)
#     sys.exit(-1)
