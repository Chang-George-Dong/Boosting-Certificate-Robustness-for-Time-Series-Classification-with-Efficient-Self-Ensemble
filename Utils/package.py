def setup_logging():
    import datetime
    import logging
    import os

    hostname = socket.gethostname()
    username = getpass.getuser()
    current_datetime = datetime.datetime.now()
    formatted_log = current_datetime.strftime("%Yy_%mm_%dd_%Hh_%Mm_%Ss.log")
    formatted_log = f"{hostname}_{username}_" + formatted_log

    log_path = os.path.join(HOME_LOC, "LOG", formatted_log)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path)],
    )


def silent(silent):
    import builtins

    stack = inspect.stack()
    caller_info = stack[1]  # 获取调用print的那一层栈帧
    filename = caller_info.filename  # 文件名
    lineno = caller_info.lineno  # 行号
    location_info = f"[File: {filename}, Line: {lineno}]"

    # 保存原始的print函数
    original_print = builtins.print

    def silence_print(*args, **kwargs):
        logging.info(f"{location_info} - " + " ".join(map(str, args))[:200])

    def print_with_log(*args, **kwargs):
        original_print(f"{location_info} -", *args, **kwargs)
        logging.info(f"{location_info} - " + " ".join(map(str, args))[:200])

    if silent:
        builtins.print = silence_print
    else:
        builtins.print = print_with_log


try:
    import os
    import sys
    import re
    import csv
    import math
    import copy
    import json
    import time
    import shutil
    import logging
    import inspect
    import tempfile
    import socket
    import getpass
    import itertools
    import random

    import numpy as np
    import pandas as pd

    from pprint import pprint
    from datetime import datetime
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.utils.class_weight import compute_class_weight

    import torch
    import torch.optim as optim
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import TensorDataset, DataLoader
    import torch.nn.init as init

    import matplotlib.pyplot as plt
    from collections.abc import Iterable
    from collections import OrderedDict

    # from scipy.stats import norm, binom_test
    from statsmodels.stats.proportion import (
        proportion_confint,
        multinomial_proportions_confint,
    )
    from rpy2.robjects.packages import importr
    from rpy2.robjects import FloatVector
    from rpy2.robjects.vectors import StrVector
    import rpy2.robjects.packages as rpackages

    # 安装R包
    utils = rpackages.importr("utils")
    utils.install_packages(StrVector(["MultinomialCI"]))
    multici = importr("MultinomialCI")

    HOME_LOC = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(HOME_LOC)
    silent(False)


except ModuleNotFoundError as e:
    print(e)
    sys.exit(-1)
