import sys

sys.path.append("/home/david/Project/ADC2024/Project")
from CODE import *
from CODE.Utils.constant import (
    Aug_Methods_Dict,
    DEVICE,
    DATASETS,
    BASE_MODEL_DICT,
)
from CODE.Models.abstract_model import *
from CODE.Trainer import Trainer

silent(True)


def run(dataset, override):
    i = 0
    for model_name in BASE_MODEL_DICT.keys():
        for aug_name in Aug_Methods_Dict.keys():
            i = i + 1
            if i % 2 == 0:
                continue
            batch_size = 128
            while True:
                try:
                    common_info = {
                        "model_path": None,
                        "base_model_name": model_name,
                        "Aug_params": {
                            aug_name: {"seed": None},
                        },
                    }
                    common_info_list = [
                        common_info,
                    ]

                    train_params = {
                        "batch_size": batch_size,
                        "dataset": dataset,
                        "device": DEVICE,
                        "epochs": 1000,
                        "override": True,
                        "model_params": {
                            "sigma": 0.8,
                            "common_info_list": common_info_list,
                            "rate_mode": 0,
                        },
                        "model": Smmothed_Model,
                        "path_parameter": os.path.join(model_name, aug_name),
                    }

                    trainer = Trainer(**train_params)

                    trainer.train_and_evaluate(
                        override=override,
                        to_device=True,
                    )
                    break
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logging.debug(str(e))
                        torch.cuda.empty_cache()
                        time.sleep(1)
                        batch_size = int(batch_size - 32)
                        if batch_size < 32:
                            raise RuntimeError(str(e))
                    elif (
                        "cudnn RNN backward can only be called in training mode"
                        in str(e)
                    ):
                        logging.warning(str(e))
                        torch.backends.cudnn.enabled = False
                    else:
                        raise RuntimeError(str(e))


for dataset in selected_dataset:
    run(dataset, override=False)


for dataset in DATASETS:
    if not dataset in selected_dataset:
        run(dataset, override=False)
