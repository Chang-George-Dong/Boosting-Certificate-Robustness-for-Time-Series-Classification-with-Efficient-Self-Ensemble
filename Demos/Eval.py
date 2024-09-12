import sys

sys.path.append("/home/david/Project/ADC2024/Project")
from CODE import *
from CODE.Models.abstract_model import *

Aug_comb = generate_combinations(Aug_Methods_Dict.keys())
silent(True)
# #认证的时候seed不能为none
for dataset in selected_dataset:
    for model_name in BASE_MODEL_DICT.keys():
        for aug_name in AUG_METHODS_DICT.keys():
            batch_size = 128
            while True:
                try:
                    common_info_list = []
                    model_paths = []
                    for i in range(5):

                        common_info_list.append(
                            {
                                "model_path": None,
                                "base_model_name": model_name,
                                "Aug_params": {
                                    aug_name: {"seed": i},
                                },
                            }
                        )

                        model_paths.append(
                            {
                                "model_name": model_name,
                                "aug_name": aug_name,
                                "dataset": dataset,
                            }
                        )

                    attack_params = {
                        "batch_size": batch_size,
                        "dataset": dataset,
                        "device": DEVICE,
                        "epochs": 100,
                        "override": True,
                        "model_params": {
                            "common_info_list": common_info_list,
                            "rate_mode": 0,
                        },
                        "model": Smmothed_Model,
                        "path_parameter": os.path.join(model_name, aug_name),
                        "attack_method_path": "PGD5",
                    }

                    from CODE.Attacker import PGD

                    attacker = PGD(**attack_params)
                    attacker.model.load_weights(model_paths)
                    attacker.perturb_all(override=False, to_device=True)
                    break

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logging.debug(str(e))
                        torch.cuda.empty_cache()
                        time.sleep(1)
                        batch_size = int(batch_size * 0.8)
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
# 单模型的已经成功攻击
