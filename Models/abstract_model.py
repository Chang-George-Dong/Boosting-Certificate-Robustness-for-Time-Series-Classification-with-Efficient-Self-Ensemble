# import torch
# import torch.backends.cudnn as cudnn
# from archs.InceptionTime import Classifier_INCEPTION
# from archs.LSTM_FCN import LSTMFCN
# from archs.MACNN import MACNN
# from torch.nn.functional import interpolate
# from torch import nn
from CODE.Utils.package import *
from CODE.Utils.utils import get_method_loc
from CODE.Utils.constant import *

common_info = {
    "num_classes": None,
    "model_path": None,
    "x_len": None,
    # "base_model": None,  # 这里最好用已经实例化的模型，不然大量的实例化会爆显存。
    "base_model_name": None,
    "Aug_params": {
        "gaussian_noise": {"p": 0.9, "amplitude": 0.2, "seed": None},
        "Jitter": {"p": 0.5, "amplitude": 0.2, "seed": None},
    },
}
# 之所以要base_model和base_model_name同时存在，是为了尽可能将这部分代码抽样化，增强泛化性。


class Masked_Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        # 注意，这里的base_model已经实例化了
        self.MASK_layer_list = []
        for key, value in kwargs["Aug_params"].items():
            self.MASK_layer_list.append((Aug_Methods_Dict[key], value))

    def forward(self, x):
        for layer, kwargs in self.MASK_layer_list:
            x = layer(x, **kwargs)
        return self.base_model(x)

    def save_weights(self, model_path=None):
        """
        Save the model weights to the specified path.

        Args:
            model_path (str): Path to save the model weights.
        """
        # Save the weights of the base model
        model_path = self.model_path if model_path is None else model_path
        torch.save(self.base_model.state_dict(), model_path)
        print(f"Weights saved successfully to {model_path}")

    def load_weights(self, load_path=None):
        """
        Load the model weights from the specified path.

        Args:
            load_path (str): Path to load the model weights.
        """
        model_path = self.model_path if load_path is None else load_path
        # Load the weights into the base model
        self.base_model.load_state_dict(torch.load(load_path))
        print(f"Weights loaded successfully from {load_path}")


class Ensemble_Model(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        # self.__class__.__name__ = "Ensemble_Model"
        self.__name__ = "Ensemble_Model"
        for key, value in kwargs.items():
            setattr(self, key, value)
        # self.rate_mode = rate_mode
        # self.common_info_list = common_info_list
        self.common_info_updater()

        self.base_model_pool = nn.ModuleDict()
        self.mask_model_pool = nn.ModuleList()
        for common_info in self.common_info_list:
            base_model = self.model_pool_update(common_info)
            common_info["base_model"] = base_model
            self.mask_model_pool.append(Masked_Model(**common_info))
        if self.rate_mode == 0:
            self.forward = self.forward_soft
        elif self.rate_mode == 1:
            self.forward = self.forward_hard

    def common_info_updater(self):
        for i_model_params in self.common_info_list:
            i_model_params["dataset"] = self.dataset
            i_model_params["device"] = self.device
            i_model_params["num_classes"] = self.num_classes
            i_model_params["seq_length"] = self.shape[-1]
            i_model_params["input_shape"] = self.shape
            i_model_params["state"] = self.state
            i_model_params["base_model"] = BASE_MODEL_DICT[
                i_model_params["base_model_name"]
            ]
            i_model_params["model_path"] = get_method_loc(
                [i_model_params["base_model_name"], i_model_params["Aug_params"]],
            )

    def model_pool_update(self, common_info):
        model_name = common_info["base_model_name"]
        model_weight_path = common_info["model_path"]
        model_info = model_name + model_weight_path
        if model_info not in self.base_model_pool:
            self.base_model_pool[model_info] = BASE_MODEL_DICT[
                common_info["base_model_name"]
            ](**common_info)
        return self.base_model_pool[model_info]

    def forward_soft(self, x):
        class_votes = torch.zeros((x.size(0), self.num_classes), device=x.device)
        for mask_model in self.mask_model_pool:
            class_votes += mask_model(x)
        return class_votes

    def forward_hard(self, x):
        class_votes = torch.zeros((x.size(0), self.num_classes), device=x.device)
        for mask_model in self.mask_model_pool:
            predicted_classes = torch.argmax(mask_model(x), dim=1)
            one_hot_predictions = F.one_hot(predicted_classes, num_classes=self.num_classes)
            class_votes += one_hot_predictions.to(class_votes.dtype).to(class_votes.device)
        return class_votes

    def save_weights(self, out_dir=None):
        """
        Save the ensemble model weights to the specified directory.

        Args:
            out_dir (str): Directory to save the model weights. Defaults to self.out_dir.
        """
        _ = os.path.join(self.out_dir, "pth_files")
        out_dir = _ if out_dir is None else out_dir

        # Ensure the output directory exists
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for mask_model, common_info in zip(self.mask_model_pool, self.common_info_list):
            model_file_name = os.path.basename(common_info["model_path"]) + ".pth"
            save_path = os.path.join(out_dir, model_file_name)

            # Save the weights for the Masked_Model's base model
            mask_model.save_weights(save_path)
            print(f"Weights saved for model '{model_file_name}' at: {save_path}")

    def load_weights(self, model_paths=None):
        """
        Load the ensemble model weights from the specified paths.

        Args:
            model_paths (list of str): List of paths to load the model weights from.
                                    Defaults to paths in self.common_info_list.
        """
        if model_paths is None:
            # Use default model paths from common_info_list
            model_paths = [
                common_info["model_path"] for common_info in self.common_info_list
            ]

        if len(model_paths) != len(self.mask_model_pool):
            raise ValueError(
                f"Expected {len(self.mask_model_pool)} model paths but got {len(model_paths)}"
            )

        for mask_model, load_path in zip(self.mask_model_pool, model_paths):
            # Load the weights for each Masked_Model
            _ = os.path.join(
                TRAIN_OUTPUT_PATH,
                f"{self.__name__}_{load_path['model_name']}",
                load_path["aug_name"],
                load_path["dataset"],
                "pth_files",
                f"{load_path['model_name']}_{load_path['aug_name']}=seed=None_{load_path['aug_name']}=seed=None.pth",
            )
            mask_model.load_weights(_)
            print(f"Weights loaded from: {load_path}")


class Smmothed_Model(Ensemble_Model):
    def __init__(self, sigma=0.8, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.smooth_layer = Aug_Methods_Dict["gaussian_noise"]
        self.__name__ = "Smmothed_Model"

    def forward(self, x):
        x = self.smooth_layer(x, std=self.sigma)
        super(Ensemble_Model).forward(x)


def common_info_list_generator(
    base_model_list, mask_type_list, num_classes, x_len, seed
):
    common_info_list = []
    for base_model in base_model_list:
        for mask_type in mask_type_list:
            common_info["num_classes"] = num_classes
            common_info["mask_type"] = mask_type
            common_info["x_len"] = x_len
            common_info["base_model"] = base_model  # 后面再改
            common_info["base_model_name"] = base_model.__name__
            common_info["seed"] = seed
            common_info_list.append(common_info.copy())
    return common_info_list
