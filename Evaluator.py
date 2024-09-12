from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *
from CODE.Trainer import Trainer


class Evaluator(object):
    """A smoothed classifier g"""

    def __init__(
        self,
        # base_classifier: torch.nn.Module,
        # num_classes: int,
        # sigma: float,
        # seq_length=None,
        **kwargs,
    ):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.config = kwargs
        (
            self.train_loader,
            self.model_params["shape"],
            self.model_params["num_classes"],
            self.class_weights,
            self.x_tensor,
            self.y_tensor,
        ) = load_data(
            self.dataset,
            phase="TEST",
            batch_size=self.batch_size,
            data_path=DATASET_PATH,
            eval=True,
        )
        self.model_params["out_dir"] = self.__set_output_dir__()
        self.state = kwargs.get("state", "train")
        all_attrs = vars(self)
        for key, value in all_attrs.items():
            self.model_params[key] = value
        self.model = self.model(**self.model_params)
        self.model.to(self.device)

        kwargs["state"] = "attack"
        # trainer = Trainer(init_model_only=True, eval=True**kwargs)
        # all_attrs = vars(trainer)
        # for key, value in all_attrs.items():
        #     setattr(self, key, value)
        self.base_classifier = self.model.to(self.device)
        self.seq_length = self.model_params["shape"][0]
        self.num_classes = self.model_params["num_classes"]
        self.sigma = kwargs["sigma"]
        self.out_dir = os.path.join(
            EVAL_OUTPUT_PATH,
            self.eval_method_path,
            self.dataset,
            self.path_parameter,
        )

    def __set_output_dir__(self):
        self.method_path = self.model.__name__ + "_"
        self.method_path += self.path_parameter
        self.out_dir = os.path.join(
            TRAIN_OUTPUT_PATH,
            self.method_path,
            self.dataset,
        )
        return self.out_dir

    def certify(self, x: torch.tensor, n: int, alpha: float, batch_size: int):
        self.base_classifier.eval()
        counts_selection = self._sample_noise(x, n, batch_size)
        pA = torch.argmax(counts_selection, dim=-1)
        bound = self.bound(counts_selection, self.sigma, alpha)
        return pA.cpu().numpy().item(), bound.cpu().numpy().item()

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int):
        self.base_classifier.eval()
        counts_selection = self._sample_noise(x, n, batch_size)
        pA = torch.argmax(counts_selection, dim=-1)
        return pA.cpu().numpy().item()

    def bound(self, prob, sd, alpha):
        # 在CPU上计算置信区间
        sorted_indices = np.sort(prob.cpu().numpy())[::-1]
        fv = FloatVector(sorted_indices)
        ci = np.array(multici.multinomialCI(fv, 0.05))

        # 将结果移动到GPU上
        qi = torch.tensor(ci[0, 0], device=prob.device)
        qj = torch.tensor(ci[1, 1], device=prob.device)
        alpha = torch.linspace(1.01, 2, 100, device=prob.device)
        # 在GPU上计算鲁棒性上界
        qi_term = qi ** (1 - alpha)
        qj_term = qj ** (1 - alpha)
        bound_term = 1 - qi - qj + 2 * ((qi_term + qj_term) / 2) ** (1 / (1 - alpha))
        bound = (-torch.log(bound_term) / alpha).max()
        return torch.sqrt(bound * 2.0 * sd**2)

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> torch.Tensor:
        """Sample the base classifier's prediction under noisy corruptions of the input x.
        :param x: the input [batch x channel x height x width]
        :param num: number of samples to collect
        :param batch_size:
        :return: a torch.Tensor of length num_classes containing the per-class counts
        """
        x = x.to(self.device)
        counts = torch.zeros(self.num_classes, dtype=torch.int64, device=self.device)
        num_batches = math.ceil(num / batch_size)
        for _ in range(num_batches):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size
            batch = x.repeat((this_batch_size, 1, 1))
            noise = torch.randn_like(batch, device=self.device) * self.sigma
            predictions = self.base_classifier(batch + noise).argmax(1)
            counts = counts.index_add_(
                0, predictions, torch.ones_like(predictions, device=self.device)
            )
        return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def evaluate(self, batch_size, override=False, to_device=False):
        res_path = os.path.join(self.out_dir, f"sigma_{self.sigma}_results.csv")
        _ = folder_contains_files(
            self.out_dir,
            f"sigma_{self.sigma}_results.csv",
        )
        if to_device and (not override) and _:
            logging.info(f"{res_path} exist,skip!")
            load_data_from_csv(self)
            return
        self.certify_all(batch_size, res_path)
        logging.info(f"{res_path} finished!")

    def certify_all(self, batch_size, res_path):

        # batch_size 不是 self.batch_size
        # 这里的 batch_size 是用于 certify 方法的参数
        self.batch_size = 1
        create_directory(self.out_dir)
        with open(res_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "label", "predict", "radius", "correct", "time"])

            # for batch_id, (x, y) in enumerate(self.test_loader):
            #     before_time = time.time()

            #     # 检查 x 的维度，并确保仅移除批次维度
            #     # if x.dim() == 4 and x.size(0) == 1:
            #     x = x.squeeze(0)  # 仅移除 batch 维度
            #     y = y.item()

            for i in range(len(self.x_tensor)):
                x, label = self.x_tensor[i], self.y_tensor[i]
                before_time = time.time()

                # 由于 batch_size=1，直接调用 certify 方法
                prediction, radius = self.certify(x, 1000, 1e-3, batch_size)

                after_time = time.time()
                correct = int(prediction == label)
                time_elapsed = after_time - before_time

                # 写入 CSV 文件
                writer.writerow(
                    [
                        i,
                        label,
                        prediction,
                        f"{radius:.3f}",
                        correct,
                        time_elapsed,
                    ]
                )
