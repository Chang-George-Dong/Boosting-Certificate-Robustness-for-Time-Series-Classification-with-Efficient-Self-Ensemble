from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *
from CODE.Trainer import Trainer

kwargs = {
    "dataset": "ACSF1",
    "model": "ResNet",
    "epoch": 100,
    "eps": 0.1,
    "device": "cuda:0",
}

# train_method_path=None,  # know train_method pth location
# path_parameter=None,  # know attack output location


# swap=None,
# swap_index=None,
# CW=None,
# c=None,
# eps_init=None,
# sign_only=None,
# alpha=None,
# make_demo=None,


class abstract_Attacker:
    def __init__(self, **kwargs):
        kwargs["state"] = "attack"
        trainer = Trainer(init_model_only=True, **kwargs)
        all_attrs = vars(trainer)
        for key, value in all_attrs.items():
            setattr(self, key, value)

        self.out_dir = os.path.join(
            ATTACK_OUTPUT_PATH,
            self.attack_method_path,
            self.dataset,
            self.path_parameter,
        )
        self.model.to(self.device)
        self.model.eval()

    def f(self, x):
        # If the `run` method is not defined, it dynamically assigns the `forward` method of the parent class to `run`.
        # This ensures that the model remains functional even if `run` is not explicitly defined,
        # thereby offering flexibility and a safeguard against method resolution issues in the class inheritance.
        x.to(self.device)
        try:
            return self.model.run(x)
        except AttributeError:
            self.model.run = self.model.forward
            return self.model(x)

    def perturb(self):
        logging.info("_" * 50)
        logging.info(f"Doing: {self.dataset}")
        start = time.time()
        all_perturbed_x = []
        all_perturbed_y = []
        all_predicted_y = []
        self.all_sum_losses = np.zeros(self.epochs)
        self.dist = []

        i = 1
        self.details = dict()
        # self.loader 改为 self.train_loader
        for batch_id, (x, y) in enumerate(self.train_loader):
            self.__batch_id__ = batch_id
            self.details[batch_id] = {"x": x.detach().cpu().numpy()}

            logging.debug(f"batch: {i}")
            logging.debug(">" * 50)
            perturbed_x, perturbed_y, predicted_y, sum_losses = self.__perturb__(x)
            perturbed_x = perturbed_x.detach().cpu().numpy()
            perturbed_x = np.squeeze(perturbed_x, axis=1)
            self.dist.extend(
                np.sum((perturbed_x - np.squeeze(x.numpy(), axis=1)) ** 2, axis=1)
            )
            all_perturbed_x.append(perturbed_x)
            perturbed_y = perturbed_y.detach().cpu().numpy()
            all_perturbed_y.append(perturbed_y)
            predicted_y = predicted_y.detach().cpu().numpy()
            all_predicted_y.append(predicted_y)

            self.all_sum_losses += sum_losses
            i += 1

        self.duration = time.time() - start
        self.x_perturb = np.vstack(all_perturbed_x)
        self.y_perturb = np.hstack(all_perturbed_y)
        self.y_predict = np.vstack(all_predicted_y).argmax(axis=1)

    def metrics(self):
        map_ = self.y_perturb != self.y_predict
        self.nb_samples = self.x_perturb.shape[0]

        Count_Success = sum(map_)
        Count_Fail = self.nb_samples - Count_Success
        ASR = Count_Success / self.nb_samples
        # distance = np.hstack(self.dist)
        distance = np.array(self.dist)
        success_distances = distance[map_]
        failure_distances = distance[~map_]

        # Create a dictionary with the data
        self.data = {
            "ASR": ASR,
            "mean_success_distance": np.mean(success_distances),
            "mean_failure_distance": np.mean(failure_distances),
            "overall_mean_distance": np.mean(distance),
            "median_success_distance": np.median(success_distances),
            "median_failure_distance": np.median(failure_distances),
            "overall_median_distance": np.median(distance),
            "Count_Success": Count_Success,
            "Count_Fail": Count_Fail,
            "duration": self.duration,
        }

    def perturb_all(self, override=False, to_device=False):
        _ = folder_contains_files(
            self.out_dir,
            "results.csv",
            "x_perturb.tsv",
            "y_perturb.npy",
            "loss.txt",
        )
        if to_device and (not override) and _:
            logging.info(f"Dataset: {self.dataset} exist, skip!")
            load_data_from_csv(self)
            return
        self.perturb()
        self.metrics()
        if to_device:
            create_directory(self.out_dir)
            save_perturb(self)
            save_conf_to_json(self.out_dir, self.finished_params)

    def plot_comparison(self, index, original_file=None, perturbed_file=None):
        perturbed_file = (
            os.path.join(self.out_dir, "x_perturb.tsv")
            if perturbed_file is None
            else perturbed_file
        )
        original_file = (
            os.path.join(DATASET_PATH, self.dataset, f"{self.dataset}_TEST.tsv")
            if original_file is None
            else original_file
        )

        # 读取数据
        original_data = pd.read_csv(original_file, sep="\t", header=None)
        perturbed_data = pd.read_csv(perturbed_file, sep="\t", header=None)

        # 获取特定索引的样本
        original_sample = original_data.iloc[index][
            1:
        ]  # 假设第一个元素是标签或其他非数据项
        original_sample = original_sample[1:].reset_index(drop=True)
        perturbed_sample = perturbed_data.iloc[index]

        # 绘制图形进行对比
        plt.figure(figsize=(12, 6))
        plt.title(f"Original vs Perturbed Sample (Index: {index})")
        plt.plot(original_sample, label="Original Sample")
        plt.plot(perturbed_sample, label="Perturbed Sample", linestyle="--")
        plt.legend()
        plt.show()


class abstract_attack_method:
    def __get_y_target_SWAP__(self, inputs):
        with torch.no_grad():
            # Compute predictions for inputs
            predictions = self.f(inputs)
            # Get the indices of the top classes for swapping
            _, top_class_indices = torch.topk(predictions, self.swap_index + 1, dim=1)
            # Initialize target tensor based on whether KL loss is used
            targets = (
                torch.zeros_like(predictions)
                if not self.kl_loss
                else predictions.clone()
            )

            for i in range(len(predictions)):
                # Indices of top classes for current prediction
                top_indices = top_class_indices[i]
                # Compute mean of the highest and swap_index-th predicted values
                mean_value = (
                    predictions[i, top_indices[0]]
                    + predictions[i, top_indices[self.swap_index]]
                ) / 2
                # Adjust target values for swap_index-th and highest class based on mean_value and gamma
                # targets[i, top_indices[self.swap_index]] = mean_value + self.gamma
                # targets[i, top_indices[0]] = mean_value - self.gamma
                targets[i, top_indices[self.swap_index]] = 1
                targets[i, top_indices[0]] = 0

        return targets, top_class_indices[:, 0]

    def __get_y_target_RAND__(self, inputs):
        with torch.no_grad():
            # Compute predictions for inputs
            predictions = self.f(inputs)
            # Get the indices of the maximum predicted class
            _, predicted_classes = torch.max(predictions, dim=1)
            # Initialize target tensor based on whether KL loss is used
            targets = (
                torch.zeros_like(predictions)
                if not self.kl_loss
                else predictions.clone()
            )

            for i in range(len(predictions)):
                # Get all class indices except for the predicted class
                alternative_classes = torch.arange(
                    predictions.shape[1], device=predictions.device
                )
                alternative_classes = alternative_classes[
                    alternative_classes != predicted_classes[i]
                ]
                # Randomly select a new class from alternatives
                new_class = alternative_classes[
                    torch.randint(0, len(alternative_classes), (1,))
                ]
                # Set target for the selected class to 1.0
                targets[i, new_class] = 1.0

        return targets, predicted_classes

    def __CW_loss_fun__(self, x, r, y_target, top1_index):
        y_pred_adv = self.f(x + r)
        loss = self.__LOSS__(y_pred_adv, y_target)

        mask = torch.zeros_like(loss, dtype=torch.bool)
        _, top1_index_adv = torch.max(y_pred_adv, dim=1)

        for i in range(len(y_target)):
            if not top1_index_adv[i] == top1_index[i]:
                mask[i] = True
        loss[mask] = 0

        # Combine the attack loss with the L2 regularization
        l2_reg = torch.norm(r, p=2)

        return l2_reg * self.c + loss.mean()

    def __NoCW_loss_fun__(self, x, r, y_target, top1_index):
        y_pred_adv = self.f(x + r)
        return self.__LOSS__(y_pred_adv, y_target).mean()

    def __LOSS__(self, y_pred_adv, y_target):
        return nn.functional.cross_entropy(y_pred_adv, y_target, reduction="none")

    def __perturb_s__(self, x):
        x = x.to(self.device)  # Move x to the device first
        y_pred = self.f(x)
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)
        y_target, top1_index = self._get_y_target(x)
        sum_losses = np.zeros(self.epochs)

        for epoch in range(self.epochs):
            self.iepoch = epoch
            loss = self.__loss_function__(x, r, y_target, top1_index)
            optimizer.zero_grad()
            loss.backward()

            # Here, we use the sign of the gradient for the update
            grad_sign = r.grad.sign()
            r.data = r.data - self.alpha * grad_sign
            # alpha is your step size for BIM
            r.data = torch.clamp(r.data, -self.eps, self.eps)

            sum_losses[epoch] += loss.item()
            if not (epoch + 1) % 100:
                logging.debug(f"Epoch: {epoch+1}/{self.epochs}")

        x_adv = x + r
        y_adv = self.f(x_adv).argmax(1)

        return x_adv, y_adv, y_pred, sum_losses

    def __perturb_g__(self, x):
        x = x.to(self.device)  # Move x to the device first
        y_pred = self.f(x)
        self.details[self.__batch_id__]["y_pred"] = y_pred.detach().cpu().numpy()
        r = self.__init_r__(x)
        optimizer = self.__get_optimizer__(r)
        y_target, top1_index = self._get_y_target(x)
        # 这里看起来不需要to_device
        sum_losses = np.zeros(self.epochs)

        for epoch in range(self.epochs):
            self.iepoch = epoch
            loss = self.__loss_function__(x, r, y_target, top1_index)
            optimizer.zero_grad()

            loss.backward(retain_graph=True)

            optimizer.step()
            r.data = torch.clamp(r.data, -self.eps, self.eps)
            sum_losses[epoch] += loss.item()
            if not (epoch + 1) % 100:
                logging.debug(f"Epoch: {epoch+1}/{self.epochs}")

            if self.make_demo:
                self.details[self.__batch_id__][epoch] = {
                    "loss": loss.item(),
                    "x_adv": (x + r).cpu().detach().numpy(),
                    "y_adv": self.f(x + r).cpu().detach().numpy(),
                }

        x_adv = x + r
        y_adv = self.f(x_adv).argmax(1)

        return x_adv, y_adv, y_pred, sum_losses

    def __init_r__(self, x):
        r_data = (
            torch.randint(2, x.shape, dtype=x.dtype, device=x.device) * 2 - 1
        ) * self.eps_init
        r = torch.nn.Parameter(r_data, requires_grad=True)
        return r

    def __get_optimizer__(self, r):
        return optim.Adam([r], lr=0.001, betas=(0.9, 0.999), eps=1e-07, amsgrad=True)


class Attacker(abstract_Attacker, abstract_attack_method):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method_chooser()

    def method_chooser(self):
        if self.swap:
            self._get_y_target = self.__get_y_target_SWAP__
        else:
            self._get_y_target = self.__get_y_target_RAND__

        if self.sign_only:
            self.__perturb__ = self.__perturb_s__
        else:
            self.__perturb__ = self.__perturb_g__

        if self.CW:
            self.__loss_function__ = self.__CW_loss_fun__
        else:
            self.__loss_function__ = self.__NoCW_loss_fun__


class SWAP(Attacker):
    def __init__(self, **kwargs):

        kwargs.setdefault("swap", True)
        kwargs.setdefault("swap_index", 1)
        kwargs.setdefault("CW", False)
        kwargs.setdefault("kl_loss", False)
        kwargs.setdefault("eps_init", 0.001)
        kwargs.setdefault("eps", 0.1)
        kwargs.setdefault("sign_only", False)
        kwargs.setdefault("make_demo", False)

        super().__init__(**kwargs)
        self.finished_params = copy.deepcopy(locals())
        self.finished_params.pop("self")


class PGD(Attacker):
    def __init__(self, **kwargs):

        kwargs.setdefault("swap", False)
        # kwargs.setdefault("swap_index", 1)
        kwargs.setdefault("CW", False)
        kwargs.setdefault("kl_loss", False)
        kwargs.setdefault("eps_init", 0.001)
        kwargs.setdefault("eps", 0.1)
        kwargs.setdefault("sign_only", False)
        kwargs.setdefault("make_demo", False)

        super().__init__(**kwargs)
        self.finished_params = copy.deepcopy(locals())
        self.finished_params.pop("self")
