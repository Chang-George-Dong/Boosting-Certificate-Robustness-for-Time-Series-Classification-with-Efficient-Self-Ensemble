from CODE.Utils.package import *
from CODE.Utils.utils import *
from CODE.Utils.constant import *


class Trainer:

    def __init__(self, init_model_only=False, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.config = kwargs

        (
            self.train_loader,
            self.test_loader,
            self.model_params["shape"],
            _,
            self.model_params["num_classes"],
            self.class_weights,
        ) = data_loader(
            self.dataset,
            batch_size=self.batch_size,
            data_path=DATASET_PATH,
        )
        self.model_params["out_dir"] = self.__set_output_dir__()
        self.state = kwargs.get("state", "train")
        all_attrs = vars(self)
        for key, value in all_attrs.items():
            self.model_params[key] = value
        self.model = self.model(**self.model_params)
        self.model.to(self.device)
        if init_model_only:
            return

        self.loss_fn = kwargs.get(
            "loss_fn", nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        )
        self.optimizer = kwargs.get("optimizer", Adam(self.model.parameters()))
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=50,
            min_lr=0.0001,
        )
        self.model_info = {
            "architecture": str(self.model),
            "i_epoch": None,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": kwargs,
            "out_dir": self.out_dir,
        }

    ####################### __init_finisehd__ #######################

    def __set_output_dir__(self):
        self.method_path = self.model.__name__ + "_"
        self.method_path += self.path_parameter
        self.out_dir = os.path.join(
            TRAIN_OUTPUT_PATH,
            self.method_path,
            self.dataset,
        )
        return self.out_dir

    def __check_resume__(self, to_device):
        def check_check_point(path):
            checkpoint = torch.load(path, map_location=self.device)
            for key, value in checkpoint["config"].items():
                if key in self.config.keys():
                    if self.config[key] != value:
                        logging.warning(f"{key} is not match, be careful!")
                        print(f"now config, {key}: {self.config[key]}"[:200])
                        print(f"old config, {key}: {value}"[:200])

            if checkpoint["architecture"] != str(self.model):
                logging.warning(
                    """Model structure is not match, unable to resume!
                    Please check the model name or set override=True or change the out_dir."""
                )
                return -1, checkpoint

            return checkpoint["i_epoch"], checkpoint

        def __resume__(checkpoint: dict):
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logging.info(f"Pth file load from {checkpoint['out_dir']}")

        start = 1
        target_file = os.path.join(self.out_dir, MODEL_NAME)

        if os.path.exists(target_file):
            res = os.path.join(self.out_dir, "test_metrics.csv")
            if self.override:
                logging.info(f"Del task {self.dataset} all files.")
                shutil.rmtree(self.out_dir)
            else:
                if os.path.exists(res):
                    return -1
                start, checkpoint = check_check_point(target_file)
                if not os.path.exists(res) and start == -1:
                    logging.warning(f"File {res} not found, del {target_file}")
                    logging.warning("Broken file Project! Must be override!")
                    self.override = True
                    return self.__check_resume__(to_device)

                __resume__(checkpoint)
        create_directory(self.out_dir)
        return start

    def train_and_evaluate(self, override=False, to_device=True):
        self.override = override
        start_epoch = self.__check_resume__(to_device)
        if start_epoch == -1:
            return
        test_loss_file = open(os.path.join(self.out_dir, "test_loss.txt"), "a")
        logging.info(f"Start locking File {test_loss_file.name}")

        learning_rate_file = open(os.path.join(self.out_dir, "learningRate.txt"), "a")
        logging.info(f"Start locking File {learning_rate_file.name}")

        # current_time = datetime.now()
        # logging.info(f"Current time: {current_time} \n")
        self.start_time = time.time()
        last_saved_time = self.start_time
        for epoch in range(start_epoch, self.epochs + 1):
            self.__train_one_epoch__()
            last_saved_time = self.__save_check_point__(
                epoch,
                last_saved_time,
                test_loss_file,
                learning_rate_file,
                to_device,
            )

            # Evaluation Phase
            test_loss = self.__cal_loss__()

            # Record test loss and learning rate
            test_loss_file.write(f"{test_loss}\n")
            learning_rate_file.write(f"{self.optimizer.param_groups[0]['lr']}\n")
            self.scheduler.step(test_loss)

        test_loss_file.close()
        learning_rate_file.close()
        self.model.save_weights()

    def __save_check_point__(
        self,
        epoch,
        last_saved_time,
        test_loss_file,
        learning_rate_file,
        to_device,
    ):
        # Save model weights every 50 epochs and delete the old one
        if (time.time() - last_saved_time > 600) or epoch >= self.epochs:
            checkpoint_path = os.path.join(self.out_dir, MODEL_NAME)
            self.model_info["i_epoch"] = epoch
            torch.save(
                self.model_info,
                checkpoint_path,
            )
            learning_rate_file.flush()
            test_loss_file.flush()
            last_saved_time = time.time()
        if epoch == self.epochs and to_device:
            self.train_result["duration"] = time.time() - self.start_time
            self.evaluate()
            save_metrics(self.out_dir, "train", self.train_result)
            save_metrics(self.out_dir, "test", self.test_result)
            save_conf_to_json(self.out_dir, vars(self))
            logging.info(f"Task {self.dataset} Finished")
            logging.info("-" * 80)

        return last_saved_time

    def __cal_loss__(
        self,
    ):
        test_loss = 0
        for x_batch, y_batch in self.test_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            loss, _ = self.loss_function(x_batch, y_batch)
            test_loss += loss.item()
        test_loss /= len(self.test_loader)
        return test_loss

    def evaluate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        test_preds, test_targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                # predictions = self.__f__(model, x_batch)
                loss, predictions = self.loss_function(x_batch, y_batch)
                test_loss += loss.item()
                # test_loss += loss_function(x_batch, y_batch).item()
                pred = predictions.argmax(dim=1, keepdim=True)
                correct += pred.eq(y_batch.view_as(pred)).sum().item()
                test_preds.extend(pred.squeeze().cpu().numpy())
                test_targets.extend(y_batch.cpu().numpy())

            test_loss /= len(self.test_loader)

        accuracy = correct / len(self.test_loader.dataset)
        precision, recall, f1 = metrics(test_targets, test_preds)

        self.test_result = {
            "loss": test_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def loss_function(self, x_batch, y_batch):
        try:
            predictions = self.model.run(x_batch)
        except AttributeError:
            self.model.run = self.model.forward
            predictions = self.model.run(x_batch)
        # CE loss
        loss_CE = self.loss_fn(predictions, y_batch)
        return loss_CE, predictions

    def __train_one_epoch__(self):
        self.model.train()

        train_loss = 0
        train_preds, train_targets = [], []
        for x_batch, y_batch in self.train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            loss, predictions = self.loss_function(x_batch, y_batch)

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_preds.extend(predictions.argmax(dim=1).cpu().numpy())
            train_targets.extend(y_batch.cpu().numpy())

        train_loss /= len(self.train_loader)

        accuracy = np.mean(np.array(train_preds) == np.array(train_targets))
        precision, recall, f1 = metrics(train_targets, train_preds)

        self.train_result = {
            "loss": train_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        self.scheduler.step(train_loss)
