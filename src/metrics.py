import numpy as np


class MetricSRCNN:
    def __init__(self):
        self.best_epoch = 0
        self.best_psnr = float("-inf")
        self.total_train_loss = []
        self.total_train_psnr = []
        self.total_eval_loss_set5 = []
        self.total_eval_psnr_set5 = []
        self.total_eval_loss_set14 = []
        self.total_eval_psnr_set14 = []

        self.eval_psnr = []

    def load_checkpoint(self, checkpoint: dict):
        self.best_epoch = checkpoint["best_epoch"]
        self.best_psnr = checkpoint["best_psnr"]
        self.total_train_loss = checkpoint["total_train_loss"]
        self.total_train_psnr = checkpoint["total_train_psnr"]
        self.total_eval_loss_set5 = checkpoint["total_eval_loss_set5"]
        self.total_eval_psnr_set5 = checkpoint["total_eval_psnr_set5"]
        self.total_eval_loss_set14 = checkpoint["total_eval_loss_set14"]
        self.total_eval_psnr_set14 = checkpoint["total_eval_psnr_set14"]

    def save_checkpoint(self):
        return {
            "best_epoch": self.best_epoch,
            "best_psnr": self.best_psnr,
            "total_train_loss": self.total_train_loss,
            "total_train_psnr": self.total_train_psnr,
            "total_eval_loss_set5": self.total_eval_loss_set5,
            "total_eval_psnr_set5": self.total_eval_psnr_set5,
            "total_eval_loss_set14": self.total_eval_loss_set14,
            "total_eval_psnr_set14": self.total_eval_psnr_set14,
        }

    def add_eval(self, arr):
        self.eval_psnr.extend(arr)

    def get_eval_score(self):
        curr_psnr = np.mean(self.eval_psnr)
        self.eval_psnr = []

        return curr_psnr


class MetricSRGAN:
    def __init__(self):
        self.best_epoch = 0
        self.best_psnr = float("-inf")
        self.total_train_gen_loss = []
        self.total_train_disc_loss = []
        self.total_train_psnr = []
        self.total_eval_loss_set5 = []
        self.total_eval_psnr_set5 = []
        self.total_eval_loss_set14 = []
        self.total_eval_psnr_set14 = []

        self.eval_psnr = []

    def load_checkpoint(self, checkpoint: dict):
        self.best_epoch = checkpoint["best_epoch"]
        self.best_psnr = checkpoint["best_psnr"]
        self.total_train_gen_loss = checkpoint["total_train_gen_loss"]
        self.total_train_disc_loss = checkpoint["total_train_disc_loss"]
        self.total_train_psnr = checkpoint["total_train_psnr"]
        self.total_eval_loss_set5 = checkpoint["total_eval_loss_set5"]
        self.total_eval_psnr_set5 = checkpoint["total_eval_psnr_set5"]
        self.total_eval_loss_set14 = checkpoint["total_eval_loss_set14"]
        self.total_eval_psnr_set14 = checkpoint["total_eval_psnr_set14"]

    def save_checkpoint(self):
        return {
            "best_epoch": self.best_epoch,
            "best_psnr": self.best_psnr,
            "total_train_gen_loss": self.total_train_gen_loss,
            "total_train_disc_loss": self.total_train_disc_loss,
            "total_train_psnr": self.total_train_psnr,
            "total_eval_loss_set5": self.total_eval_loss_set5,
            "total_eval_psnr_set5": self.total_eval_psnr_set5,
            "total_eval_loss_set14": self.total_eval_loss_set14,
            "total_eval_psnr_set14": self.total_eval_psnr_set14,
        }

    def add_eval(self, arr):
        self.eval_psnr.extend(arr)

    def get_eval_score(self):
        curr_psnr = np.mean(self.eval_psnr)
        self.eval_psnr = []

        return curr_psnr
