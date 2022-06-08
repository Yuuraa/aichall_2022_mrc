import os
from time import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# from apex import amp


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss,
        metrics,
        device,
        logger,
        amp,
        tokenizer,
        interval=100,
    ):

        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.device = device
        self.logger = logger
        self.amp = amp
        self.interval = interval
        self.tokenizer = tokenizer

        # History
        self.loss_sum = 0  # Epoch loss sum
        self.loss_mean = 0  # Epoch loss mean
        self.y = list()
        self.y_preds = list()
        self.score_dict = dict()  # metric score
        self.elapsed_time = 0

    def train(self, train_dataloader, val_dataloader):
        # Set Recorder row
        row_dict = dict()
        row_dict["epoch_index"] = self.epoch_index
        row_dict["train_serial"] = self.train_serial

        for mode in ["train", "valid"]:
            print(f"{mode.upper()} {self.epoch_index}/{self.n_epochs}")
            self.logger.info(f"--{mode.upper()} {self.epoch_index}/{self.n_epochs}")

            start_timestamp = time()
            self.model.train() if mode == "train" else self.model.eval()
            dataloader = train_dataloader if mode == "train" else val_dataloader

            for batch_index, batch in enumerate(tqdm(dataloader, leave=True)):

                # initialize calculated gradients (from prev step)
                self.optimizer.zero_grad()
                # pull all the tensor batches required for training
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                start_positions = batch["start_positions"].to(self.device)
                end_positions = batch["end_positions"].to(self.device)

                # train model on batch and return outputs (incl. loss)
                # Inference
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )

                loss = outputs.loss
                start_score = outputs.start_logits
                end_score = outputs.end_logits

                start_idx = torch.argmax(start_score, dim=1).cpu().tolist()
                end_idx = torch.argmax(end_score, dim=1).cpu().tolist()

                # Update
                if mode == "train":

                    if self.amp is None:
                        loss.backward()

                    else:
                        with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()

                    self.optimizer.step()

                elif mode in ["val", "test"]:
                    pass

                # History
                self.loss_sum += loss.item()

                # create answer; list of strings
                for i in range(len(input_ids)):
                    if start_idx[i] > end_idx[i]:
                        output = ""

                    self.y_preds.append(
                        self.tokenizer.decode(input_ids[i][start_idx[i] : end_idx[i]])
                    )
                    self.y.append(
                        self.tokenizer.decode(
                            input_ids[i][start_positions[i] : end_positions[i]]
                        )
                    )

                # Logging
                if batch_index % self.interval == 0:
                    msg = f"batch: {batch_index}/{len(dataloader)} loss: {loss.item()}"
                    self.logger.info(msg)

                # Epoch history
                self.loss_mean = self.loss_sum / len(dataloader)  # Epoch loss mean

                # Metric
                for metric_name, metric_func in self.metrics.items():
                    score = metric_func(self.y, self.y_preds)
                    self.score_dict[metric_name] = score

                # Elapsed time
                end_timestamp = time()
                self.elapsed_time = end_timestamp - start_timestamp

            row_dict[f"{mode}_loss"] = self.loss_mean
            row_dict[f"{mode}_elapsed_time"] = self.elapsed_time

            for metric_str, score in self.score_dict.items():
                row_dict[f"{mode}_{metric_str}"] = score
            self.clear_history()
        return row_dict

    def clear_history(self):
        self.loss_sum = 0
        self.loss_mean = 0
        self.y_preds = list()
        self.y = list()
        self.score_dict = dict()
        self.elapsed_time = 0
