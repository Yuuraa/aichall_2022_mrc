import argparse
import copy
import os
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from models.utils import get_model
from modules.datasets import QADataset, CustomQADataset, json_to_df
from modules.earlystoppers import EarlyStopper
from modules.losses import get_loss
from modules.metrics import get_metric
from modules.optimizers import get_optimizer
from modules.preprocessing import get_tokenizer
from modules.recorders import Recorder
from modules.trainer import Trainer
from modules.utils import get_logger, load_yaml, save_yaml, save_pickle, load_pickle

parser = argparse.ArgumentParser()
parser.add_argument("--train_cfg", type=str, default="train_config.yml")
args = parser.parse_args()

# Root directory
PROJECT_DIR = os.path.dirname(__file__)

# Load config
config_path = os.path.join(PROJECT_DIR, "config", args.train_cfg)
config = load_yaml(config_path)

# Train Serial
kst = timezone(timedelta(hours=9))
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

# Recorder directory
DEBUG = config["TRAINER"]["debug"]
print(f"debug {DEBUG}")

RECORDER_DIR = os.path.join(PROJECT_DIR, "results", "train", train_serial)
os.makedirs(RECORDER_DIR, exist_ok=True)

# Data directory
DATA_DIR = config["DIRECTORY"]["dataset"]
PREPROCESSED_DIR = os.path.join(
    DATA_DIR, "preprocessed" #, config["TRAINER"]["pretrained"]
)

# Seed
torch.manual_seed(config["TRAINER"]["seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(config["TRAINER"]["seed"])
random.seed(config["TRAINER"]["seed"])

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["TRAINER"]["gpu"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

if __name__ == "__main__":
    """
    00. Set Logger
    """
    logger = get_logger(name="train", dir_=RECORDER_DIR, stream=False)
    logger.info(f"Set Logger {RECORDER_DIR}")
    print("/".join(logger.handlers[0].baseFilename.split("/")[:-1]))

    """
    01. Load data
    """
    # Load tokenizer
    tokenizer = get_tokenizer(
        config["TRAINER"]["tokenizer"], config["TRAINER"]["pretrained"]
    )

    # Load Dataset

    # preprocessor
    train_file_name = "train_dataset.pkl"
    val_file_name = "val_dataset.pkl"
    if not DEBUG and os.path.isfile(os.path.join(PREPROCESSED_DIR, train_file_name)):
        train_dataset = load_pickle(os.path.join(PREPROCESSED_DIR, train_file_name))
        val_dataset = load_pickle(os.path.join(PREPROCESSED_DIR, val_file_name))
        logger.info("loaded existing .pt")
    else:
        file_name = 'sample.json' if DEBUG else "train.json"
        df_dataset = json_to_df(
            data_dir=os.path.join(DATA_DIR, file_name), 
            mode='train',
            pororo_dir=os.path.join(DATA_DIR, 'train_para_final.pkl'),
            gpt_dir=None,
        )
        train_dataset = df_dataset[:-1 * int(len(df_dataset) * 0.1)]
        val_dataset = df_dataset[-1 * int(len(df_dataset) * 0.1):]

        if not DEBUG:
            os.makedirs(PREPROCESSED_DIR, exist_ok=True)
            save_pickle(
                path=os.path.join(PREPROCESSED_DIR, train_file_name),
                obj=train_dataset
            )
            save_pickle(
                path=os.path.join(PREPROCESSED_DIR, val_file_name),
                obj=val_dataset
            )
        logger.info("loaded data, created .pkl")

    train_dataset = CustomQADataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_len=tokenizer.model_max_length,
        mode='train',
        question_shuffle_aug=config["AUGMENTATION"]["question_shuffle_aug"],
        pororo_aug=config["AUGMENTATION"]["pororo_aug"],
        gpt_aug=config["AUGMENTATION"]["gpt_aug"],
    )
    val_dataset = CustomQADataset(
        dataset=val_dataset,
        tokenizer=tokenizer,
        max_seq_len=tokenizer.model_max_length,
        mode='val',
        question_shuffle_aug=False,
        pororo_aug=False,
        gpt_aug=False,
    )

    if DEBUG:
        print(f'train: {len(train_dataset)}, val: {len(val_dataset)}')
        answers, decoded_answers = [], []
        for i in range(len(train_dataset)):
            tmp = train_dataset[i]
            input_ids = tmp["input_ids"]
            start_idx = tmp["start_positions"]
            end_idx = tmp["end_positions"]
            answers.append(tmp['answer_text'])
            decoded_answers.append(tokenizer.decode(input_ids[start_idx:end_idx]))
        print(f'answers: {answers}')
        print(f'decoded: {decoded_answers}')

    # DataLoader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config["DATALOADER"]["batch_size"],
        num_workers=config["DATALOADER"]["num_workers"],
        shuffle=config["DATALOADER"]["shuffle"],
        pin_memory=config["DATALOADER"]["pin_memory"],
        drop_last=config["DATALOADER"]["drop_last"],
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config["DATALOADER"]["batch_size"],
        num_workers=config["DATALOADER"]["num_workers"],
        shuffle=False,
        pin_memory=config["DATALOADER"]["pin_memory"],
        drop_last=config["DATALOADER"]["drop_last"],
    )

    logger.info(f"Load data, train:{len(train_dataset)} val:{len(val_dataset)}")

    """
    02. Set model
    """
    # Load model
    model_name = config["TRAINER"]["model"]
    model = get_model(
        model_name=model_name, pretrained=config["TRAINER"]["pretrained"]
    ).to(device)

    """
    03. Set trainer
    """
    # Optimizer
    optimizer = get_optimizer(optimizer_name=config["TRAINER"]["optimizer"])
    optimizer = optimizer(
        params=model.parameters(), lr=config["TRAINER"]["learning_rate"]
    )

    # Loss
    loss_fn = get_loss(loss_name=config["TRAINER"]["loss"])

    # Metric
    metrics = {
        metric_name: get_metric(metric_name)
        for metric_name in config["TRAINER"]["metric"]
    }

    # Early stoppper
    early_stopper = EarlyStopper(
        patience=config["TRAINER"]["early_stopping_patience"],
        mode=config["TRAINER"]["early_stopping_mode"],
        logger=logger,
    )
    # AMP
    if config["TRAINER"]["amp"] == True:
        from apex import amp

        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics=metrics,
        device=device,
        logger=logger,
        tokenizer=tokenizer,
        amp=amp if config["TRAINER"]["amp"] else None,
        interval=config["LOGGER"]["logging_interval"],
        grad_accum=config["TRAINER"]["grad_accum"] if "grad_accum" in config["TRAINER"].keys() else 1
    )

    """
    Logger
    """
    # Recorder
    recorder = Recorder(
        record_dir=RECORDER_DIR,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        amp=amp if config["TRAINER"]["amp"] else None,
        logger=logger,
    )

    # !Wandb
    USE_WANDB = config["LOGGER"]["wandb"]["use"]
    if USE_WANDB:
        wandb_project_serial = config["LOGGER"]["wandb"]["project_serial"]
        wandb_username = config["LOGGER"]["wandb"]["username"]
        wandb.init(
            project=wandb_project_serial, dir=RECORDER_DIR, entity=wandb_username
        )
        wandb.run.name = train_serial
        wandb.config.update(config)
        wandb.watch(model)

    # Save train config
    save_yaml(os.path.join(RECORDER_DIR, args.train_cfg), config)

    """
    04. TRAIN
    """
    # Train
    n_epochs = config["TRAINER"]["n_epochs"]
    for epoch_index in range(n_epochs):

        # Set Recorder row
        row_dict = dict()
        row_dict["epoch_index"] = epoch_index
        row_dict["train_serial"] = train_serial

        """
        Train
        """
        print(f"Train {epoch_index}/{n_epochs}")
        logger.info(f"--Train {epoch_index}/{n_epochs}")

        trainer.train(
            dataloader=train_dataloader,
            epoch_index=epoch_index,
            tokenizer=tokenizer,
            mode="train",
            random_masking=config['AUGMENTATION']['random_masking']
        )

        row_dict["train_loss"] = trainer.loss_mean
        row_dict["train_elapsed_time"] = trainer.elapsed_time

        for metric_str, score in trainer.score_dict.items():
            row_dict[f"train_{metric_str}"] = score
        trainer.clear_history()

        """
        Validation
        """
        print(f"Val {epoch_index}/{n_epochs}")
        logger.info(f"--Val {epoch_index}/{n_epochs}")

        # trainer.validate(dataloader=val_dataloader, epoch_index=epoch_index)
        trainer.train(
            dataloader=val_dataloader,
            epoch_index=epoch_index,
            tokenizer=tokenizer,
            mode="val",
            random_masking=False
        )

        row_dict["val_loss"] = trainer.loss_mean
        row_dict["val_elapsed_time"] = trainer.elapsed_time

        for metric_str, score in trainer.score_dict.items():
            row_dict[f"val_{metric_str}"] = score
        trainer.clear_history()

        """
        Record
        """
        recorder.add_row(row_dict)
        recorder.save_plot(config["LOGGER"]["plot"])

        #!WANDB
        if USE_WANDB:
            wandb.log(row_dict)

        """
        Early stopper
        """
        early_stopping_target = config["TRAINER"]["early_stopping_target"]
        early_stopper.check_early_stopping(loss=row_dict[early_stopping_target])

        if early_stopper.patience_counter == 0:
            recorder.save_weight(epoch=epoch_index)
            best_row_dict = copy.deepcopy(row_dict)

        if early_stopper.stop == True:
            logger.info(
                f"Early stopped, counter {early_stopper.patience_counter}/{config['TRAINER']['early_stopping_patience']}"
            )

            if USE_WANDB:
                wandb.log(best_row_dict)
            break
