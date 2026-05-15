import os
from datetime import timedelta
from typing import Annotated

import typer
from massive import RESTClient

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
print(f"current file path {current_file_path}")

# Get the directory containing the current file
current_dir = os.path.dirname(current_file_path)
print(f"current dir path {current_dir}")
OPHIR_DIR = os.path.join(current_dir, ".ophir")
DATA_DIR = os.path.join(OPHIR_DIR, "data")
MODEL_DIR = os.path.join(OPHIR_DIR, "model")
BASE_NAME = "ophir-ohlc-base"
FINETUNE_NAME = "ophire-ohlc-finetuned"
BASE_MODEL_CKPT = os.path.join(MODEL_DIR, f"{BASE_NAME}.ckpt")
TIME_MODIFIER = "-time-check"
EPOCH_MODIFIER = "best-{epoch:02d}-{val_loss:.5f}"

if not os.path.exists(OPHIR_DIR):
    os.makedirs(OPHIR_DIR)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def fetch_base_trainer(file_name=None):
    import lightning as L
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger

    if file_name is None:
        file_name = BASE_NAME

    # 1. Checkpoint every N minutes
    # Checkpoints will be saved with a format like 'time_check-{time}.ckpt'
    time_checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_DIR,
        filename=file_name + TIME_MODIFIER,
        train_time_interval=timedelta(minutes=1),  # Set N to your desired interval
        save_on_train_epoch_end=False,  # Prevents this callback from also saving at epoch end
    )

    # 2. Checkpoint at the end of every epoch
    # Checkpoints will be saved with a format like 'epoch_check-epoch={epoch}.ckpt'
    epoch_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=MODEL_DIR,
        filename=file_name + EPOCH_MODIFIER,
        save_top_k=1,  # Saves at the end of every epoch
        # To save all epoch checkpoints, or use 0 to save only the last
        save_on_train_epoch_end=True,
    )

    trainer = L.Trainer(
        max_steps=100000,
        precision="16-mixed",
        default_root_dir=MODEL_DIR,
        accelerator="cuda",
        callbacks=[
            time_checkpoint_callback,
            epoch_checkpoint_callback,
            LearningRateMonitor("step"),
        ],
        logger=TensorBoardLogger(MODEL_DIR, name="tensorboard-logger"),
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
    )
    return trainer


def fetch_finetune_trainer():
    import lightning as L
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger

    trainer = L.Trainer(
        precision="16-mixed",
        max_epochs=10000,
        default_root_dir=MODEL_DIR,
        accelerator="cuda",
        callbacks=[
            ModelCheckpoint(
                dirpath=MODEL_DIR,
                filename=FINETUNE_NAME,
                every_n_epochs=25,
                save_on_train_epoch_end=True,
            ),
            LearningRateMonitor("epoch"),
        ],
        logger=TensorBoardLogger(MODEL_DIR, name="tensorboard-logger"),
        check_val_every_n_epoch=1,
    )
    return trainer


def get_massive_client():
    assert os.path.exists(os.path.join(OPHIR_DIR, ".massive_key"))
    with open(os.path.join(OPHIR_DIR, ".massive_key")) as f:
        key = f.readline().strip()
    return RESTClient(key)


def _latest_base_ckpt(filename=None):
    base_paths = sorted([path for path in os.listdir(MODEL_DIR) if filename in path])
    if len(base_paths) > 1:
        base_versions = sorted(
            [
                (
                    int(version.removeprefix(f"{filename}-v").removesuffix(".ckpt")),
                    version,
                )
                for version in base_paths
                if f"{filename}-v" in version
            ]
        )
        latest_version = base_versions[-1][1]
    else:
        latest_version = base_paths[0]

    return latest_version


def _latest_finetuned_ckpt():
    fintune_paths = sorted([path for path in os.listdir(MODEL_DIR) if FINETUNE_NAME in path])
    if len(fintune_paths) > 1:
        base_versions = sorted(
            [
                (
                    int(version.removeprefix(f"{FINETUNE_NAME}-v").removesuffix(".ckpt")),
                    version,
                )
                for version in fintune_paths
                if f"{FINETUNE_NAME}-v" in version
            ]
        )
        latest_version = base_versions[-1][1]
    else:
        latest_version = fintune_paths[0]

    return latest_version


def get_default_data_days_dir():
    return os.path.join(DATA_DIR, "days")


def clear_ignore_symbols() -> None:
    print("Reseting ignore symbols mode...")
    path = os.path.join(DATA_DIR, "ignore-symbols.txt")
    if os.path.exists(path):
        os.remove(path)


def set_ignore_symbols(symbols) -> None:
    symbols = set(fetch_ignore_symbols_list()).union(symbols)
    symbols = sorted(symbols)
    with open(os.path.join(DATA_DIR, "ignore-symbols.txt"), "w") as f:
        for symbol in symbols:
            f.write(f"{symbol}\n")
    print(f"Entering Ignore Symbol mode...currently ignoring {len(symbols)} symbols")


def fetch_ignore_symbols_list():
    if not os.path.exists(os.path.join(DATA_DIR, "ignore-symbols.txt")):
        return []

    with open(os.path.join(DATA_DIR, "ignore-symbols.txt")) as f:
        symbols = [symbol.strip() for symbol in f.readlines()]
    return symbols


def load_base_model_ckpt(strict=True, return_ckpt_path=False, file_name=None, time_version=True):
    from ophir.training_models import LightningOHLCPredictor

    if file_name is None:
        file_name = BASE_NAME

    if time_version:
        file_name += TIME_MODIFIER
    else:
        file_name += EPOCH_MODIFIER
    file_name = file_name.split("{")[0]

    latest_version = _latest_base_ckpt(filename=file_name)
    print(f"loading {latest_version}")

    last_ckpt = os.path.join(MODEL_DIR, latest_version)
    if not return_ckpt_path:
        return LightningOHLCPredictor.load_from_checkpoint(last_ckpt, strict=strict)
    return LightningOHLCPredictor.load_from_checkpoint(last_ckpt, strict=strict), last_ckpt


def load_fintuned_ckpt(strict=True, return_ckpt_path=False):
    from ophir.training_models import LightningOHLCPredictor

    latest_version = _latest_finetuned_ckpt()
    print(f"loading {latest_version}")

    last_ckpt = os.path.join(MODEL_DIR, latest_version)
    if not return_ckpt_path:
        return LightningOHLCPredictor.load_from_checkpoint(last_ckpt, strict=strict)
    return LightningOHLCPredictor.load_from_checkpoint(last_ckpt, strict=strict), last_ckpt


def predict_trainer():
    import lightning as L

    trainer = L.Trainer(precision="16-mixed", accelerator="cuda")
    return trainer


app = typer.Typer()


@app.command()
def massive_key(
    key: Annotated[str, typer.Argument(help="MASSIVE API key")],
) -> None:
    with open(os.path.join(OPHIR_DIR, ".massive_key"), "w") as f:
        f.write(f"{key}\n")
