import os
import logging
import re
import sys
import h5py
import numpy as np
import wandb
import timeit
from time import time
from functools import wraps
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import torch.nn as nn
import yaml
import argparse
import uuid
from collections import OrderedDict
from torch import Tensor
from pathlib import Path, PosixPath
from torch.nn import Module
from tqdm.auto import tqdm
from dataset.SmallEarthNet import SmallEarthNet
from typing import Callable, Dict, List, Optional, Union, Tuple
from serverpFedGen import FedGen
from vit_pytorch import ViT
from torchmetrics import (
    F1Score,
    Precision,
    Recall,
    Accuracy,
    Dice,
    MatthewsCorrCoef,
    LabelRankingAveragePrecision,
    LabelRankingLoss,
)
from torch.utils.data import (
    DataLoader,
    Dataset,
    random_split,
    ConcatDataset,
)


# %% Logging


def init_info(_log):
    _log.setLevel(logging.INFO)
    _log.addHandler(logging.StreamHandler(sys.stderr))
    _formatter = logging.Formatter(f"main: %(message)s")
    for _handler in _log.handlers:
        _handler.setFormatter(_formatter)
    return _log.info


_log = None
if "info" not in locals():  # if info is not defined
    _log = logging.getLogger(__name__)
    info = init_info(_log)

info("Logger initialized")

# %% General information

user_name = os.getlogin()

if user_name in ["root", "ubuntu"]:
    user_name = "amer"

info(f"User: {user_name}")
info(f"torch version: {torch.__version__}")
info(f"torchvison version: {torchvision.__version__}")

# %% get country name from arguments

available_countries = [
    "Finland",
    "Portugal",
    "Ireland",
    "Lithuania",
    "Serbia",
    "Austria",
    "Switzerland",
    "TinyFinland",
    "UnitedNations",
]

parser = argparse.ArgumentParser(description="SmallEarthNet")
parser.add_argument(
    "--test_sets",
    nargs="+",
    type=str,
    choices=available_countries,
    help=f"Available countries {available_countries}. it can be an array of countries",
    default="Switzerland",
)


parser.add_argument(
    "--experiment_id",
    type=str,
    help=f"An id to identify multiple runs of the same experiment. Used as a tag in wandb.",
    default=str(uuid.uuid4())[:8],
)

parser.add_argument(
    "--fedgen_config",
    type=str,
    help=f"fedgen config file to load",
)
parser.add_argument(
    "--backbone",
    type=str,
    choices=["resnet50", "vit"],
    help="Backbone model to use",
    default="train",
)

parser.add_argument("-tse", "--test-server-every", type=int, default=5, help="test sever every n epoch")

parser.add_argument(
    "--data_root",
    type=str,
    help="Where all the counties folders or/and the hdf5 files, *-train.csv, *_mean-std.yml files are stored.",
    default=f"{os.getenv('HOME')}/data/BigEarthNet/",
)
parser.add_argument(
    "--output_dir",
    type=str,
    help="Where to save the output, e.g. the models",
    default=f"{os.getenv('HOME')}/data/BigEarthNet-output/",
)
parser.add_argument(
    "--batch_size",
    type=int,
    help="Batch size",
    default=128,
)
parser.add_argument(
    "--workers",
    type=int,
    help="Number of workers to load the data",
    default=0,
)
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "cuda", "mps"],
    help="Backbone model to use",
)
parser.add_argument(
    "--epochs",
    type=int,
    help="Number of epochs to train inside each federated training cycle"
    "(i.e. for each round of federated training the model is trained for this many epochs)",
)
parser.add_argument(
    "--rounds",
    type=int,
    help="(Communication rounds) Number of rounds the clients are going to communicate with the server",
)
parser.add_argument(
    "--scale",
    type=str,
    choices=["true", "false", "True", "False"],
    default="false",
    help="When loading a data point, scale its values to be between 0 and 1",
)
parser.add_argument(
    "--fed_clients",
    nargs="+",
    type=str,
    choices=available_countries,
    help=str(
        f"Countries to use as clients. e.g --fed_clients Finland Portugal. "
        f"Minimum of 2 clients are required."
        f"Available countries {available_countries}",
    ),
)
parser.add_argument(
    "--fed_servers_output_dir",
    type=str,
    help="Where to save the output of the server",
)
parser.add_argument(
    "--fed_clients_source_dir",
    type=str,
    help="Where to find the stored models after each epoch for the clients",
)
parser.add_argument(
    "--timeit",
    type=str,
    choices=["true", "false", "True", "False"],
    default="false",
    help="Time certain functions",
)
parser.add_argument(
    "--cache",
    type=str,
    choices=["true", "false", "True", "False"],
    default="False",
    help="Cache the dataset in RAM when before training",
)
parser.add_argument(
    "--fed",
    type=str,
    choices=["avg", "bn", "gen"],
    help="Federated learning method",
    required=True,
)


parser.add_argument("--mode")  # Useless argument. Keep it to avoid crashes when running in Notebook
parser.add_argument("--port")  # Useless argument. Keep it to avoid crashes when running in Notebook
args = parser.parse_args()

args.timeit = args.timeit.lower() == "true"
args.cache = args.cache.lower() == "true"
args.scale = args.scale.lower() == "true"

assert args.fed in ["avg", "bn", "gen"], "Federated learning method must be avg, bn or gen"
assert args.epochs > 0, "Number of epochs must be greater than 0"
assert args.rounds > 0, "Number of rounds must be greater than 0"

info(f"Clients: {args.fed_clients}")
info(f"Test sets: {args.test_sets}")
info(f"Federated learning method: {args.fed}")
info(f"Experiment ID: {args.experiment_id}")
info(f"Backbone: {args.backbone}")

# %% Check if a GPU is available

# Check if GPU is available

device = None
if torch.cuda.is_available():
    info(f"You have {torch.cuda.device_count()} GPU")
    device = torch.device("cuda")
else:
    raise ValueError("No GPU available")

assert device.type == "cuda", "GPU is not available"


def data_root():
    return Path(args.data_root)


def output_dir():
    return Path(args.output_dir)


def country_setup(_country_name: str in available_countries):
    _country_path = data_root() / _country_name
    return data_root(), _country_path, _country_name


# Available countries:
#   "Finland",
#   "Portugal",
#   "Ireland",
#   "Lithuania",
#   "Serbia",
#   "Austria",
#   "Switzerland",
#   "TinyFinland",


# %% Set config, lr, momentum, epochs, etc.
config = {  # TODO: based on nothing values. I pulled those values from my head.
    "lr": 0.001,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "epochs": args.epochs,
    "checkpoint": None,
    "test_sets": args.test_sets,
    "backbone": args.backbone,
    "validation_split": 0.2,  # 20% of data for validation
    "seed": 42,
    "num_classes": 19,  # 19 or 43 number of classes
    "batch_size": args.batch_size,
    "workers": args.workers,
    "fed": args.fed,
    "rounds": args.rounds,
    "scale": args.scale,
}

if args.fed == "gen":
    with open("config/fedgen_config.yaml") as file:
        fedgen_dict = yaml.safe_load(file)
    with open("config/label_frequencies.yaml") as lf:
        label_freq = yaml.safe_load(lf)
    config["dataset"] = "bigearthnet2"
    config["device"] = device
    config["train"] = True
    config.update(fedgen_dict)
    config["label_frequencies"] = label_freq
    fed_config = {
        "epochs": args.epochs,
        "clients": args.fed_clients,
        "server_output_dir": args.fed_servers_output_dir,
    }
    config.update(fed_config)


def get_clients_dirs():
    """
    Returns a dict of client names and their directories that contain the minimal
     number of required epochs.
    """

    # search in data_root for the clients directories
    _clients_dirs = {}

    # Get only the directories from the fed_clients_source_dir
    _p = Path(args.fed_clients_source_dir)
    _dirs = [x for x in _p.iterdir() if x.is_dir()]
    _clients = [x for x in args.fed_clients]

    # find the directories that match the clients names using regex
    for _dir in _dirs:
        _dir_name = _dir.name
        for client in _clients:
            _backbone = config["backbone"]
            reg = rf"{client}.*_{_backbone}_([0-9]+)-epochs"
            found = re.search(reg, _dir_name)

            if found:
                info(f"{found.group(0)} -> Pretrained models: {found.group(1)}")
                epochs_count = int(found.group(1))
                if epochs_count >= args.fed_epochs:
                    # Find stored epochs files in the directory
                    _pth_files = [x for x in _dir.iterdir() if x.suffix == ".pth"]
                    if len(_pth_files) >= args.fed_epochs:
                        _clients_dirs[client] = _dir
                    else:
                        info(f"Not enough epochs in {_dir_name} for {client}")

    if _clients_dirs == {}:
        raise ValueError(
            f'No matching clients directories found. Regex pattern: r"{reg}" in:\n'
            f"{_dirs}\n"
            f"Required Epochs count: {args.fed_epochs}\n"
            f"The specified directory: {args.fed_clients_source_dir}"
        )

    return _clients_dirs


fed_config = {
    "epochs": args.epochs,
    "clients": args.fed_clients,
    "server_output_dir": args.fed_servers_output_dir,
}
config.update(fed_config)

# %% Utils

class_names = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of Natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]


def label_to_class_name(label: int) -> str:
    """
    Converts a label to a class name.
    """
    return class_names[label]


def add_class_names(labels: Union[list, Tensor]) -> dict:
    """
    Adds class names to a list of labels.
    """

    # if labels is a tensor, convert it to a list
    if isinstance(labels, Tensor):
        labels = labels.tolist()

    assert len(labels) == len(class_names), "Labels and class names must have the same length"
    return dict(zip(class_names, labels))


def run_name():
    return f"{wandb.run.id}_{config['backbone']}_{config['epochs']}-epochs"


def experiment_root():
    _path = output_dir()
    _path.mkdir(parents=True, exist_ok=True)
    return _path


def experiment_path():
    return experiment_root() / f"{args.experiment_id}/"


def load_model(file):
    state_dict = torch.load(file, map_location=device)
    _model = init_model()
    _model.load_state_dict(state_dict)
    return _model


def model_file_name(epoch):
    return f"epoch_{epoch}.pth"


def h5_file_path(_country_name: str) -> PosixPath:
    return PosixPath(data_root() / f"{_country_name}.h5")


def save_model(_model, _epoch):
    _path = experiment_path()
    _path.mkdir(parents=True, exist_ok=True)
    file = _path / model_file_name(_epoch)
    torch.save(_model.state_dict(), file)
    info(f"Model ({file}) saved at epoch {_epoch}.")


def plot(
    sample: Tensor,
) -> plt.Figure:
    image = np.rollaxis(sample[0][[3, 2, 1]].numpy(), 0, 3)
    image = np.clip(image / 2000, 0, 1)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image)
    ax.axis("off")
    return fig


def timeit(f):
    # if the timeit option is false, return the original function. (do not measure time)
    if not args.timeit:
        return f
    else:

        @wraps(f)
        def wrap(*args, **kw):
            ts = time()
            result = f(*args, **kw)
            te = time()
            info("Func: %s() took: %2.4f sec" % (f.__name__, te - ts))
            return result

    return wrap


# %% Define a custom dataset class CountryDataset


class CountryDataset(Dataset):
    def __init__(
        self,
        source: Union[SmallEarthNet, PosixPath] = None,
        transform=None,
        scale=False,
        _country_name: str = None,
    ):

        assert source is not None
        assert type(source) in [
            SmallEarthNet,
            PosixPath,
        ], f"Source type {type(source)} not supported"

        self.kind = None
        self.source = source

        if type(self.source) == PosixPath:
            self.kind = "hdf5"
        elif type(self.source) == SmallEarthNet:
            self.kind = "dataset"

        self.num_classes = None
        self.label_dset = None
        self.image_dset = None
        self.transform = transform
        self.scale = scale  # scale x values to be in the range 0-1
        self.len = None
        self.h5_file = h5_file_path(_country_name=_country_name)
        self.cache = args.cache

        assert self.cache in [
            True,
            False,
        ], "cache must be True or False"

        self.file = None

        if self.kind == "hdf5":
            info("Loading HDF5 dataset")
            self.h5_file = self.source
            assert self.h5_file.exists(), f"{self.h5_file} does not exist"
            self.load()
        elif self.kind == "dataset":
            info("Loading smallEarthNet dataset from folders")
            self.len = len(self.source)
            self.num_classes = self.source.num_classes

        info(
            "\n----------------------------------------------------------------------------------------\n"
            f"Dataset {_country_name} loaded with {self.len} samples, with cache sat to {self.cache}."
            "\n----------------------------------------------------------------------------------------\n"
        )

    def __del__(self):
        self.file.close()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.kind == "dataset":
            batch = self.source[idx]
            # Image shape:  torch.Size([12, 120, 120])
            x = batch["image"]
            y = batch["label"]
            if self.transform:
                x = self.transform(x)
            if self.scale:
                # scale to 0-1
                x -= x.min(1, keepdim=True)[0]
                x /= x.max(1, keepdim=True)[0]
            return x, y
        if self.kind == "hdf5":
            if self.cache:
                return self.load_from_cache(idx)
            else:
                return self.load_from_disk(idx)

    def load_from_cache(self, idx):
        image = torch.tensor(self.image_dset[idx])
        label = torch.tensor(self.label_dset[idx])
        if self.transform:
            image = self.transform(image)
        if self.scale:
            # scale to 0-1
            image -= image.min(1, keepdim=True)[0]
            image /= image.max(1, keepdim=True)[0]
        return image, label

    def load_from_disk(self, idx):
        image = torch.tensor(self.file.get("images")[idx])
        label = torch.tensor(self.file.get("labels")[idx])
        if self.transform:
            image = self.transform(image)
        if self.scale:  # scale to 0-1
            image -= image.min(1, keepdim=True)[0]
            image /= image.max(1, keepdim=True)[0]
        return image, label

    def set_transform(self, transform):
        self.transform = transform

    def n_out(self):
        return self.num_classes

    @timeit
    def load(self):
        self.file = h5py.File(self.h5_file, "r")
        if self.cache:
            # cache the dataset in memory
            # this is nuts, I know.
            info(
                "\n----------------------------------------------------------------------------------------\n"
                f"Caching {self.h5_file} in RAM... Wish RAM luck!"
                "\n----------------------------------------------------------------------------------------\n"
            )
            self.image_dset = self.file["images"][:]
            self.label_dset = self.file["labels"][:]
        else:
            self.image_dset = self.file.get("images")
            self.label_dset = self.file.get("labels")

        # set length of dataset and num_classes
        self.num_classes = self.label_dset.shape[1]
        self.len = len(self.image_dset)

        info(f"Length of dataset: {self.len}")
        info(f"Images dataset shape: {self.image_dset.shape}")
        info(f"Labels dataset shape: {self.label_dset.shape}")

    def save(self):
        # check if file exists
        if self.h5_file.exists():
            info(f"File {self.h5_file} exists. Skipping.")
            return
        else:
            info(f"File {self.h5_file} does not exist. Creating.")

            # Not related to training, just to save the dataset
            _params = {
                "batch_size": 128,  # is also the chunk size
                "shuffle": False,
                "num_workers": args.workers,
            }

            dl = DataLoader(self, **_params)
            # Create a new HDF5 file and write the data to it
            with tqdm(
                total=len(dl),
                desc="Saving dataset as h5 file",
                unit="datapoint",
                dynamic_ncols=True,
            ) as progress_bar:
                with h5py.File(self.h5_file, "a") as f:
                    for i, (images, labels) in enumerate(dl):
                        if i == 0:  # create datasets and write first batch
                            f.create_dataset(
                                "images",
                                data=images,
                                compression="gzip",
                                chunks=True,  # current size is (13, 2, 15, 30)
                                # TODO: in next life, pick the chunk size manually to speed read up.
                                #  Read this first:
                                #  https://www.oreilly.com/library/view/python-and-hdf5/9781491944981/ch04.html
                                maxshape=(None,) + images.shape[1:],
                            )
                            f.create_dataset(
                                "labels",
                                data=labels,
                                compression="gzip",
                                chunks=True,
                                maxshape=(None,) + labels.shape[1:],
                            )
                            progress_bar.update(1)
                            continue  # don't write the first batch again

                        # append to existing datasets
                        f["images"].resize((f["images"].shape[0] + images.shape[0]), axis=0)
                        f["images"][-images.shape[0] :] = images

                        f["labels"].resize((f["labels"].shape[0] + labels.shape[0]), axis=0)
                        f["labels"][-labels.shape[0] :] = labels
                        progress_bar.update(1)
                f.close()


# %% splits the dataset into training and validation sets


def split_train_val(
    dataset: CountryDataset,
    _params: Dict,
    seed: int = 42,
    validation_split: float = 0.2,
    _country_name: str = None,
):
    assert _country_name is not None, "Country name must be provided"
    mean, std = get_mean_std(_country_name, DataLoader(dataset, **_params))

    preprocess = transforms.Compose(
        [
            transforms.Normalize(mean=mean, std=std, inplace=False),
            # random vertical flip
            transforms.RandomVerticalFlip(p=0.5),
            # random horizontal flip
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )

    dataset.set_transform(preprocess)

    train_len = int(len(dataset) * (1 - validation_split))
    val_len = int(len(dataset) - train_len)
    _train_set, _val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(seed),
    )

    _train_loader = DataLoader(_train_set, **_params)
    _val_loader = DataLoader(_val_set, **_params)

    return _train_set, _val_set, _train_loader, _val_loader


# %% Check if a hdf5 country dataset exists
#  if it does, load it
#  if it doesn't, create it
#
# if h5_file_path(_country_name=country_name).exists():
#     info(f"Found h5 file: {h5_file_path(_country_name=country_name)}. Skipping creation of h5 file...")
#     info(f"Skipping creation of h5 file...")
# else:
#     info(f"No h5 file found: {h5_file_path(_country_name=country_name)}")
#     info("Trying to load dataset from folders...")
#     # Load the dataset from folders using small_earth SmallEarthNet
#     small_earth_set = SmallEarthNet(
#         root=str(country_root),
#         bands="s2",
#         download=False,
#         country=country_name,
#         num_classes=config["num_classes"],  # 19 or 43 number of classes
#     )
#     country_set = CountryDataset(source=small_earth_set)
#     info("Saving dataset as h5 file...")
#     country_set.save()  # save the dataset as h5 file


# %% Calculate mean and std if not provided


def get_mean_std(_country_name, _loader):
    def calculate_mean_std(loader):
        _channels_count = len(loader.dataset[0][0])
        _mean = torch.zeros(_channels_count)
        _std = torch.zeros(_channels_count)

        for images, _ in tqdm(loader, desc="Calculating mean and std", unit="batch"):
            # batch size (the last batch can have smaller size!)
            batch_size = images.size(0)
            images = images.view(batch_size, images.size(1), -1)
            _mean += images.mean(2).sum(0)
            _std += images.std(2).sum(0)

        _mean /= len(loader.dataset)
        _std /= len(loader.dataset)

        return {"mean": _mean.tolist(), "std": _std.tolist()}

    def cache_file_path(country_name: str):
        assert country_name is not None, "Country name must be provided"
        return data_root() / f"{country_name}_mean-std.yml"

    def write_cache(_country_name, _mean_std: dict):
        with open(cache_file_path(_country_name), "w") as outfile:
            yaml.dump(_mean_std, outfile, default_flow_style=False)
        info(f"Saved mean-std to {cache_file_path(_country_name)}")

    def read_cache(_country_name):
        assert _country_name is not None, "Country name must be provided"
        with open(cache_file_path(_country_name), "r") as infile:
            __mean_std = yaml.load(infile, Loader=yaml.FullLoader)
        info(f"Loaded mean-std from {cache_file_path(_country_name)}")
        return __mean_std

    if not cache_file_path(_country_name).exists():
        _mean_std = calculate_mean_std(_loader)
        write_cache(_country_name, _mean_std)
    else:
        _mean_std = read_cache(_country_name)
    return _mean_std["mean"], _mean_std["std"]


# %% Model definition


class ben_resnet50(nn.Module):
    def __init__(self, num_classes=19, device=torch.device("cpu")):
        super(ben_resnet50, self).__init__()
        tmp = models.resnet50(num_classes=config["num_classes"])
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        print("pretrained resnet50 has been loaded")
        self.model.fc = tmp.fc
        self.model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.final_layer = list(self.model.children())[-1]
        self.model.to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, start_layer_idx=0, logit=False):
        if start_layer_idx == 0:
            x = self.model(x)
        elif start_layer_idx == -1:
            if (len(x.shape)) == 2:
                x = self.final_layer(x)
            else:
                try:
                    x = x.reshape((x.shape[0], -1))
                    x = x = self.final_layer(x)
                except:
                    print("dimension not matching")
        else:
            print("the implementation for forward pass from layers before the last has not been implemented yet")
            return NotImplementedError

        results = {"output": self.sigmoid(x)}
        if logit:
            results["logit"] = x
        return results


def init_model(backbone: str = "resnet50"):
    if backbone == "resnet50" and args.fed == "gen":
        return ben_resnet50(device=device)
    elif backbone == "resnet50":
        _model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        tmp = models.resnet50(num_classes=config["num_classes"])
        _model.fc = tmp.fc  # replace the last layer of _model with the last layer of tmp
        _model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        _model.to(device)
        return _model
    elif backbone == "vit":
        _model = ViT(
            image_size=120,
            patch_size=12,
            num_classes=config["num_classes"],
            dim=1024,
            channels=12,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        )
        _model.to(device)
        return _model
    else:
        print("not implemented")


# %% Loss function initialization

criterion = nn.BCELoss()
# "Categorical Cross-Entropy loss or Softmax Loss is a Softmax activation plus a Cross-Entropy loss.
#   If we use this loss, we will train a CNN to output a probability over the C classes for each image.
#   It is used for multi-class classification.
#   What you want is multi-label classification, so you will use Binary Cross-Entropy Loss or
#   Sigmoid Cross-Entropy loss. It is a Sigmoid activation plus a Cross-Entropy loss.
#   Unlike Softmax loss it is independent for each vector component (class), meaning that the loss computed for
#   every CNN output vector component is not affected by other component values.
#   That’s why it is used for multi-label classification, where the insight of an element belonging to a certain
#   class should not influence the decision for another class.
#   Now for handling class imbalance, you can use weighted Sigmoid Cross-Entropy loss.
#   So you will penalize for wrong prediction based on the number/ratio of positive examples."
#
# Source: https://stackoverflow.com/questions/59336899/which-loss-function-and-metrics-to-use-for-multi-label
# -classification-with-very?answertab=trending#tab-top


# %% Metrics class definition


class Metrics:
    def __init__(
        self,
        threshold=0.5,
        num_classes: int = None,
        _criterion=nn.BCELoss(),
        _device: torch.device = torch.device("cuda"),
    ):
        self.acc = 0.0  # accuracy
        self.prec = 0.0  # precision
        self.rec = 0.0  # recall
        self.f1 = 0.0  # f1 score
        self.dice = 0.0  # dice coefficient
        self.loss = 0.0  # loss
        self.mcc = 0.0  # matthews correlation coefficient
        self.thresh = threshold
        self.update_count = 0
        self.sample_count = 0
        self.num_classes = num_classes
        self.device = _device
        self.criterion = _criterion
        self.lrloss = 0.0  # LabelRankingLoss
        self.lrap = 0.0  # LabelRankingAveragePrecision

        # average=micro says the function to compute f1 by considering total true positives, false negatives
        #   and false positives (no matter of the prediction for each label in the dataset)
        #
        # average=macro says the function to compute f1 for each label, and returns the average without
        #   considering the proportion for each label in the dataset.
        #
        # average=weighted says the function to compute f1 for each label, and returns the average
        #   considering the proportion for each label in the dataset.
        #
        # average=samples says the function to compute f1 for each instance,
        #   and returns the average. Use it for multilabel classification.
        #
        # Source: https://stackoverflow.com/a/55759038/216953

        self._f1 = F1Score(
            num_classes=self.num_classes,
            multiclass=False,
            average="samples",
            threshold=0.5,
        ).to(self.device)

        self._mcc = MatthewsCorrCoef(
            num_classes=self.num_classes,
            threshold=0.5,
        ).to(self.device)

        self._dice = Dice(
            num_classes=self.num_classes,
            multiclass=False,
            average="samples",
            threshold=0.5,
        ).to(self.device)

        self._precision = Precision(
            num_classes=self.num_classes,
            multiclass=False,
            average="samples",
            threshold=0.5,
        ).to(self.device)

        self._recall = Recall(
            num_classes=self.num_classes,
            multiclass=False,
            average="samples",
            threshold=0.5,
        ).to(self.device)

        self._accuracy = Accuracy(
            num_classes=self.num_classes,
            multiclass=False,
            average="samples",
            threshold=0.5,
        ).to(self.device)

        self._lrloss = LabelRankingLoss(
            num_classes=self.num_classes,
            threshold=0.5,
        ).to(self.device)

        self._lrap = LabelRankingAveragePrecision(
            num_classes=self.num_classes,
            threshold=0.5,
        ).to(self.device)

        def label_error(y_pred, y_true):
            """
            Compute the label error for each class.
            When a prediction is incorrect, it accumulates the error as positive value.
            """
            return torch.logical_xor(y_pred.int(), y_true.int()).sum(dim=0)

        self.label_error = label_error

        def label_frequency(y_true: torch.Tensor):
            """
            Compute the label frequency for each class.
            when class is absent, add -1 from the frequency.
            when class is present, add 1 to the frequency.
            """
            y_true[y_true == 0] = -1  # replace 0 with -1
            return y_true.sum(dim=0)  # sum over the batch dimension and return

        self.label_frequency = label_frequency

        self.lf = torch.zeros(self.num_classes).int().to(self.device)

    def update(self, preds: torch.Tensor, _target: torch.Tensor):
        sigmoid = nn.Sigmoid()
        if args.fed == "gen":
            _preds = (preds["output"] > self.thresh).int()
            self.loss += self.criterion(preds["output"], _target.float()).item()
        else:
            _preds = (sigmoid(preds) > self.thresh).int()
            self.loss += self.criterion(preds.sigmoid(), _target.float()).item()
        self.f1 += self._f1(_preds, _target)
        self.mcc += self._mcc(_preds, _target)
        self.dice += self._dice(_preds, _target)
        self.lrap += self._lrap(_preds, _target)
        self.rec += self._recall(_preds, _target)
        self.acc += self._accuracy(_preds, _target)
        self.lrloss += self._lrloss(_preds, _target)
        self.prec += self._precision(_preds, _target)

        self.lf += self.label_frequency(_target)

        self.update_count += 1
        self.sample_count += _target.size(dim=0)  # number of samples

    def __str__(self) -> str:
        return (
            f"\tF1: {self.f1 / self.update_count:.8f}\n"
            f"\tMCC: {self.mcc / self.update_count:.8f}\n"
            f"\tDice: {self.dice / self.update_count:.8f}\n"
            f"\tPrecision: {self.prec / self.update_count:.8f}\n"
            f"\tRecall: {self.rec / self.update_count:.8f}\n"
            f"\tAccuracy: {self.acc / self.update_count:.8f}\n"
            f"\tValLoss: {self.loss / self.update_count:.8f}\n"
            f"\tLabelRankingLoss: {self.lrloss / self.update_count:.8f}\n"
            f"\tLabelRankingAveragePrecision: {self.lrap / self.update_count:.8f}\n"
            # f"\tLabelFrequency: {add_class_names(self.lf / self.sample_count)}\n"
        )

    def get(self) -> Dict[str, float]:
        return {
            "f1": self.f1 / self.update_count,
            "mcc": self.mcc / self.update_count,
            "dice": self.dice / self.update_count,
            "precision": self.prec / self.update_count,
            "recall": self.rec / self.update_count,
            "accuracy": self.acc / self.update_count,
            "val_loss": self.loss / self.update_count,
            "lrloss": self.lrloss / self.update_count,
            "lrap": self.lrap / self.update_count,
            # "label_frequency": add_class_names(self.lf / self.sample_count),
        }

    def reset(self):
        self.acc = 0.0  # accuracy
        self.prec = 0.0  # precision
        self.rec = 0.0  # recall
        self.f1 = 0.0  # f1 score
        self.dice = 0.0  # dice coefficient
        self.loss = 0.0  # loss
        self.mcc = 0.0  #
        self.lrloss = 0.0  # LabelRankingLoss
        self.lrap = 0.0  # LabelRankingAveragePrecision
        self.update_count = 0
        self.sample_count = 0


# %% Evaluator


def evaluator(_model, _loader):
    _model.eval()  # set model to evaluation mode
    _metrics = Metrics(num_classes=config["num_classes"], _device=device)
    for _data, _target in tqdm(_loader, total=len(_loader), desc="Evaluator", unit="Batch", dynamic_ncols=True):
        _data = _data.type(torch.float32).to(device)
        _target = _target.type(torch.int8).to(device)
        _preds = _model(_data)
        _metrics.update(_preds, _target)
    return _metrics


# %% Training loop function


def train(
    _model: nn.Module,
    _epochs: int = None,
    _train_loader: DataLoader = None,
    _val_loader: DataLoader = None,
    _wandb_run: wandb = None,
    _country: str = None,
    _round: int = None,
    _optimizer: torch.optim.Optimizer = None,
    _test_sets: List[str] = None,
) -> nn.Module:
    info(f"Training country: {_country}")
    info(f"Testing against: {_test_sets}")

    for _epoch in tqdm(range(_epochs), desc=f"{_country} Epochs", unit="Epoch", dynamic_ncols=True):
        # Training
        _model.train()  # set model to training mode
        # wandb.watch(_model, criterion=criterion, log="all", log_freq=50)
        train_loss = 0.0
        for _data, targets in tqdm(
            _train_loader, total=len(_train_loader), desc=f"Training", unit="Batch", dynamic_ncols=True
        ):
            _data, targets = _data.to(device), targets.to(device)

            _optimizer.zero_grad()
            _preds = _model.forward(_data)  # Forward pass
            loss = criterion(_preds.sigmoid(), targets.float())
            loss.backward()  # Compute the gradients
            _optimizer.step()  # Update the weights
            train_loss += loss.item()  # Accumulate the loss

        # Validation
        __metrics = evaluator(_model, _val_loader)

        info(f"\n\nMetrics ({_country}):\n" f"\tTrainLoss: {train_loss / len(_train_loader):.8f}\n" f"{__metrics}")

        _logs = {
            f"{_country}/train_loss": train_loss / len(_train_loader),
        }

        # add _country as a prefix to the metrics dict
        for k, v in __metrics.get().items():
            _logs[f"{_country}/{k}"] = v

        _wandb_run.log(_logs, step=_round)
    return _model


# %% FedBN and FedAvg implementation

# Training datasets sizes
# Switzerland: train dataset size:  3952
# Austria: train dataset size:      8394
# Serbia: train dataset size:       11865
# Lithuania: train dataset size:    12623


def fedbn(server: nn.Module, _clients: Dict[str, nn.Module]) -> Tuple[nn.Module, Dict[str, nn.Module]]:
    _clients_models = list(_clients.values())

    # Count the number of clients
    _client_count = len(_clients_models)

    # for example, if we have 3 clients, then the weights are [1/3, 1/3, 1/3]
    _client_weights = [1 / _client_count for _ in range(_client_count)]

    with torch.no_grad():
        for key in server.state_dict().keys():
            if "bn" not in key:
                temp = torch.zeros_like(server.state_dict()[key], dtype=torch.float32)
                for client_idx in range(_client_count):
                    temp += _client_weights[client_idx] * _clients_models[client_idx].state_dict()[key]
                server.state_dict()[key].data.copy_(temp)
                for client_idx in range(_client_count):
                    _clients_models[client_idx].state_dict()[key].data.copy_(server.state_dict()[key])

    # make clients list a dict with country as key
    _clients = {__country: client for __country, client in zip(list(_clients.keys()), _clients_models)}
    return server, _clients


def fedavg(server: nn.Module, _clients: Dict[str, nn.Module]) -> tuple[nn.Module, Dict[str, nn.Module]]:
    _clients_models = list(_clients.values())

    # Count the number of clients
    _client_count = len(_clients_models)

    # for example, if we have 3 clients, then the weights are [1/3, 1/3, 1/3]
    _client_weights = [1 / _client_count for _ in range(_client_count)]

    with torch.no_grad():
        for key in server.state_dict().keys():
            temp = torch.zeros_like(server.state_dict()[key], dtype=torch.float32)
            for client_idx in range(_client_count):
                temp += _client_weights[client_idx] * _clients_models[client_idx].state_dict()[key]
            server.state_dict()[key].data.copy_(temp)
            for client_idx in range(_client_count):
                _clients_models[client_idx].state_dict()[key].data.copy_(server.state_dict()[key])

    # make clients list a dict with country as key
    _clients = {__country: client for __country, client in zip(list(_clients.keys()), _clients_models)}
    return server, _clients


def communicate(
    _server: nn.Module, _clients: Dict[str, nn.Module], _external_epoch: int = None
) -> tuple[nn.Module, Dict[str, nn.Module]]:
    output_model = None

    if args.fed not in ["bn", "avg", "gen"]:
        raise ValueError(f"Federation type {args.fed} not supported")
    if args.fed == "bn":
        output_model = fedbn(_server, _clients)
    if args.fed == "avg":
        output_model = fedavg(_server, _clients)
    if args.fed == "gen":
        pass

    assert output_model is not None, "output_model is None"
    return output_model


# %% FedBN and FedAvg execution


def new_wandb_run(_params: Dict, _config: Dict):
    # Weights & Biases update config
    run = wandb.init(
        project=f"fedgen",
        entity="fedlearning",
        reinit=False,
        tags=[args.experiment_id, config["backbone"], config["fed"]],
        settings=wandb.Settings(start_method="fork"),
    )
    run.name = f"fed{config['fed']}_clients{args.fed_clients}_{config['backbone']}_{config['rounds']}rounds_{args.experiment_id}_test_sets{args.test_sets} "
    info(f"Wandb run name: {wandb.run.name}")
    run.config.update(_config)
    run.config.update(_params)
    return run


def save_server(_model):
    # create a new directory for the server model
    _path = Path(args.fed_servers_output_dir)
    _path.mkdir(parents=True, exist_ok=True)
    name = f"{args.experiment_id}_{args.fed_method}_{'-'.join(args.fed_clients)}_{config['backbone']}_{args.epochs}-epochs.pt"
    file = _path / f"{name}.pth"
    torch.save(_model.state_dict(), file)
    info(f"Saved model to {file}")


# load_client returns a list of ordered models by epoch of a specific country/client
def load_client(_path, _country):
    # FedBN: load client model
    _models = []
    for _epoch in range(config["fed_epochs"]):
        _file = _path / model_file_name(_epoch)
        info(f"Loading {_country} model at {_epoch} epoch from: {_file}")
        _model = load_model(_file)
        _models.append(_model)
    return _models


@timeit
def load_single_epoch(_path: PosixPath) -> nn.Module:
    _file = _path
    _model = load_model(_file)
    info(f"{_file.parent.name}: Loaded {_file.name}")
    return _model


# Returns a list of clients models paths
def get_clients_paths(_path, _country):
    _paths = []
    for _epoch in range(config["fed_epochs"]):
        _file = _path / model_file_name(_epoch)
        if _file.exists():
            _paths.append(_file)
        else:
            info(f"Error: File {_file} is missing")
            raise FileNotFoundError
    return _paths


def save_client_model(_model, _country, _epoch):
    _path = experiment_path() / _country
    _path.mkdir(parents=True, exist_ok=True)
    file = _path / model_file_name(_epoch)
    torch.save(_model.state_dict(), file)
    info(f"Model ({file}) saved at epoch {_epoch}.")


def load_client_model(path: Union[PosixPath, None]) -> nn.Module:
    if path is None:
        info("No client model loaded, initializing new model")
        return init_model(config["backbone"])
    else:
        state_dict = torch.load(path, map_location=device)
        _model = init_model(config["backbone"])
        _model.load_state_dict(state_dict)
        info(f"Loaded model from {path}")
        return _model


def optimizer(_model):
    return torch.optim.Adam(
        _model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        amsgrad=True,
    )


@timeit
def get_train_val_loader(_country_name: str) -> Dict[str, DataLoader]:
    _, _, train_loader, val_loader = split_train_val(
        CountryDataset(source=h5_file_path(_country_name=_country_name), _country_name=_country_name),
        params,
        seed=config["seed"],
        validation_split=config["validation_split"],  # Only used for as evaluation dataset
        _country_name=_country_name,
    )
    return {"train": train_loader, "val": val_loader}


if __name__ == "__main__":
    """
    The code above is for a federated learning experiment. The clients are the countries in the args.fed_clients
    list and the server is initialized as the first client's model. The server and clients communicate at each round
    of federated training.
    """

    # Params dict is used to create dataLoaders
    params = {
        "batch_size": config["batch_size"],
        "shuffle": True,
        "num_workers": args.workers,
    }

    wandb_run = new_wandb_run(_params=params, _config=config)

    server = None

    clients_models = {}
    for _country in args.fed_clients:
        info(f"Initializing {_country} model")
        clients_models[_country] = init_model(config["backbone"])

    clients_loaders = {}
    # append args.test_sets + args.fed_clients lists
    for _country in args.test_sets + args.fed_clients:
        info(f"Loading {_country} train and val loaders...")
        clients_loaders[_country] = get_train_val_loader(_country_name=_country)

    def combine_test_loaders(_test_sets: List[str], _clients_loaders: Dict, _params: Dict) -> DataLoader:
        _test_loaders = []
        for _test_set in _test_sets:
            _test_loaders.append(_clients_loaders[_test_set]["val"])

        _test_datasets = [loader.dataset for loader in _test_loaders]
        _test_loader = DataLoader(ConcatDataset(_test_datasets), **_params)
        return _test_loader

    test_loader = combine_test_loaders(args.test_sets, clients_loaders, params)

    if args.fed == "gen":
        # this contains the r
        with open(args.fedgen_config) as file:
            fedgen_dict = yaml.safe_load(file)
        config.update(fedgen_dict)
        # label freq requried by fedgen algorithm
        with open("config/label_frequencies.yaml") as lf:
            label_freq = yaml.safe_load(lf)
        # these extra  params are required by fedgen class
        config["label_frequencies"] = label_freq
        config["dataset"] = "bigearthnet2"
        config["device"] = device
        config["train"] = True
        config["epochs"] = args.epochs
        config["clients"] = args.fed_clients
        id_countries = {i + 1: args.fed_clients[i] for i in range(len(args.fed_clients))}
        config["id_countries"] = id_countries

        server_model = init_model(config["backbone"])
        train_loaders = {i: clients_loaders[country]["train"] for (i, country) in id_countries.items()}
        data = [list(train_loaders.keys()), train_loaders, test_loader]
        metrics = Metrics(num_classes=19, _device=device)
        fedgen = FedGen(config, data, server_model, metrics, torch.manual_seed(42))
        fedgen.train_test()
    else:
        for _round in tqdm(
            range(config["rounds"]),
            unit="Round",
            desc=f"fed{config['fed']} round",
            dynamic_ncols=True,
        ):
            # train the model of each client (country)
            # i.e. if args.fed_clients = ["Switzerland", "Serbia"] it train a model for each country
            # then store it in clients dict like {"Switzerland": model, "Serbia": model}
            for idx, _country in enumerate(args.fed_clients):
                # Train and evaluate the model
                clients_models[_country] = train(
                    _epochs=config["epochs"],  # only one epoch for each client
                    _round=_round,
                    _country=_country,
                    _model=clients_models[_country],
                    _train_loader=clients_loaders[_country]["train"],
                    _val_loader=test_loader,
                    _wandb_run=wandb_run,
                    _optimizer=optimizer(_model=clients_models[_country]),
                    _test_sets=args.test_sets,
                )
                # Init server model as one of the clients
                if server is None:
                    server = list(clients_models.values())[0]

                # Federated training starts here
                # Communicate the trained models to the server and back to the clients'
                server, clients_models = communicate(_server=server, _clients=clients_models, _external_epoch=_round)

            info(f"Experiment id: {args.experiment_id}")
            info(">>>>>>>Done.<<<<<<<")
# Finland Portugal Ireland Lithuania Serbia Austria Switzerland
