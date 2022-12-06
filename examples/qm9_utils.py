from __future__ import annotations

import json
import os
from collections import namedtuple
from random import seed as python_seed

import torch
import yaml
from dgl.data import QM9EdgeDataset
from dgl.dataloading import GraphDataLoader
from dgl.random import seed as dgl_seed
from munch import Munch, munchify
from numpy.random import seed as numpy_seed
from torch.utils.data import random_split


def prepare_munch_object(path: str) -> Munch:
    with open(path) as f:
        munch_object = munchify(yaml.load(f, Loader=yaml.FullLoader))

    return munch_object


def prepare_config(path: str) -> Munch:
    config = prepare_munch_object(path)

    # for k, v in config.model.items():
    #     config[k] = prepare_munch_object(f'{root}/{k}/{v}.yaml')

    return config


def compute_data_stats(dataset) -> tuple:
    graphs, targets = zip(*dataset)
    targets = torch.cat(targets)

    z_mean_list = []
    num_bond_mean_list = []

    for g in graphs:
        z_mean_list.append(torch.mean(g.ndata["attr"]))
        temp = 0
        for ii in range(g.num_nodes()):
            temp += len(g.successors(ii))
        num_bond_mean_list.append(torch.tensor(temp / g.num_nodes()))

    data_std, data_mean = torch.std_mean(targets)

    data_zmean = torch.mean(torch.stack(z_mean_list))
    num_bond_mean = torch.mean(torch.stack(num_bond_mean_list))

    return data_std, data_mean, data_zmean, num_bond_mean


def prepare_data(config: Munch) -> tuple:
    print("## Started data processing ##")

    if config.data.dataset == "qm9":
        dataset = QM9EdgeDataset(**config.data.source)

    val_size = config.data.split.val_size
    test_size = config.data.split.test_size
    train_size = len(dataset) - val_size - test_size

    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

    data_std, data_mean, data_zmean, num_bond_mean = compute_data_stats(train_data)

    data = namedtuple("data", ["train", "val", "test", "std", "mean"])

    data.train = train_data
    data.val = val_data
    data.test = test_data
    data.std = data_std
    data.mean = data_mean
    data.z_mean = data_zmean
    data.num_bond_mean = num_bond_mean

    print("## Finished data processing ##")

    return data


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    python_seed(seed)
    numpy_seed(seed)
    dgl_seed(seed)


def create_dataloaders(config: Munch, data: tuple):
    dataloaders = namedtuple("Dataloaders", ["train", "val", "test"])

    dataloaders.train = GraphDataLoader(
        data.train,
        pin_memory=False,
        batch_size=config.data.batch_size
        # **config.experiment.train,
    )
    dataloaders.val = GraphDataLoader(data.val)  # , **config.experiment.val)
    dataloaders.test = GraphDataLoader(data.test)  # , **config.experiment.test)

    return dataloaders


class StreamingJSONWriter:
    """
    Serialize streaming data to JSON.

    This class holds onto an open file reference to which it carefully
    appends new JSON data. Individual entries are input in a list, and
    after every entry the list is closed so that it remains valid JSON.
    When a new item is added, the file cursor is moved backwards to overwrite
    the list closing bracket.
    """

    def __init__(self, filename, encoder=json.JSONEncoder):
        if os.path.exists(filename):
            self.file = open(filename, "r+")
            self.delimeter = ","
        else:
            self.file = open(filename, "w")
            self.delimeter = "["
        self.encoder = encoder

    def dump(self, obj):
        """
        Dump a JSON-serializable object to file.
        """
        data = json.dumps(obj, cls=self.encoder)
        close_str = "\n]\n"
        self.file.seek(max(self.file.seek(0, os.SEEK_END) - len(close_str), 0))
        self.file.write(f"{self.delimeter}\n    {data}{close_str}")
        self.file.flush()
        self.delimeter = ","

    def close(self):
        self.file.close()
