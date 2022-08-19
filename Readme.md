# Federated Learning – Method Evaluation

A Comparative Analysis of Federated Domain Adaptation Methods in the Context of Remote Sensing Image Classification

## About the BigEarthNet dataset

To prepare the test bed of this experiment, we used the BigEarthNet-S2 (Sumbul et al., 2019) dataset (formerly called BigEarthNet)
BigEarthNet-S2 was built using Sentinel-2 image correction, between June 2017 and May 2018, samples from ten countries
(Austria, Belgium, Finland, Ireland, Kosovo, Lithuania, Luxembourg, Portugal and Serbia, Switzerland).
However, for the purpose of this experiment, we used only a subset of the countries (Switzerland, Austria, Finland, Ireland, Lithuania, Portugal, and Serbia.)

The size of the image patches on the ground is 1.2 x 1.2 kilometers, with variable image sizes depending on the channel resolution.
The dataset has 43 unbalanced classes.

## Execution Pipeline

![](docs/flow_fed.png)

### Technical documentation

The technical documentation of the code is available in HTML format. You can find it in the `./docs/` folder, or download it [here](https://github.com/amer/federated-learning/blob/c2592f16c82898c9c32d7cdea1b3a7cb2b81a2a2/docs/Technical%20documentation.zip). 

### Used bands
- B01: Coastal aerosol; 60m
- B02: Blue; 10m
- B03: Green; 10m
- B04: Red; 10m
- B05: Vegetation red edge; 20m
- B06: Vegetation red edge; 20m
- B07: Vegetation red edge; 20m
- B08: NIR; 10m
- B09: Water vapor; 60m
- B11: SWIR; 20m
- B12: SWIR; 20m

## Requirements
- Ubuntu 20.04 LTS
- Python 3.10.4
- CUDA enabled GPU with at least 40GB of memory
- CUDA Version: 11.7
- Minimum 100GB of RAM
- poetry (https://poetry.eustace.io/)

## Dependencies

- `sudo apt-get install libatlas-base-dev gfortran`
- [PROJ](https://proj.org/) is a generic coordinate transformation software that transforms geospatial coordinates from one coordinate reference system (CRS) to another.
- [GDAL](https://gdal.org/) is a translator library for raster and vector geospatial data formats that is released under an MIT style Open Source License by the Open Source Geospatial Foundation.
- [HDF5](https://docs.h5py.org/en/stable/build.html) is a library for storing and managing data in a self-describing, portable, and extensible format.
    ```bash
    # On MacOS & Linux. You need to check the directory of the HDF5 library.
    brew install hdf5
    # $(readlink -f $(brew --prefix hdf5)) should be something like /opt/homebrew/Cellar/hdf5/1.12.2
    export HDF5_DIR=$(readlink -f $(brew --prefix hdf5))
    ```

## Installation

```bash
git clone git@github.com:amer/federated-learning.git
cd federated-learning
pyenv install 3.10.4
poetry install --no-dev
poetry run python main.py --help
```

## Dataset preparation

Before running the code, you need to rename the `.env.example` file to `.env` and fill in the required parameters.

Additionally, you need to download the BigEarthNet dataset, extract it, and run the following script:

```bash
python prepare-data-step1.py
```

This will `{home}/BigEarthNet/BigEarthNet-v1.0` will extract a subset of the BigEarthNet dataset to and stor it in `{home}/BigEarthNet/step1`.
Each country will have its own directory in `{home}/BigEarthNet/step1` base on `scripts/patch_country_summer.csv` file.


The last step to prepare the dataset is to run the following script:

```bash
poetry run python generate-csv-train-files.py
```

This will generate the `<CountryName>-train.csv` in the corresponding country directory. These files are needed for loading the dataset.

## Usage

The following command is an example run the experiment on the TinyImageNet dataset:

```bash
# This script is a smoke test.
poetry run python main.py \
  --fed bn \
  --scale false \
  --cache true \
  --batch_size 170 \
  --rounds 2 \
  --timeit true \
  --epochs 1 \
  --test_sets TinyFinland \
  --fed_clients TinyFinland TinyFinland\
  --backbone resnet50 \
  --device cuda \
  --workers 4 \
  --data_root "data/" \
  --output_dir "data/experiments/clients/" \
  --fed_servers_output_dir "data/experiments/servers/" \
  --fed_clients_source_dir "/data/experiments/clients/"
```

## for more information on the parameters, run the following command:

```bash
poetry run python main.py --help
```

```bash
Using python3.10 (3.10.4)
main: Logger initialized
main: User: amer
main: torch version: 1.12.1+cu102
main: torchvison version: 0.13.1+cu102
usage: main.py [-h]
               [--test_sets {Finland,Portugal,Ireland,Lithuania,Serbia,Austria,Switzerland,TinyFinland,TinyFinland2,TinyFinland3,TinyFinland4,UnitedNations} [{Finland,Portugal,Ireland,Lithuania,Serbia,Austria,Switzerland,TinyFinland,TinyFinland2,TinyFinland3,TinyFinland4,UnitedNations} ...]]
               [--experiment_id EXPERIMENT_ID] [--backbone {resnet50,vit}] [--data_root DATA_ROOT] [--output_dir OUTPUT_DIR]
               [--batch_size BATCH_SIZE] [--workers WORKERS] [--device {cpu,cuda,mps}] [--epochs EPOCHS] [--rounds ROUNDS]
               [--scale {true,false,True,False}]
               [--fed_clients {Finland,Portugal,Ireland,Lithuania,Serbia,Austria,Switzerland,TinyFinland,TinyFinland2,TinyFinland3,TinyFinland4,UnitedNations} [{Finland,Portugal,Ireland,Lithuania,Serbia,Austria,Switzerland,TinyFinland,TinyFinland2,TinyFinland3,TinyFinland4,UnitedNations} ...]]
               [--fed_servers_output_dir FED_SERVERS_OUTPUT_DIR] [--fed_clients_source_dir FED_CLIENTS_SOURCE_DIR]
               [--timeit {true,false,True,False}] [--cache {true,false,True,False}] --fed {avg,bn} [--mode MODE] [--port PORT]

SmallEarthNet

options:
  -h, --help            show this help message and exit
  --test_sets {Finland,Portugal,Ireland,Lithuania,Serbia,Austria,Switzerland,TinyFinland,TinyFinland2,TinyFinland3,TinyFinland4,UnitedNations} [{Finland,Portugal,Ireland,Lithuania,Serbia,Austria,Switzerland,TinyFinland,TinyFinland2,TinyFinland3,TinyFinland4,UnitedNations} ...]
                        Available countries ['Finland', 'Portugal', 'Ireland', 'Lithuania', 'Serbia', 'Austria', 'Switzerland',
                        'TinyFinland', 'TinyFinland2', 'TinyFinland3', 'TinyFinland4', 'UnitedNations']. it can be an array of countries
  --experiment_id EXPERIMENT_ID
                        An id to identify multiple runs of the same experiment. Used as a tag in wandb.
  --backbone {resnet50,vit}
                        Backbone model to use
  --data_root DATA_ROOT
                        Where all the counties folders or/and the hdf5 files, *-train.csv, *_mean-std.yml files are stored.
  --output_dir OUTPUT_DIR
                        Where to save the output, e.g. the models
  --batch_size BATCH_SIZE
                        Batch size
  --workers WORKERS     Number of workers to load the data
  --device {cpu,cuda,mps}
                        Backbone model to use
  --epochs EPOCHS       Number of epochs to train inside each federated training cycle(i.e. for each round of federated training the model
                        is trained for this many epochs)
  --rounds ROUNDS       (Communication rounds) Number of rounds the clients are going to communicate with the server
  --scale {true,false,True,False}
                        When loading a data point, scale its values to be between 0 and 1
  --fed_clients {Finland,Portugal,Ireland,Lithuania,Serbia,Austria,Switzerland,TinyFinland,TinyFinland2,TinyFinland3,TinyFinland4,UnitedNations} [{Finland,Portugal,Ireland,Lithuania,Serbia,Austria,Switzerland,TinyFinland,TinyFinland2,TinyFinland3,TinyFinland4,UnitedNations} ...]
                        Countries to use as clients. e.g --fed_clients Finland Portugal. Minimum of 2 clients are required.Available
                        countries ['Finland', 'Portugal', 'Ireland', 'Lithuania', 'Serbia', 'Austria', 'Switzerland', 'TinyFinland',
                        'TinyFinland2', 'TinyFinland3', 'TinyFinland4', 'UnitedNations']
  --fed_servers_output_dir FED_SERVERS_OUTPUT_DIR
                        Where to save the output of the server
  --fed_clients_source_dir FED_CLIENTS_SOURCE_DIR
                        Where to find the stored models after each epoch for the clients
  --timeit {true,false,True,False}
                        Time certain functions
  --cache {true,false,True,False}
                        Cache the dataset in RAM when before training
  --fed {avg,bn}        Federated learning method
  --mode MODE
  --port PORT
```

## Smoke test

Run the following command to test the installation:

```bash
chmod +x smoke-test.sh
./smoketest.sh
```

## Prepare data
- Split the data by country using `prepare_data_step1.py`
- Generate the train csv files using `generate_train_csv.py`. Used by `CountryDataset` class

## Contribution guidelines
- Would be nice to use [Conventional Commits](https://conventionalcommits.org/) (Optional but highly appreciated). Conventional commits  Way to make your commits more readable and maintainable.


## Citations

```bibtex
@article{Sumbul2019BigEarthNetAL,
  title={BigEarthNet: A Large-Scale Benchmark Archive For Remote Sensing Image Understanding},
  author={Gencer Sumbul and Marcela Charfuelan and Beg{"u}m Demir and Volker Markl},
  journal={CoRR},
  year={2019},
  volume={abs/1902.06148}
}
```

```bibtex
@article{DBLP:journals/corr/abs-2102-07623,
  author    = {Xiaoxiao Li and Meirui Jiang and Xiaofei Zhang and Michael Kamp and Qi Dou},
  title     = {FedBN: Federated Learning on Non-IID Features via Local Batch Normalization},
  journal   = {CoRR},
  volume    = {abs/2102.07623},
  year      = {2021},
  url       = {https://arxiv.org/abs/2102.07623},
  eprinttype = {arXiv},
  eprint    = {2102.07623},
  timestamp = {Thu, 14 Oct 2021 09:14:35 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2102-07623.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```bibtex
@article{https://doi.org/10.48550/arxiv.1602.05629,
  doi = {10.48550/ARXIV.1602.05629},
  url = {https://arxiv.org/abs/1602.05629},
  author = {McMahan, H. Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and Arcas, Blaise Agüera y},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Communication-Efficient Learning of Deep Networks from Decentralized Data},
  publisher = {arXiv},
  year = {2016},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

```bibtex
@article{https://doi.org/10.48550/arxiv.2105.10056,
  doi = {10.48550/ARXIV.2105.10056},
  url = {https://arxiv.org/abs/2105.10056},
  author = {Zhu, Zhuangdi and Hong, Junyuan and Zhou, Jiayu},
  keywords = {Machine Learning (cs.LG), Distributed, Parallel, and Cluster Computing (cs.DC), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Data-Free Knowledge Distillation for Heterogeneous Federated Learning},
  publisher = {arXiv},
  year = {2021},
  copyright = {Creative Commons Zero v1.0 Universal}
}
```