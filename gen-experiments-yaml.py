countries = [
    # "UnitedNations",
    "Finland",
    "Portugal",
    "Ireland",
    "Lithuania",
    "Serbia",
    "Austria",
    "Switzerland",
    # "TinyFinland",
]

#  -
#    run_mode: fed
#    batch_size: 8
#    workers: 0
#    device: cpu
#    country: Portugal # in fed mode this is the country evaluate against.
#    backbone: resnet50
#    fed_method: fedbn
#    fed_epochs: 27
#    fed_clients: [ Switzerland, Austria, Serbia, Finland, Ireland, Lithuania ]
#    fed_servers_output_dir: "/home/ubuntu/BigEarthNet/experiments/servers/"
#    fed_clients_source_dir: "/home/ubuntu/BigEarthNet/experiments/clients/"
#    data_root: "/home/ubuntu/BigEarthNet/datasets/"


yaml_experiment = "runs:"

backbones = [
    "vit",
    "resnet50",
]

backbone = 256

for country in countries:
    for backbone in backbones:
        if backbone == "vit":
            batch_size = 350
        if backbone == "resnet50":
            batch_size = 512
        yaml_experiment += f"""
        
  -
    run_mode: train
    country: {country}
    batch_size: {batch_size}
    workers: 4
    epochs: 20
    backbone: {backbone}
    device: cuda
    data_root: "/home/ubuntu/BigEarthNet/datasets/"
    output_dir: "/home/ubuntu/BigEarthNet/experiments/clients/"
    timeit: false
    cache: true
  """

print(yaml_experiment)
