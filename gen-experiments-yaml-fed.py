# countries = {
#     "UnitedNations": [
#         "Austria",
#         "Serbia",
#         "Finland",
#         "Ireland",
#         "Lithuania",
#         "Portugal",
#         "Switzerland",
#     ],
#     "Finland": [
#         "Austria",
#         "Serbia",
#         "Ireland",
#         "Lithuania",
#         "Portugal",
#         "Switzerland",
#     ],
# "Portugal": [
#     "Austria",
#     "Serbia",
#     "Finland",
#     "Ireland",
#     "Lithuania",
#     "Switzerland",
# ],
# "Ireland": [
#     "Austria",
#     "Serbia",
#     "Finland",
#     "Lithuania",
#     "Portugal",
#     "Switzerland",
# ],
# "Lithuania": [
#     "Austria",
#     "Serbia",
#     "Finland",
#     "Ireland",
#     "Portugal",
#     "Switzerland",
# ],
# "Serbia": [
#     "Austria",
#     "Finland",
#     "Ireland",
#     "Lithuania",
#     "Portugal",
#     "Switzerland",
# ],
# "Austria": [
#     "Serbia",
#     "Finland",
#     "Ireland",
#     "Lithuania",
#     "Portugal",
#     "Switzerland",
# ],
# "Switzerland": [
#     "Austria",
#     "Serbia",
#     "Finland",
#     "Ireland",
#     "Lithuania",
#     "Portugal",
# ],
# }

# countries = {
#     "UnitedNations": [
#         "Austria",
#         "Serbia",
#         "Finland",
#         "Ireland",
#         "Lithuania",
#         "Portugal",
#         "Switzerland",
#     ],
#     "Finland": [
#         "Austria",
#         "Serbia",
#         "Ireland",
#         "Lithuania",
#         "Portugal",
#         "Switzerland",
#     ],
#     "Portugal": [
#         "Austria",
#         "Serbia",
#         "Finland",
#         "Ireland",
#         "Lithuania",
#         "Switzerland",
#     ],
#     "Ireland": [
#         "Austria",
#         "Serbia",
#         "Finland",
#         "Lithuania",
#         "Portugal",
#         "Switzerland",
#     ],
#     "Lithuania": [
#         "Austria",
#         "Serbia",
#         "Finland",
#         "Ireland",
#         "Portugal",
#         "Switzerland",
#     ],
#     "Serbia": [
#         "Austria",
#         "Finland",
#         "Ireland",
#         "Lithuania",
#         "Portugal",
#         "Switzerland",
#     ],
#     "Austria": [
#         "Serbia",
#         "Finland",
#         "Ireland",
#         "Lithuania",
#         "Portugal",
#         "Switzerland",
#     ],
#     "Switzerland": [
#         "Austria",
#         "Serbia",
#         "Finland",
#         "Ireland",
#         "Lithuania",
#         "Portugal",
#     ],
# }

# Example
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


countries = {
    "UnitedNations": [
        "Austria",
        "Serbia",
        "Finland",
        "Ireland",
        "Lithuania",
        "Portugal",
        "Switzerland",
    ],
    "Finland": [
        "Austria",
        "Serbia",
        "Ireland",
        "Lithuania",
        "Portugal",
        "Switzerland",
    ],
}

backbones = ["vit", "resnet50"]
fed_methods = ["fedavg", "fedbn"]

yaml_experiment = "runs:"
for fed_method in fed_methods:
    for eval_country, clients in countries.items():
        assert clients != [], f"{eval_country} has no clients"

        for backbone in backbones:
            if backbone == "vit":
                batch_size = 370
            if backbone == "resnet50":
                batch_size = 450
            yaml_experiment += f"""

  -
    run_mode: fed
    batch_size: {batch_size}
    workers: 4
    device: cuda
    country: {eval_country} # in fed mode this is the country evaluate against.
    backbone: {backbone}
    fed_method: {fed_method}
    fed_epochs: 20
    fed_clients: {clients}
    fed_clients_source_dir: "/home/ubuntu/BigEarthNet/experiments/clients/"
    fed_servers_output_dir: "/home/ubuntu/BigEarthNet/experiments/servers/"
    data_root: "/home/ubuntu/BigEarthNet/datasets/"
    timeit: true
    cache: true
"""

print(yaml_experiment)
