import subprocess
import uuid
import sys
import yaml
import argparse
from pathlib import Path
import time

# Detect if we are running on a Darwin or Linux and set default path of the config file
if sys.platform == "darwin":
    yaml_path = Path("experiment.darwin.yaml")
elif sys.platform == "linux":
    yaml_path = Path("experiment.linux.yaml")
else:
    print("Unsupported platform")
    sys.exit(1)

parser = argparse.ArgumentParser(description="Experiment runner")
parser.add_argument(
    "--yaml",
    type=str,
    help="A yaml file containing the experiment runs configuration",
    default=str(yaml_path),
)
args = parser.parse_args()

config_path = Path(args.yaml)
print(f"Running experiment with config file: {config_path}")
assert config_path.exists(), f"Config file not found at {config_path}"

# load the config file
print(f"Loading config file at {config_path}")
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

print(f"Config file loaded: {config}")


# Randomly generate a unique ID for the experiment (tag for multiple runs)
experiment_id = str(uuid.uuid4())[:8]

# Find the path of python executable
python_bin = f"{sys.executable}"

print(f"Experiment ID: {experiment_id}")
print(f"Python executable: {python_bin}")

countries = [
    "UnitedNations",
    "Finland",
    "Portugal",
    "Ireland",
    "Lithuania",
    "Serbia",
    "Austria",
    "Switzerland",
    "TinyFinland",
]

runs = config["runs"]

for run in runs:
    assert run["country"] in countries, f"Unknown country: {run['country']}"

# Execute the experiment for each country
for run in runs:
    command = [
        [python_bin],
        ["main.py"],
        ["--experiment_id"],
        [experiment_id],
    ]
    for k, v in run.items():
        # if value is a list, we need to join it with spaces
        if isinstance(v, list):
            v = " ".join(v)
        command.append([f"--{k}", f"{v}"])
    # flatten the command list
    command = " ".join(item for sublist in command for item in sublist)
    print(f"Executing: {command}")
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    # sleep for a bit to make sure the process is done
    print(f"Sleeping for 10 seconds...")
    time.sleep(10)
