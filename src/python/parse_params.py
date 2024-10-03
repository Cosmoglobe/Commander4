import yaml
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-p",
                    "--parameter_file",
                    required=True,
                    help="Path to YAML-formatted parameter file. A default can be found in 'params/param_default.yml'.")

params = parser.parse_args()

if not os.path.isfile(params.parameter_file):
    raise FileExistsError(f"Could not find parameter file {parser.parameter_file}")

with open(params.parameter_file, "r") as f:
    params = yaml.safe_load(f)