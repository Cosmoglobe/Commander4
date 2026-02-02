import yaml
import os
from argparse import ArgumentParser
from commander4.utils.params import Params

parser = ArgumentParser()
parser.add_argument("-p",
                    "--parameter_file",
                    required=True,
                    help="Path to YAML-formatted parameter file. A default can be found in 'params/param_default.yml'.")

commandline_params = parser.parse_args()

if not os.path.isfile(commandline_params.parameter_file):
    raise FileExistsError(f"Could not find parameter file {commandline_params.parameter_file}")

with open(commandline_params.parameter_file, "r") as f:
    params_dict = yaml.safe_load(f)

params = Params(params_dict)