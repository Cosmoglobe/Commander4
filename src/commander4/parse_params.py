import yaml
import os
from argparse import ArgumentParser
from commander4.utils.params import Params


# TODO: Below is code for finding either the Commander4 PIP version number, or the git hash in case
# of an editable install. I don't want to introduce this code yet because I'm unsure about having
# the git package as a dependency just for this. Can alternatively be avoided by manually probing
# the .git folder to get the has.

# import sys
# import git
# from importlib.metadata import version, PackageNotFoundError
# def get_version_identifier():
#     """
#     Retrieves a unique identifier for the current version of the package.
#     Priority:
#     1. Current Git Commit Hash (if running from a git repo/editable install).
#     2. Installed Package Version (if standard pip install).
#     """
    
#     # 1. Identify the absolute path of the package source code
#     # __file__ points to the __init__.py of the package
#     try:
#         # Replace 'my_package' with your actual package name imported above
#         package_path = os.path.dirname(sys.modules['my_package'].__file__)
#     except (KeyError, AttributeError):
#         # Fallback if the module isn't loaded or is a namespace package
#         return "unknown"

#     # 2. Attempt to retrieve Git Hash (Editable / Dev Install)
#     try:
#         # Search parent directories for .git because source might be in src/my_package
#         repo = git.Repo(package_path, search_parent_directories=True)
        
#         sha = repo.head.object.hexsha
        
#         # Critical: Check if the code has uncommitted changes
#         if repo.is_dirty():
#             sha += "-dirty"
            
#         return f"git-{sha}"
        
#     except (git.InvalidGitRepositoryError, git.NoSuchPathError):
#         # This happens when installed in site-packages with no .git folder
#         pass

#     # 3. Fallback to Pip Version (Standard Install)
#     try:
#         return f"v{version('my_package')}"
#     except PackageNotFoundError:
#         return "unknown-version"




parser = ArgumentParser()
parser.add_argument("-p",
                    "--parameter_file",
                    required=True,
                    help="Path to YAML-formatted parameter file. A default can be found in 'params/param_default.yml'.")

commandline_params = parser.parse_args()

if not os.path.isfile(commandline_params.parameter_file):
    raise FileExistsError(f"Could not find parameter file {commandline_params.parameter_file}")

with open(commandline_params.parameter_file, "r") as f:
    binary_yaml_data = f.read()
params_dict = yaml.safe_load(binary_yaml_data)

params = Params(params_dict)

# For reproducability, create custom entries in the parameter object which holds the entire
# parameter file, both as a single string, and as a binary YAML file.
params.parameter_file_as_string = yaml.dump(params_dict)
params.parameter_file_binary_yaml = binary_yaml_data

# Storing Commander4 version number or git commit.
# params.metadata.version_number = get_version_identifier