import yaml
import os
from argparse import ArgumentParser
from pixell.bunch import Bunch
import yaml_include


# TODO: Below is code for finding either the Commander4 PIP version number, or the git hash in case
# of an editable install. I don't want to introduce this code yet because I'm unsure about having
# tons of MPI tasks thrashing the file system just to get a git hash. Ideally this should be done
# only by the master rank.

# from importlib.metadata import version, PackageNotFoundError
# def get_version_info(package_name, script_location):
#     """
#     Retrieves a unique identifier for the current version of the package.
    
#     Priority:
#     1. Current Git Commit Hash (if running from a git repo/editable install).
#     2. Installed Package Version (if standard pip install).
#     3. "unknown"
#     """
    
#     # Helper function (without git dependency) which manually parses the `.git` folder to find the
#     # current git hash of Commander4.
#     def _get_git_hash(start_path):
#         # 1. Find the .git directory
#         root_path = os.path.abspath(start_path)
#         git_dir = None

#         while True:
#             possible_git = os.path.join(root_path, ".git")
#             if os.path.isdir(possible_git):
#                 git_dir = possible_git
#                 break
#             parent = os.path.dirname(root_path)
#             if parent == root_path: 
#                 return None
#             root_path = parent

#         # 2. Read HEAD
#         head_path = os.path.join(git_dir, "HEAD")
#         if not os.path.exists(head_path):
#             return None
            
#         with open(head_path, "r") as f:
#             head_content = f.read().strip()

#         # 3. Handle Detached HEAD (It's already a hash)
#         if not head_content.startswith("ref:"):
#             return head_content

#         # 4. Handle Branch Ref (Follow the path)
#         target_ref = head_content.split(" ", 1)[1] # e.g. "refs/heads/main"
        
#         # Check loose file
#         loose_ref_path = os.path.join(git_dir, target_ref)
#         if os.path.exists(loose_ref_path):
#             with open(loose_ref_path, "r") as f:
#                 return f.read().strip()

#         # Check packed-refs
#         packed_refs_path = os.path.join(git_dir, "packed-refs")
#         if os.path.exists(packed_refs_path):
#             with open(packed_refs_path, "r") as f:
#                 for line in f:
#                     if line.startswith("#") or not line.strip(): continue
#                     parts = line.split()
#                     if len(parts) >= 2 and parts[1] == target_ref:
#                         return parts[0]
                        
#         return None

#     # 1. Try to get the Git Hash (Editable / Dev Install)
#     git_hash = _get_git_hash(script_location)
#     if git_hash:
#         return f"git-{git_hash}"

#     # 2. Fallback to Pip Version (Standard Install)
#     try:
#         # Note: 'package_name' must match the name in pyproject.toml
#         return f"v{version(package_name)}"
#     except PackageNotFoundError:
#         pass

#     return "unknown"


def as_bunch_recursive(dict_of_dicts, name=None):
    res = Bunch()
    
    # 1. Inject the name into the instance, bypassing Bunch's data _dict
    if name is not None:
         object.__setattr__(res, "_name", name)
         
    # 2. Recursively populate the bunch
    for key, val in dict_of_dicts.items():
        if isinstance(val, dict):
            # Pass the key down as the name for the child Bunch
            res[key] = as_bunch_recursive(val, name=key)
        else:
            res[key] = val

    return res

# ------------------------------------------------------------------------
# Parse parameter file
# ------------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument("-p",
                    "--parameter_file",
                    required=True,
                    help="Path to YAML-formatted parameter file. See the 'params/' directory.")

commandline_params = parser.parse_args()

if not os.path.isfile(commandline_params.parameter_file):
    raise FileExistsError(f"Could not find parameter file {commandline_params.parameter_file}")

# Register the !inc constructor so that YAML files can include other YAML files via
# e.g.  detectors: !inc detectors/LiteBIRD40GHz_L1.yml
# Paths in !inc directives are resolved relative to the parameter file's directory.
param_file_dir = os.path.dirname(os.path.abspath(commandline_params.parameter_file))
yaml.add_constructor(
    "!inc",
    yaml_include.Constructor(base_dir=param_file_dir),
)

with open(commandline_params.parameter_file, "r") as f:
    binary_yaml_data = f.read()
params_dict = yaml.full_load(binary_yaml_data)

params = as_bunch_recursive(params_dict)

# For reproducability, create custom entries in the parameter object which holds the entire
# parameter file as a fully resolved YAML string (with any !inc directives expanded).
params.parameter_file_as_string = yaml.dump(params_dict)

# Storing Commander4 version number or git commit.
# params.metadata.version_number = # print(get_version_info("commander4", __file__))