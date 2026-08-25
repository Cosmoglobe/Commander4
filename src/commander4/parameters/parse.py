"""Loading of Commander4 YAML parameter files."""

import os

import yaml
import yaml_include
from pixell.bunch import Bunch

from commander4.parameters.bunch import as_bunch_recursive


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

def params_from_dict(params_dict: dict) -> Bunch:
    """Build the parameter ``Bunch`` used by Commander4 from an already-resolved dictionary."""
    params = as_bunch_recursive(params_dict)
    params.parameter_file_as_string = yaml.dump(params_dict, sort_keys=False)
    return params


def load_params(parameter_file: str) -> tuple[Bunch, dict, str]:
    """Load one YAML parameter file and resolve its ``!inc`` directives.

    Args:
        parameter_file: Path to the main YAML file. Include paths are relative to this file.

    Returns:
        The parameter ``Bunch``, resolved dictionary, and resolved YAML text.
    """
    if not os.path.isfile(parameter_file):
        raise FileNotFoundError(f"Could not find parameter file {parameter_file}")

    parameter_file = os.path.abspath(parameter_file)
    parameter_dir = os.path.dirname(parameter_file)

    # Keep the include base directory local to this load. Registering the constructor globally
    # lets one parameter file's directory leak into the next file loaded by an offline tool.
    class ParameterLoader(yaml.FullLoader):
        pass

    ParameterLoader.add_constructor(
        "!inc", yaml_include.Constructor(base_dir=parameter_dir),
    )
    with open(parameter_file, encoding="utf-8") as handle:
        params_dict = yaml.load(handle, Loader=ParameterLoader)

    if not isinstance(params_dict, dict):
        raise ValueError(f"Parameter file {parameter_file} must contain a YAML mapping at its root")

    resolved_yaml = yaml.dump(params_dict, sort_keys=False)
    params = params_from_dict(params_dict)
    return params, params_dict, resolved_yaml
