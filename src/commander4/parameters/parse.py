"""Loading of Commander4 YAML parameter files."""

import os
import re

import yaml
from pixell.bunch import Bunch

from commander4.parameters.bunch import as_bunch_recursive

# A line containing nothing but `!import <path>` (plus an optional trailing comment) is replaced by
# the text of that file. Paths may be quoted.
INCLUDE_PATTERN = re.compile(r"^(\s*)!import\s+(.+?)(?:\s+#.*)?$")


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


def expand_includes(parameter_file: str, include_stack: tuple[str, ...] = ()) -> list[str]:
    """Read one YAML file and splice in the files named by its ``!import`` lines.

    The splice is textual: the included file's lines are indented to the column of the ``!import``
    that pulled them in. An ``!import`` written as the value of a key therefore becomes that key's
    value, and one written among the entries of a mapping adds its keys to that mapping. Include
    paths are relative to the directory of the file containing the ``!import`` line.

    Args:
        parameter_file: Absolute path of the file to read.
        include_stack: Files currently being expanded, used to detect circular includes.

    Returns:
        The lines of the file, with every ``!import`` line replaced by the expanded lines of the
        included file.
    """
    if parameter_file in include_stack:
        chain = " -> ".join(include_stack + (parameter_file,))
        raise ValueError(f"Circular !import chain in parameter files: {chain}")

    expanded_lines = []
    with open(parameter_file, encoding="utf-8") as handle:
        for line in handle.read().splitlines():
            match = INCLUDE_PATTERN.match(line)
            if match is None:
                if "!inc " in line and not line.lstrip().startswith("#"):
                    raise ValueError(f"In {parameter_file}: the include directive '!inc' has been "
                                     f"renamed to '!import', and must sit alone on its own line. "
                                     f"Offending line: {line}")
                if "!import" in line and not line.lstrip().startswith("#"):
                    raise ValueError(f"In {parameter_file}: an !import must sit alone on its own "
                                     f"line, written as '!import <path>'. Offending line: {line}")
                expanded_lines.append(line)
                continue
            indent, include_name = match.group(1), match.group(2).strip("\"'")
            include_file = os.path.join(os.path.dirname(parameter_file), include_name)
            if not os.path.isfile(include_file):
                raise FileNotFoundError(f"Could not find !import file {include_file}, imported "
                                        f"from {parameter_file}")
            for included_line in expand_includes(include_file, include_stack + (parameter_file,)):
                # Blank lines are left blank rather than filled with trailing whitespace.
                expanded_lines.append(indent + included_line if included_line.strip() else "")
    return expanded_lines


def load_params(parameter_file: str) -> tuple[Bunch, dict, str]:
    """Load one YAML parameter file and resolve its ``!import`` directives.

    Args:
        parameter_file: Path to the main YAML file. Include paths are relative to this file.

    Returns:
        The parameter ``Bunch``, resolved dictionary, and resolved YAML text.
    """
    if not os.path.isfile(parameter_file):
        raise FileNotFoundError(f"Could not find parameter file {parameter_file}")

    parameter_file = os.path.abspath(parameter_file)
    params_dict = yaml.safe_load("\n".join(expand_includes(parameter_file)))

    if not isinstance(params_dict, dict):
        raise ValueError(f"Parameter file {parameter_file} must contain a YAML mapping at its root")

    resolved_yaml = yaml.dump(params_dict, sort_keys=False)
    params = params_from_dict(params_dict)
    return params, params_dict, resolved_yaml
