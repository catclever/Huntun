import argparse
import dataclasses
import json
import os
from typing import Type, TypeVar, Any

T = TypeVar('T')

def build_parser_from_config(config_cls: Type[T], parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    """
    Dynamically adds arguments to an argparse.ArgumentParser based on a dataclass fields.
    """
    if parser is None:
        parser = argparse.ArgumentParser()
        
    for field in dataclasses.fields(config_cls):
        # We don't want to override existing args (e.g. if common args are already added)
        arg_name = f"--{field.name}"
        
        # Skip if the argument is already defined
        has_arg = any(opt == arg_name for action in parser._actions for opt in action.option_strings)
        if has_arg:
            continue
            
        field_type = field.type
        # Handle optional types or unions simply by falling back to str if complex
        # For our simple configs, it's usually int, float, bool, or str.
        
        if field_type == bool:
            parser.add_argument(arg_name, action=argparse.BooleanOptionalAction, default=field.default)
        else:
            parser.add_argument(arg_name, type=field_type, default=field.default)
            
    return parser

def extract_config_from_args(args: argparse.Namespace, config_cls: Type[T]) -> T:
    """
    Extracts relevant arguments from the parsed args namespace to instantiate the config dataclass.
    """
    kwargs = {}
    for field in dataclasses.fields(config_cls):
        if hasattr(args, field.name):
            kwargs[field.name] = getattr(args, field.name)
    return config_cls(**kwargs)

def save_config(config: Any, save_dir: str, filename: str = "config.json"):
    """
    Saves a dataclass config to a JSON file.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(config), f, indent=4, ensure_ascii=False)

def load_config(save_dir: str, config_cls: Type[T], filename: str = "config.json") -> T:
    """
    Loads a dataclass config from a JSON file.
    """
    file_path = os.path.join(save_dir, filename)
    if not os.path.exists(file_path):
        return None
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Only pass arguments that exist in the dataclass
    valid_keys = {f.name for f in dataclasses.fields(config_cls)}
    filtered_data = {k: v for k, v in data.items() if k in valid_keys}
    
    return config_cls(**filtered_data)
