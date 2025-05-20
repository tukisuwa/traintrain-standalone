import argparse
import os
import json
import subprocess
import sys
import ast
import traceback
import datetime
import pathlib

def parse_override_value(value_str):
    try:
        try:
             return ast.literal_eval(value_str)
        except (ValueError, SyntaxError, TypeError):
             return value_str
    except Exception as e:
        print(f"Warning: Failed to parse override value '{value_str}': {e}")
        return value_str

def main():
    parser = argparse.ArgumentParser(
        description="Load JSON config, apply overrides, save to a specified/default directory, and run train_j.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("json_path", type=str, help="Path to the original JSON configuration file")
    parser.add_argument("--models-dir", type=str, default=None, help="Base directory for models (passed to train_j.py)")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Directory for Checkpoints (passed to train_j.py)")
    parser.add_argument("--vae-dir", type=str, default=None, help="Directory for VAE models (passed to train_j.py)")
    parser.add_argument("--lora-dir", type=str, default=None, help="Directory for LoRA models (passed to train_j.py)")

    parser.add_argument(
        "--override",
        nargs='+',
        metavar="KEY:VALUE",
        default=[],
        help='Override a JSON parameter. Format is "key:value". Value type is inferred.'
    )
    parser.add_argument(
        "--train-script-path",
        type=str,
        default="train_j.py",
        help="Path to the train_j.py script"
    )
    parser.add_argument(
        "--temp-config-dir",
        type=str,
        default="temp_configs",
        help="Directory to save the modified temporary JSON configuration files."
    )
    parser.add_argument(
        "--delete-temp-config",
        action="store_true",
        help="Delete the generated temporary config file after train_j.py finishes. (Default: keep the file)"
    )

    args = parser.parse_args()

    try:
        print(f"Loading original JSON config: {args.json_path}")
        original_json_path = pathlib.Path(args.json_path)
        if not original_json_path.is_file():
             print(f"Error: Original JSON file not found at '{args.json_path}'")
             return 1
        with open(original_json_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        if not isinstance(config_data, dict):
            print(f"Error: Expected JSON root to be a dictionary, but got {type(config_data)}.")
            return 1
        print("Original JSON loaded successfully.")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from '{args.json_path}': {e}")
        return 1
    except Exception as e:
        print(f"Error loading JSON file '{args.json_path}': {e}")
        traceback.print_exc()
        return 1

    overrides_applied = False
    if args.override:
        print("\nApplying overrides:")
        for item in args.override:
            parts = item.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value_str = parts[1].strip()
                if not key:
                    print(f"  Warning: Invalid override format (empty key): '{item}'. Skipping.")
                    continue

                parsed_value = parse_override_value(value_str)
                keys = key.split('.')
                current_level = config_data
                try:
                    for i, k in enumerate(keys[:-1]):
                        if isinstance(current_level, dict) and k in current_level:
                             if isinstance(current_level[k], dict):
                                 current_level = current_level[k]
                             else:
                                 print(f"  Warning: Intermediate key '{k}' in '{key}' exists but is not a dictionary. Cannot traverse further. Skipping override.")
                                 current_level = None
                                 break
                        else:
                             print(f"  Warning: Intermediate key '{k}' in '{key}' not found. Skipping override.")
                             current_level = None
                             break

                    if current_level is not None:
                        final_key = keys[-1]
                        if isinstance(current_level, dict):
                            if final_key in current_level:
                                original_value = current_level.get(final_key, '<Key did not exist>')
                                print(f"  Overriding key '{key}': '{original_value}' (original) -> '{parsed_value}' ({type(parsed_value).__name__})")
                                current_level[final_key] = parsed_value
                                overrides_applied = True
                            else:
                                print(f"  Warning: Key '{key}' (or final key '{final_key}') not found in the configuration. Skipping override.")
                        else:
                            print(f"  Warning: Cannot apply final key '{final_key}' because the target level is not a dictionary. Skipping override for '{key}'.")

                except Exception as e:
                    print(f"  Error applying override for key '{key}' with value string '{value_str}': {e}. Skipping.")
            else:
                print(f"  Warning: Invalid override format (missing ':'?): '{item}'. Skipping.")
        if not overrides_applied and args.override:
             print("  No valid overrides were applied.")
    else:
        print("\nNo overrides specified.")


    temp_config_file_path = None
    try:
        temp_dir = pathlib.Path(args.temp_config_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nEnsured temporary config directory exists: {temp_dir.resolve()}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = original_json_path.stem
        temp_filename = f"{base_name}_{timestamp}.json"
        temp_config_file_path = temp_dir / temp_filename

        with open(temp_config_file_path, 'w', encoding='utf-8') as tmp_f:
            json.dump(config_data, tmp_f, indent=2, ensure_ascii=False)
        print(f"Modified configuration saved to: {temp_config_file_path}")

        command = [
            sys.executable,
            args.train_script_path,
            str(temp_config_file_path)
        ]

        if args.models_dir: command.extend(["--models-dir", args.models_dir])
        if args.ckpt_dir:   command.extend(["--ckpt-dir", args.ckpt_dir])
        if args.vae_dir:    command.extend(["--vae-dir", args.vae_dir])
        if args.lora_dir:   command.extend(["--lora-dir", args.lora_dir])

        print("\nExecuting train_j.py with the following command:")
        print(" ".join(f'"{arg}"' if ' ' in arg else arg for arg in command))
        print("-" * 20)

        result = subprocess.run(command, check=True)

        print("-" * 20)
        print(f"train_j.py finished with exit code: {result.returncode}")
        return result.returncode

    except FileNotFoundError as e:
         print(f"Error: File not found - {e}. Please check script paths.")
         return 1
    except subprocess.CalledProcessError as e:
        print(f"Error: train_j.py execution failed with exit code {e.returncode}.")
        return e.returncode
    except Exception as e:
        print("\nAn error occurred during processing:")
        traceback.print_exc()
        return 1
    finally:
        if temp_config_file_path and temp_config_file_path.exists():
            if args.delete_temp_config:
                try:
                    temp_config_file_path.unlink()
                    print(f"\nTemporary config file deleted: {temp_config_file_path}")
                except OSError as e:
                    print(f"Warning: Failed to delete temporary config file '{temp_config_file_path}': {e}")
            else:
                print(f"\nTemporary config file kept at: {temp_config_file_path}")
        elif args.delete_temp_config and not temp_config_file_path:
             print("\nNo temporary config file was generated to delete.")


if __name__ == "__main__":
    sys.exit(main())