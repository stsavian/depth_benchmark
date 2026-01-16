#!/usr/bin/env python3
"""
Convert command-line style arguments to launch.json format.

Usage:
    python args_to_launch_json.py

Then paste your command-line arguments and press Enter.
"""

import sys

def convert_args_to_launch_json(args_string):
    """
    Convert command-line style arguments to launch.json array format.

    Args:
        args_string: String like "--index 230 --sensor radar --viz-3d"

    Returns:
        String formatted for launch.json: '["--index", "230", "--sensor", "radar", "--viz-3d"]'
    """
    # Split by whitespace
    parts = args_string.strip().split()

    # Create JSON array format
    result = '["' + '", "'.join(parts) + '"]'

    return result

def main():
    print("=== Command-Line Args to launch.json Converter ===")
    print("Paste your command-line arguments below (e.g., --index 230 --sensor radar --viz-3d)")
    print("Press Enter when done:\n")

    # Read input
    args_input = input("> ").strip()

    if not args_input:
        print("No arguments provided!")
        return

    # Convert
    launch_json_format = convert_args_to_launch_json(args_input)

    # Output
    print("\nlaunch.json format:")
    print(launch_json_format)

    # Copy-friendly output
    print("\nCopy this line to your launch.json 'args' field:")
    print(f'    "args": {launch_json_format}')

if __name__ == "__main__":
    main()