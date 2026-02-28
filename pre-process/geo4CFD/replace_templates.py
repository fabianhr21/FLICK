#!/usr/bin/env python3
"""
replace_templates.py

Copies p2 and p3 template directories from a shared templates folder into
a new MN5 directory under the case folder, then replaces all template
placeholders in .sh, .geo, and .json files.

Placeholders replaced:
  {{BASENAME}}      -> BASENAME argument (typically the case subdirectory name)
  {{LZ}}            -> z_length from domain_dimensions.txt
  {{y_length}}      -> y_length from domain_dimensions.txt
  {{prec_length}}   -> precursor_length from domain_dimensions.txt

Usage:
  python3 replace_templates.py <case_dir> <BASENAME> <templates_dir>

  <case_dir>      : The case subdirectory (e.g. /path/to/BARCELONA/275-76)
                    domain_dimensions.txt is expected at <case_dir>/output/domain_dimensions.txt
                    The MN5 directory will be created at <case_dir>/MN5/
  <BASENAME>      : The basename string to substitute for {{BASENAME}}
  <templates_dir> : Directory containing the p2 and p3 template folders

Example:
  python3 replace_templates.py /home/user/CASES/BARCELONA/275-76 275-76 /home/user/scripts/
"""

import os
import sys
import shutil
import argparse

# File extensions to process — .py files intentionally excluded
# to avoid touching library scripts like gmsh2sod2d.py or interpolate.py
TARGET_EXTENSIONS = {'.sh', '.geo', '.json', '.dat'}

# Template subdirectories to copy into MN5
TEMPLATE_DIRS = ['p2', 'p3']


def parse_domain_dimensions(filepath):
    """
    Parse a key=value formatted domain_dimensions.txt file.
    Returns a dict of all key/value pairs as strings.
    """
    dims = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            key, _, value = line.partition('=')
            dims[key.strip()] = value.strip()

    required = {'z_length', 'y_length', 'precursor_length'}
    missing = required - dims.keys()
    # Round every value to 2 decimal places for cleaner substitution
    for key in required:
        if key in dims:
            try:
                dims[key] = f"{float(dims[key]):.8f}"
                dims['z_length'] = f"{float(dims['z_length']):.3f}"
            except ValueError:
                raise ValueError(f"Invalid numeric value for {key} in domain_dimensions.txt: '{dims[key]}'")
    
    if missing:
        raise ValueError(f"Missing required keys in domain_dimensions.txt: {missing}")

    return dims


def build_replacements(basename, dims):
    """
    Build the placeholder -> value mapping from parsed dimensions and BASENAME.
    """
    return {
        '{{BASENAME}}':    basename,
        '{{LZ}}':          dims['z_length'],
        '{{y_length}}':    dims['y_length'],
        '{{prec_length}}': dims['precursor_length'],
    }


def process_file(filepath, replacements):
    """
    Read a file, apply all replacements, and write it back only if changed.
    Returns True if the file was modified.
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        original = f.read()

    modified = original
    for placeholder, value in replacements.items():
        modified = modified.replace(placeholder, value)

    if modified != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(modified)
        return True

    return False


def copy_templates_to_mn5(templates_dir, mn5_dir):
    """
    Copy p2 and p3 from templates_dir into mn5_dir.
    If mn5_dir already exists it is removed first to ensure a clean state.
    """
    if os.path.exists(mn5_dir):
        # Only remove directories inside MN5, not the entire MN5 if it already exists (to preserve any non-template files)
        for item in os.listdir(mn5_dir):
            item_path = os.path.join(mn5_dir, item)
            if os.path.isdir(item_path) and item in TEMPLATE_DIRS:
                print(f"  Removing existing template directory: {item_path}")
                shutil.rmtree(item_path)
        
        
        # print(f"  Removing existing MN5 directory: {mn5_dir}")
        # shutil.rmtree(mn5_dir)

    if not os.path.exists(mn5_dir):
        os.makedirs(mn5_dir)
        print(f"  Created MN5 directory: {mn5_dir}")

    for tdir in TEMPLATE_DIRS:
        src = os.path.join(templates_dir, tdir)
        dst = os.path.join(mn5_dir, tdir)
        if not os.path.isdir(src):
            raise FileNotFoundError(f"Template directory not found: {src}")
        shutil.copytree(src, dst)
        print(f"  Copied template: {src} -> {dst}")


def rename_geo_files(mn5_dir, basename):
    """
    Rename Buildings_p2.geo -> {BASENAME}_Buildings_p2.geo
    and     Buildings_p3.geo -> {BASENAME}_Buildings_p3.geo
    inside their respective subdirectories under mn5_dir.
    """
    # Map: subdirectory -> expected generic name -> new name
    targets = {
        'p2': ('Buildings_p2.geo', f'{basename}_Buildings_p2.geo'),
        'p3': ('Buildings_p3.geo', f'{basename}_Buildings_p3.geo'),
    }

    for subdir, (old_name, new_name) in targets.items():
        old_path = os.path.join(mn5_dir, subdir, old_name)
        new_path = os.path.join(mn5_dir, subdir, new_name)
        if os.path.isfile(old_path):
            os.rename(old_path, new_path)
            print(f"  [RENAMED]   {os.path.join(subdir, old_name)} -> {os.path.join(subdir, new_name)}")
        else:
            print(f"  [WARNING]   Expected geo file not found, skipping rename: {old_path}", file=sys.stderr)


def walk_and_replace(root_dir, replacements):
    """
    Recursively walk root_dir and apply replacements to all TARGET_EXTENSIONS files.
    """
    total_checked = 0
    total_modified = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        filenames.sort()

        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in TARGET_EXTENSIONS:
                continue

            filepath = os.path.join(dirpath, filename)
            total_checked += 1

            try:
                changed = process_file(filepath, replacements)
                rel_path = os.path.relpath(filepath, root_dir)
                if changed:
                    print(f"  [MODIFIED]  {rel_path}")
                    total_modified += 1
                else:
                    print(f"  [unchanged] {rel_path}")
            except Exception as e:
                rel_path = os.path.relpath(filepath, root_dir)
                print(f"  [ERROR]     {rel_path} -> {e}", file=sys.stderr)

    return total_checked, total_modified


def main():
    parser = argparse.ArgumentParser(
        description="Copy p2/p3 templates into a case MN5 directory and replace placeholders."
    )
    parser.add_argument(
        'case_dir',
        help="Case subdirectory (e.g. /path/to/BARCELONA/275-76). "
             "domain_dimensions.txt is read from <case_dir>/output/domain_dimensions.txt "
             "and MN5/ will be created at <case_dir>/MN5/"
    )
    parser.add_argument(
        'basename',
        help="The BASENAME value to substitute for {{BASENAME}}"
    )
    parser.add_argument(
        'templates_dir',
        help="Directory containing the p2 and p3 template folders"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ validate
    if not os.path.isdir(args.case_dir):
        print(f"Error: case_dir '{args.case_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(args.templates_dir):
        print(f"Error: templates_dir '{args.templates_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    domain_dims_path = os.path.join(args.case_dir, 'output', 'domain_dimensions.txt')
    if not os.path.isfile(domain_dims_path):
        print(f"Error: domain_dimensions.txt not found at '{domain_dims_path}'.", file=sys.stderr)
        sys.exit(1)

    mn5_dir = os.path.join(args.case_dir, 'output/MN5')

    # ------------------------------------------------------------------ summary
    print(f"\n--- replace_templates.py ---")
    print(f"Case dir:  {os.path.abspath(args.case_dir)}")
    print(f"BASENAME:  {args.basename}")
    print(f"Templates: {os.path.abspath(args.templates_dir)}")
    print(f"Dims file: {domain_dims_path}")
    print(f"Output:    {mn5_dir}\n")

    # ------------------------------------------------------------------ parse dims
    try:
        dims = parse_domain_dimensions(domain_dims_path)
    except ValueError as e:
        print(f"Error parsing domain_dimensions.txt: {e}", file=sys.stderr)
        sys.exit(1)

    replacements = build_replacements(args.basename, dims)

    print("Substitutions to apply:")
    for placeholder, value in replacements.items():
        print(f"  {placeholder:20s} -> {value}")

    # ------------------------------------------------------------------ copy templates
    print(f"\nCopying templates into MN5...")
    try:
        copy_templates_to_mn5(args.templates_dir, mn5_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------ rename .geo files
    print(f"\nRenaming .geo files...")
    rename_geo_files(mn5_dir, args.basename)

    # ------------------------------------------------------------------ replace
    print(f"\nApplying replacements in: {mn5_dir}")
    checked, modified = walk_and_replace(mn5_dir, replacements)

    print(f"\nDone. {modified}/{checked} file(s) modified.")
    print(f"MN5 directory ready at: {mn5_dir}\n")


if __name__ == '__main__':
    main()