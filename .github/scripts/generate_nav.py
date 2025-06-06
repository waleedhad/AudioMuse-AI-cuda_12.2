import os
import re
import yaml

def generate_mkdocs_nav(web_dir, toc_file, template_file, output_file):
    nav_entries = []

    # Start with Home: toc.md as the first navigation item
    nav_entries.append({"Home": toc_file})

    # Read toc.md to get other navigation entries created by mdsplit -l 1
    toc_path = os.path.join(web_dir, toc_file)
    if os.path.exists(toc_path):
        with open(toc_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Regex to parse lines like '- [Link Name](file-name.md)' from toc.md
                match = re.match(r'^\s*-\s*\[(.*?)\]\((.*?)\)\s*$', line)
                if match:
                    link_name = match.group(1).strip()
                    file_path = match.group(2).strip()
                    # Add only if it's not the toc_file itself (to avoid duplicates)
                    if file_path != toc_file: 
                        nav_entries.append({link_name: file_path})
    else:
        print(f"Warning: {toc_path} not found. Navigation might be incomplete.")
        # If toc.md is not found, nav_entries will just contain {"Home": "toc.md"}
        # or be empty if that line was skipped/removed.

    # 1. Load the mkdocs.yml.template content into a Python dictionary.
    #    yaml.safe_load is used for security against arbitrary code execution.
    with open(template_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. Add the dynamically generated 'nav' list to the config dictionary.
    config['nav'] = nav_entries

    # 3. Dump the complete modified dictionary back to mkdocs.yml.
    #    yaml.safe_dump ensures proper YAML formatting and indentation automatically.
    #    default_flow_style=False ensures block style for readability (multi-line, indented lists).
    #    indent=2 ensures 2-space indentation.
    #    sort_keys=False preserves the order of items in 'nav_entries'.
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False, indent=2, sort_keys=False, allow_unicode=True)

if __name__ == "__main__":
    # Define paths relative to the repository root where the workflow will execute this script.
    generate_mkdocs_nav(
        web_dir="web",                   # mdsplit output goes here
        toc_file="toc.md",               # mdsplit creates toc.md inside web_dir
        template_file="mkdocs.yml.template", # This is in the repo root
        output_file="mkdocs.yml"         # This is the final output file in repo root
    )
