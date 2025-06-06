import os
import re
import yaml

def generate_mkdocs_nav(web_dir, toc_file, template_file, output_file):
    nav_entries = []

    # Start with Home: toc.md
    # We add this explicitly as it's a standard home page
    nav_entries.append({"Home": toc_file})

    # Read toc.md to get other navigation entries
    # mdsplit -l 1 creates a flat list of L1 headings in toc.md
    toc_path = os.path.join(web_dir, toc_file)
    if os.path.exists(toc_path):
        with open(toc_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Regex to parse lines like '- [Link Name](file-name.md)'
                match = re.match(r'^\s*-\s*\[(.*?)\]\((.*?)\)\s*$', line)
                if match:
                    link_name = match.group(1).strip()
                    file_path = match.group(2).strip()
                    # Avoid adding toc.md twice if it's in the generated list
                    if file_path != toc_file: 
                        nav_entries.append({link_name: file_path})
    else:
        print(f"Warning: {toc_path} not found. Navigation might be incomplete.")
        # If toc.md is not found, we might add a fallback, or just generate with "Home"

    # Load the mkdocs.yml template content
    with open(template_file, 'r', encoding='utf-8') as f:
        template_content = f.read()

    # Generate the YAML for the 'nav' section
    # yaml.dump handles correct indentation for us
    # default_flow_style=False makes it output in block style (nicely indented)
    # indent=2 sets indentation to 2 spaces (standard for MkDocs nav)
    # sort_keys=False maintains the order we appended them
    nav_yaml = yaml.dump({'nav': nav_entries}, default_flow_style=False, indent=2, sort_keys=False, allow_unicode=True)
    
    # The yaml.dump output starts with 'nav:', which we don't need if we're replacing a placeholder.
    # We just need the content *under* 'nav:'.
    # So, we remove the first 'nav:' line and any leading/trailing whitespace.
    nav_yaml_content = nav_yaml.replace('nav:', '', 1).strip()

    # Insert the generated navigation content into the template
    # The template has 'NAV_PLACEHOLDER - DO NOT REMOVE OR MODIFY THIS LINE'
    final_content = template_content.replace("# NAV_PLACEHOLDER - DO NOT REMOVE OR MODIFY THIS LINE", nav_yaml_content)

    # Write the complete mkdocs.yml file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_content)

if __name__ == "__main__":
    # Define paths relative to the repository root where the workflow will run
    # This script assumes it's being run from the repository root.
    # If the script is in .github/scripts/, you might need to adjust paths.
    
    # For a script in .github/scripts/, and files in repo root:
    # web_directory = "web"
    # toc_filename = "toc.md"
    # mkdocs_template = "mkdocs.yml.template"
    # mkdocs_output = "mkdocs.yml"
    
    # The workflow will execute this script from the repository root, so paths are direct:
    generate_mkdocs_nav(
        web_dir="web",                   # mdsplit output goes here
        toc_file="toc.md",               # mdsplit creates toc.md inside web_dir
        template_file="mkdocs.yml.template", # This is in the repo root
        output_file="mkdocs.yml"         # This is the final output file in repo root
    )
