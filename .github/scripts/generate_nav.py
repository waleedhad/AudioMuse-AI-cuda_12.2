import os
import re
import yaml

def get_display_name_from_path(relative_path, is_folder=False):
    """
    Converts a file/folder path (e.g., '2---Deployment-and-Parameters/2.1---Quick-Start-on-K3S.md')
    into a readable display name for the menu.
    It intelligently handles numerical prefixes and slug-to-title conversion.
    """
    basename = os.path.basename(relative_path)
    filename_without_ext = os.path.splitext(basename)[0] # Remove '.md' if present

    # Regex to extract the number prefix and the slug part
    match = re.match(r'^((\d+\.)*\d+)---(.*)$', filename_without_ext)
    
    if match:
        number_prefix = match.group(1) # e.g., "1", "2.1", "3"
        slug_part = match.group(3)     # e.g., "Audio-Muse-AI", "Quick-Start-on-K3S"
        
        display_name_from_slug = slug_part.replace('-', ' ').replace('_', ' ').title()
        
        # If it's a top-level item (e.g., '1', '2', '3', '4', '5', '6', '7')
        # We append the number prefix to the display name for clarity in the menu.
        if '.' not in number_prefix: # Check if it's a single number (e.g., "1" not "2.1")
            return f"{number_prefix} - {display_name_from_slug}"
        else:
            # For sub-levels (e.g., "2.1", "3.2"), remove the number prefix as it's implied by parent
            return display_name_from_slug
    
    # Fallback for any unexpected format
    return filename_without_ext.replace('-', ' ').replace('_', ' ').title()


def generate_mkdocs_config(web_dir, template_file, output_file):
    """
    Generates the mkdocs.yml file with a dynamic navigation structure
    by walking the 'web' directory created by mdsplit -l 2.
    """
    # Load the mkdocs.yml.template content into a Python dictionary.
    with open(template_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Initialize the navigation structure
    nav_structure = []
    nav_structure.append({"Home": "toc.md"}) # Always include toc.md as Home

    # --- CRITICAL FIX: Ensure 'top_level_items' is defined here ---
    # This list will collect all the top-level pages and categories before sorting them.
    top_level_items = [] 
    # -----------------------------------------------------------------

    # Walk the 'web_dir' to discover structure
    for entry in sorted(os.listdir(web_dir)): # Sorted for consistent order in menu
        full_path = os.path.join(web_dir, entry)
        
        # Exclude toc.md as it's handled separately
        if entry == "toc.md":
            continue

        # Extract order key from the file/folder name (e.g., '1' from '1---AudioMuse-AI')
        order_match = re.match(r'^((\d+).*?)---', entry)
        order_key = int(order_match.group(2)) if order_match else 9999 # Default to high number for non-standard names

        if os.path.isdir(full_path):
            # This is a category folder (e.g., '2---Deployment-and-Parameters')
            category_display_name = get_display_name_from_path(entry, is_folder=True) # Use is_folder to apply folder naming rules
            category_items = []
            
            # Populate category items by walking through files inside this directory
            for item_name in sorted(os.listdir(full_path)): # Sort children for consistent order
                item_full_path = os.path.join(full_path, item_name)
                if os.path.isfile(item_full_path) and item_name.endswith(".md"):
                    item_relative_path = os.path.join(entry, item_name) # Path relative to web_dir
                    item_display_name = get_display_name_from_path(item_relative_path)
                    category_items.append({item_display_name: item_relative_path})
            
            # Add category to top_level_items if it has children
            if category_items:
                top_level_items.append({
                    "order_key": order_key, 
                    "type": "category", 
                    "display_name": category_display_name, 
                    "items": category_items
                })
            else:
                print(f"Warning: Directory '{entry}' is empty and not added to navigation.")

        elif os.path.isfile(full_path) and entry.endswith(".md"):
            # This is a top-level Markdown file (e.g., '1---AudioMuse-AI.md', '5---Key-Technologies.md')
            file_display_name = get_display_name_from_path(entry)
            top_level_items.append({
                "order_key": order_key, 
                "type": "page", 
                "display_name": file_display_name, 
                "path": entry
            })

    # Sort all collected top-level items by their numerical order_key
    top_level_items.sort(key=lambda x: x["order_key"])

    # Assemble the final navigation structure by adding sorted top-level items after 'Home'
    for item in top_level_items:
        if item["type"] == "page":
            nav_structure.append({item["display_name"]: item["path"]})
        elif item["type"] == "category":
            nav_structure.append({item["display_name"]: item["items"]})

    # Add the generated nav list to the config dictionary
    config['nav'] = nav_structure

    # Dump the complete modified dictionary back to mkdocs.yml.
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False, indent=2, sort_keys=False, allow_unicode=True)

if __name__ == "__main__":
    generate_mkdocs_config(
        web_dir="web",
        template_file="mkdocs.yml.template",
        output_file="mkdocs.yml"
    )
