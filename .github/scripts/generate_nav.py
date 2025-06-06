import os
import re
import yaml

def get_display_name_from_path(relative_path):
    """
    Converts a file/folder path (e.g., '2---Deployment-and-Parameters/2.1---Quick-Start-on-K3S.md')
    into a readable display name for the menu.
    """
    basename = os.path.basename(relative_path) # Get '2.1---Quick-Start-on-K3S.md' or '2---Deployment-and-Parameters'
    filename_without_ext = os.path.splitext(basename)[0] # Remove '.md'

    # Hardcoded overrides for top-level section names to match README exactly
    # These map the slugs to the desired display names in the menu
    if filename_without_ext == "1---AudioMuse-AI": return "1 - AudioMuse-AI"
    if filename_without_ext == "2---Deployment-and-Parameters": return "2 - Deployment and Parameters"
    if filename_without_ext == "3---Algorithm-details": return "3 - Algorithm details"
    if filename_without_ext == "4---Screenshots": return "4 - Screenshots"
    if filename_without_ext == "5---Key-Technologies": return "5 - Key Technologies"
    if filename_without_ext == "6---Additional-Documentation": return "6 - Additional Documentation"
    if filename_without_ext == "7---Future-Possibilities": return "7 - Future Possibilities"

    # For sub-items, try to extract number and title (e.g., "2.1---Quick-Start-on-K3S")
    match = re.match(r'^((\d+\.)*\d+)---(.*)$', filename_without_ext)
    if match:
        slug_part = match.group(3) # e.g., "Quick-Start-on-K3S"
        display_name = slug_part.replace('-', ' ').replace('_', ' ').title()
        return display_name
    
    # Fallback for any other format (e.g., if a file doesn't follow the 'NUMBER---TITLE' pattern)
    return filename_without_ext.replace('-', ' ').replace('_', ' ').title()


def generate_mkdocs_config(web_dir, template_file, output_file):
    """
    Generates the mkdocs.yml file with a dynamic navigation structure.
    """
    # Load the mkdocs.yml.template content into a Python dictionary.
    with open(template_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Build the navigation structure based on the 'web' directory content
    nav_structure = []
    nav_structure.append({"Home": "toc.md"}) # Always include toc.md as Home

    # Dictionary to hold categories by their full slugged folder name (e.g., '2---Deployment-and-Parameters')
    # This will help in building the nested structure.
    categories = {}

    # Collect all top-level files and directories in 'web_dir'
    top_level_entries = sorted(os.listdir(web_dir))
    
    # First pass: Identify categories (directories) and top-level pages
    for entry in top_level_entries:
        full_path = os.path.join(web_dir, entry)
        
        if os.path.isdir(full_path):
            # This is a category folder (e.g., '2---Deployment-and-Parameters')
            category_display_name = get_display_name_from_path(entry)
            categories[entry] = {"display_name": category_display_name, "items": []}
        elif entry.endswith(".md") and entry != "toc.md":
            # This is a top-level Markdown file (e.g., '1---AudioMuse-AI.md', '5---Key-Technologies.md')
            file_display_name = get_display_name_from_path(entry)
            nav_structure.append({file_display_name: entry})

    # Second pass: Populate category items by walking through identified directories
    for category_folder_name in sorted(categories.keys()): # Iterate through category folders in order
        category_full_path = os.path.join(web_dir, category_folder_name)
        
        # Get sorted list of files within the category folder
        category_files = sorted(os.listdir(category_full_path))
        
        for file_name in category_files:
            if file_name.endswith(".md"):
                file_relative_path = os.path.join(category_folder_name, file_name) # e.g., '2---/2.1---.md'
                file_display_name = get_display_name_from_path(file_relative_path)
                categories[category_folder_name]["items"].append({file_display_name: file_relative_path})
    
    # Now, add the populated categories to the main nav_structure in sorted order
    # It's important to sort categories based on their numerical prefix (e.g., 1, 2, 3, 4, 5, 6, 7)
    # This ensures categories appear correctly ordered alongside top-level pages.
    
    # We need a consolidated list of all items (top-level files and category definitions)
    # and then sort them by their numeric prefix.
    
    final_sorted_nav_items = []

    # Add top-level files (1, 5, 6) first as they are simple pages
    for display_name, source_name in [
        ("1 - AudioMuse-AI", "1---AudioMuse-AI.md"),
        ("5 - Key Technologies", "5---Key-Technologies.md"),
        ("6 - Additional Documentation", "6---Additional-Documentation.md"),
    ]:
        if os.path.exists(os.path.join(web_dir, source_name)): # Check if file actually exists
            final_sorted_nav_items.append({display_name: source_name})

    # Add categories (2, 3, 4, 7) with their sub-items
    for category_slug in sorted(categories.keys()): # Categories are already identified and sorted by key (slug)
        category_data = categories[category_slug]
        if category_data["items"]: # Only add category if it has items
            final_sorted_nav_items.append({category_data["display_name"]: category_data["items"]})

    # Insert the generated top-level navigation structure after the 'Home' link
    # This approach assumes 'Home' is always the first item.
    nav_structure = [nav_structure[0]] + final_sorted_nav_items


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
