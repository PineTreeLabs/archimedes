import glob
import os
import re
import shutil
from pathlib import Path


def copy_with_dark_suffix(source_dir, target_dir):
    # Convert to Path objects for easier manipulation
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Ensure target directory exists
    target_path.mkdir(exist_ok=True, parents=True)

    # Counter for files processed
    count = 0

    # Walk through the source directory
    for root, dirs, files in os.walk(source_path):
        # Get the relative path from the source directory
        rel_path = Path(root).relative_to(source_path)

        # Create the corresponding target directory
        dest_dir = target_path / rel_path
        dest_dir.mkdir(exist_ok=True, parents=True)

        # Process each PNG file
        for file in files:
            if file.lower().endswith(".png"):
                # Split the filename and extension
                name, ext = os.path.splitext(file)

                # Create the new filename with _dark suffix
                new_name = f"{name}_dark{ext}"

                # Copy the file
                source_file = Path(root) / file
                target_file = dest_dir / new_name

                shutil.copy2(source_file, target_file)
                count += 1
                print(f"Copied: {source_file} -> {target_file}")

    return count


def process_markdown_files():
    directory = "source/generated/notebooks"

    # Process all markdown files
    for md_file in glob.glob(f"{directory}/*.md"):
        with open(md_file, "r") as f:
            content = f.read()

        # Find all image references
        image_matches = re.findall(r"!\[(.*?)\]\((.*?)\.png\)", content)
        print(f"Processing {md_file}...")
        print(f"Found {len(image_matches)} images to process.")
        print("Image matches:", image_matches)

        # Replace with theme-switching versions
        modified_content = content
        for alt_text, base_path in image_matches:
            # Replace with both images, using class attributes
            # https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/light-dark.html#theme-dependent-images-and-content
            replacement = f"""
```{{image}} {base_path}.png
:class: only-light
```

```{{image}} {base_path}_dark.png
:class: only-dark
```"""
            original = f"![{alt_text}]({base_path}.png)"
            modified_content = modified_content.replace(original, replacement)

        # Write updated content
        with open(md_file, "w") as f:
            f.write(modified_content)


if __name__ == "__main__":
    # Usage
    source_directory = "source/generated/notebooks_dark"
    target_directory = "source/generated/notebooks"

    # Run the function
    files_copied = copy_with_dark_suffix(source_directory, target_directory)
    print(f"Completed. {files_copied} PNG files were copied with '_dark' suffix.")

    process_markdown_files()

    # Recursively copy source/notebooks/_static to source/generated/notebooks/_static
    static_source = Path("source/notebooks/_static")
    static_target = Path("source/generated/notebooks/_static")
    static_target.mkdir(exist_ok=True, parents=True)
    for item in static_source.glob("*"):
        target_item = static_target / item.name
        if item.is_dir():
            shutil.copytree(item, target_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target_item)
    print(f"Copied static files from {static_source} to {static_target}.")
