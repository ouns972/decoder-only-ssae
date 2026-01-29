from typing import Any, Dict, List
from PIL import Image

def flatten_dict_of_list(dict: Dict[Any, List[Any]]):
    output = []
    for k, v in dict.items():
        output = output + v
    return output


def is_in(list_a: List[str], list_b: List[str]) -> bool:
    # checks if list_a is included in list_b
    for a in list_a:
        if a.strip() not in list_b:
            return False
    return True

def display_table(data_dict):
    # Find the maximum number of rows
    max_rows = max(len(v) for v in data_dict.values())
    
    # Get the column names
    columns = list(data_dict.keys())
    
    # Determine column widths based on headers and longest data
    col_widths = []
    for col in columns:
        max_len = max([len(str(val)) for val in data_dict[col]] + [len(col)])
        col_widths.append(max_len)
    
    # Create the header
    header = "| " + " | ".join([col.ljust(col_widths[i]) for i, col in enumerate(columns)]) + " |"
    separator = "+-" + "-+-".join(['-' * col_widths[i] for i in range(len(columns))]) + "-+"
    
    # Print header
    print(separator)
    print(header)
    print(separator)
    
    # Print rows
    for i in range(max_rows):
        row = []
        for col_idx, col in enumerate(columns):
            col_values = data_dict[col]
            value = str(col_values[i]) if i < len(col_values) else ""
            row.append(value.ljust(col_widths[col_idx]))
        print("| " + " | ".join(row) + " |")
    print(separator)

def concatenate_2x2(image_paths, output_path):
    images = [Image.open(path) for path in image_paths]
    
    widths, heights = zip(*(img.size for img in images))
    
    target_width = min(widths)
    target_height = min(heights)
    images = [img.resize((target_width, target_height)) for img in images]
    
    new_width = target_width * 2
    new_height = target_height * 2
    new_image = Image.new('RGB', (new_width, new_height))
    
    new_image.paste(images[0], (0, 0))  # Top-left
    new_image.paste(images[1], (target_width, 0))  # Top-right
    new_image.paste(images[2], (0, target_height))  # Bottom-left
    new_image.paste(images[3], (target_width, target_height))  # Bottom-right
    
    new_image.save(output_path)