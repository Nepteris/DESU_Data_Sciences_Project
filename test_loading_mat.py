import os
import ast


def parse_value(val):
    val = val.strip()
    if val == 'NaN' or val == '':
        return None
    # Try to parse as Python literal (list, number)
    try:
        # safely parse string representations of numbers or lists
        parsed = ast.literal_eval(val)
        return parsed
    except (ValueError, SyntaxError):
        # fallback: return string as is
        return val


def read_matlab_txt_file(filepath):
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # First line is header
    header = lines[0].strip().split(';')

    # Parse each data line
    for line in lines[1:]:
        fields = line.strip().split(';')
        entry = {}
        for h, val in zip(header, fields):
            entry[h] = parse_value(val)
        data.append(entry)
    return data


# folder = r'YOURPATHHERE' # Update this
# file_name = r'mydata'

# import_name = os.path.join(folder, file_name + ".txt")

# data = read_matlab_txt_file(import_name)
