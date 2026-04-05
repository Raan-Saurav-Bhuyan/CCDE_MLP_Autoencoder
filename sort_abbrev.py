# Import libraries: --->
import json

# Import constants: --->
ROOT = "abbrev"

# Define the file names: --->
input_filename = f"{ROOT}/input_dict.json"  # The input file containing the dictionary
output_filename = f"{ROOT}/sorted_dict.json"  # The output file for the sorted dictionary

# Open the file and load the dictionary: --->
with open(input_filename, "r") as file:
    data = json.load(file)

# Sort the dictionary by keys (acronyms): --->
sorted_data = dict(sorted(data.items()))

# Save the sorted dictionary to a new file: --->
with open(output_filename, "w") as file:
    json.dump(sorted_data, file, indent=4)

print(f"Sorted dictionary has been saved to {output_filename}")
