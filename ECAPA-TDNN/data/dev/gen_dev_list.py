import os
import random
import argparse

# Define the path to the directory containing the audio files
base_folder = "/kaggle/input/nndang-vn-celeb/VLSP23_VN_celeb2/valid_celeb"

# Define the path to the output file
output_file = "/kaggle/working/veri_test.txt"



def generate_structure(base_folder, output_file):
    # Create a list to store the file paths
    file_paths = []

    # Iterate through the directories and collect the audio file paths
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(
                    file_path, base_folder).replace("\\", "/")
                file_paths.append(relative_path)

    # Shuffle the list of file paths
    random.shuffle(file_paths)
    print(len(file_paths))

    # Create a list to store the output lines
    output_lines = []

    # Iterate through the shuffled file paths and generate output lines
    for i in range(0, len(file_paths), 1):
        if i+1 < len(file_paths):  # Ensure there are at least two files left
            file1 = file_paths[i]
            file2 = file_paths[i + 1]
            label = 1 if os.path.dirname(file1) == os.path.dirname(file2) else 0
            output_lines.append(f"{label} {file1} {file2}\n")

    # Write the output lines to the output file
    with open(output_file, "w") as f:
        f.writelines(output_lines)

    print(f"Output written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate file structure.")
    parser.add_argument("base_folder", type=str, help="Path to the base folder")
    parser.add_argument("output_file", type=str, help="Output file name")
    args = parser.parse_args()
    generate_structure(args.base_folder, args.output_file)
