import os
import argparse

def generate_structure(base_folder, output_file):
    with open(output_file, 'w') as f:
        for user_id in os.listdir(base_folder):
            user_folder = os.path.join(base_folder, user_id)
            if os.path.isdir(user_folder):
                for utterance_id in os.listdir(user_folder):
                    utterance_folder = os.path.join(user_folder, utterance_id)
                    if os.path.isdir(utterance_folder):
                        for file_name in os.listdir(utterance_folder):
                            if file_name.endswith(".wav"):
                                file_path = os.path.join(
                                    utterance_folder, file_name)
                                relative_path = os.path.relpath(
                                    file_path, base_folder).replace("\\", "/")
                                f.write(f"{user_id} {relative_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate file structure.")
    parser.add_argument("base_folder", type=str, help="Path to the base folder")
    parser.add_argument("output_file", type=str, help="Output file name")
    args = parser.parse_args()

    generate_structure(args.base_folder, args.output_file)

# python generate_structure.py path_to_your_folder output_file.txt
# python generate_structure.py "D:\Data\Documents\HUST\lab914\VLSP-SV\E4\ECAPA-TDNN\data\train\wav" train_list.txt
