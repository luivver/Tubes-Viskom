import os

def count_labels_in_folder(folder_path):
    # Initialize counters for each label
    label_counts = {"0": 0, "1": 0}

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Ensure it's a file
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                # Read the first character (binary label)
                label = file.read(1).strip()
                if label in label_counts:
                    label_counts[label] += 1

    return label_counts

# Paths to the two folders
folder1_path = "/Users/zevapatu/train-object-detector-detectron2/data/train/anns"
folder2_path = "/Users/zevapatu/train-object-detector-detectron2/data/val/anns"

# Count labels in each folder
counts_folder1 = count_labels_in_folder(folder1_path)
counts_folder2 = count_labels_in_folder(folder2_path)

# Output the counts
print("Train:")
print(f"Label '0'/'Negative': {counts_folder1['0']} files")
print(f"Label '1'/'Positive': {counts_folder1['1']} files")

print("\nVal:")
print(f"Label '0'/'Negative': {counts_folder2['0']} files")
print(f"Label '1'/'Positive': {counts_folder2['1']} files")
