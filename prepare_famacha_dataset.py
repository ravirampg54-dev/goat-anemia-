import os
import shutil
import random

SOURCE = r"dataset with goay eye with annotation\test"

OUTPUT = "dataset_anemia"

CLASSES = {
    "famacha 1": "FAMACHA_1",
    "famacha 2": "FAMACHA_2",
    "famacha 3": "FAMACHA_3",
    "famacha 4": "FAMACHA_4",
}

random.seed(42)

# Create folders
for split in ["train", "valid", "test"]:
    for class_name in CLASSES.values():
        os.makedirs(
            os.path.join(OUTPUT, split, class_name),
            exist_ok=True
        )

for source_class, output_class in CLASSES.items():

    source_folder = os.path.join(SOURCE, source_class)

    if not os.path.exists(source_folder):
        print("WARNING: Folder not found:", source_folder)
        continue

    files = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    random.shuffle(files)

    total = len(files)

    # With very small classes, keep at least one image for test
    if total >= 5:
        train_count = max(1, int(total * 0.7))
        valid_count = max(1, int(total * 0.2))
    else:
        train_count = max(1, total - 1)
        valid_count = 0

    train_files = files[:train_count]
    valid_files = files[train_count:train_count + valid_count]
    test_files = files[train_count + valid_count:]

    print()
    print(output_class)
    print("Total:", total)
    print("Train:", len(train_files))
    print("Valid:", len(valid_files))
    print("Test:", len(test_files))

    for f in train_files:
        shutil.copy2(
            os.path.join(source_folder, f),
            os.path.join(OUTPUT, "train", output_class, f)
        )

    for f in valid_files:
        shutil.copy2(
            os.path.join(source_folder, f),
            os.path.join(OUTPUT, "valid", output_class, f)
        )

    for f in test_files:
        shutil.copy2(
            os.path.join(source_folder, f),
            os.path.join(OUTPUT, "test", output_class, f)
        )

print()
print("=" * 60)
print("FAMACHA DATASET PREPARATION COMPLETE")
print("=" * 60)