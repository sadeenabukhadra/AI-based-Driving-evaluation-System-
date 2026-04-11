import os

base_path = "dataset"

wrong_files = []

for dataset_name in os.listdir(base_path):
    dataset_path = os.path.join(base_path, dataset_name)

    # ندخل فقط على الفولدرات (cone1, cone2)
    if not os.path.isdir(dataset_path):
        continue

    for split in ["train", "valid"]:
        labels_path = os.path.join(dataset_path, split, "labels")

        if not os.path.exists(labels_path):
            continue

        for file in os.listdir(labels_path):
            file_path = os.path.join(labels_path, file)

            with open(file_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                class_id = line.strip().split()[0]

                if class_id != "0":
                    wrong_files.append((dataset_name, file, class_id))

# ✅ النتيجة
if len(wrong_files) == 0:
    print("✅ كل الليبلز في cone1 و cone2 صحيحة (كلها 0)")
else:
    print("❌ في أخطاء:")
    for wf in wrong_files[:20]:
        print(wf)