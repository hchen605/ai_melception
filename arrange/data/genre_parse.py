import os
import json
from tqdm import tqdm

# Step 1: Parse the names of *.mid files in one folder
mid_files_folder = "../lmd_full"
#mid_files = [file for root, dir, file in os.walk(mid_files_folder) if file.endswith(".mid")]
mid_files = []
for root, dirs, files in os.walk(mid_files_folder):
    for file in files:
        if file.endswith(".mid"):
            mid_files.append(file)

#print('lmd amount: ', len(mid_files)) #178561

# Step 2: Read contents of the .txt file and store them in a dictionary
mapping_file_path = "LPD/matched_ids.txt"
mapping = {}
with open(mapping_file_path, 'r') as file:
    for line in file:
        key, value = line.strip().split()
        mapping[key] = value

#print('match amount: ', len(mapping)) #44735

# Step 3 & 4: Iterate through the *.mid file names and check for matches in labels
label_files_folders = ["LPD/amg", "LPD/lastfm", "LPD/tagtraum"]
#label_files_folders = ["LPD/amg", "LPD/tagtraum"]
mid_labels = {}
any_count = 0
for mid_file in tqdm(mid_files):
    # Check if the filename matches any key in the mapping dictionary
    if mid_file.split('.')[0] in mapping:
        label = mapping[mid_file.split('.')[0]]
        #print('found mapping')
        # Step 5: Check if the label exists in any of the label files
        for label_folder in label_files_folders:
            for label_file in os.listdir(label_folder):
                if label_file.startswith("id_list") and label_file.endswith(".txt"):
                    label_file_path = os.path.join(label_folder, label_file)
                    with open(label_file_path, 'r') as lf:
                        if label in lf.read().split('\n'):
                            if mid_file not in mid_labels:
                                mid_labels[mid_file] = os.path.splitext(label_file)[0].replace("id_list_", "").lower()[:3]
                            else:
                                mid_labels[mid_file] = mid_labels[mid_file] + '_' + os.path.splitext(label_file)[0].replace("id_list_", "").lower()[:3]
                            #print('found mapping: ', os.path.splitext(label_file)[0].replace("id_list_", "").lower()[:3])
                            #break
                        #else:
                        #   continue
                    #break
        #else:
            #mid_labels[mid_file] = 'any'
    else:
        mid_labels[mid_file] = 'any'
        any_count += 1

# Save the dictionary as a file for future usage
output_file_path = "lmd_genre_labels.json"
with open(output_file_path, 'w') as output_file:
    json.dump(mid_labels, output_file)

print("Labels saved to:", output_file_path)
print('any count: ', any_count)
