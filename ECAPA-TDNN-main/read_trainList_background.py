#                                           In the name of GOD
#Author: Maryam Afshari
#05-05-2024-Yekshanbe 16-Ordibehesht-1403  ----> 
import numpy as np

background_addr = "../../../../../mnt/disk1/data/DeepMine/key/text-dependent/trn/ENG/male/100-spk/3-sess/background.trn"
#ndx_dev_TC_addr = "../../../../../mnt/disk1/data/DeepMine/key/text-dependent/ndx/ENG/male/100-spk/3-sess/dev_IC.ndx" #Target Correct

input_file_path =  background_addr 
output_file_path = '../../../ResultFile1-24-4-2024/train_labels_background.txt'

with open(input_file_path, 'r') as file:
    # lines = file.readlines()[:4]  # Read only the first 4 lines
    lines = file.readlines()[:20] 

number_of_lines = len(lines)
print(f"The file contains {number_of_lines} lines in {input_file_path}.")

for line in lines:
    parts = line.strip().split()
    utt_path = parts[0]
    speaker_id = parts[1]
    phrase_id = parts[2]
    label = '1'
    print(f" {speaker_id} {utt_path} {label} {phrase_id}\n")