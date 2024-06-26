#                                           In the name of GOD
#Author: Maryam Afshari
#05-05-2024-Yekshanbe 16-Ordibehesht-1403  ----> initial this code
#06-05-2024-Doshanbe 17-Ordibehesht-1403 ---> write trainlist in a code
#07-05-2024-Seshanbe 18-Ordibehesht-1403-  "YA arhamarahemin" ba tawasol be hazrat Zahar va Emam Sadegh (S)
#08-05-2024-May Chaharshanbe 19-2-1403-Ordibehesht - "Ya HAio ya ghaiom"
import numpy as np
import argparse, glob, os, torch, warnings, time
import torch, sys, os, tqdm, numpy, soundfile, time, pickle
#/mnt/disk1/data/DeepMine/key/text-dependent/trn/ENG/male/100-spk
background_addr = "../../../../../mnt/disk1/data/DeepMine/key/text-dependent/trn/ENG/male/100-spk/background.trn"
#ndx_dev_TC_addr = "../../../../../mnt/disk1/data/DeepMine/key/text-dependent/ndx/ENG/male/100-spk/3-sess/dev_IC.ndx" #Target Correct

input_file_path =  background_addr 
output_file_path = '../../../ResultFile1-24-4-2024/train_labels_background.txt'

with open(input_file_path, 'r') as file:
    # lines = file.readlines()[:20]  # Read only the first 4 lines
    lines = file.readlines()

number_of_lines = len(lines)
print("The file contains " + str(number_of_lines) + " lines in " + input_file_path + ".")


# Open the output file to write the results
with open(output_file_path, 'w') as output_file:
        
        for line in lines:
            parts = line.strip().split()
            utt_path = parts[0]
            speaker_id = parts[1]
            phrase_id = parts[2]
            label = '1'
            output_file.write(f"{speaker_id} {utt_path} {phrase_id} {label}\n")

print("Data for the lines has been written to train_labels.txt successfully.")

#check the number of lines written in 
line_Writing_count = 0
with open(output_file_path, 'r') as file:
    for line in file:
        line_Writing_count += 1

print(f"The file contains {number_of_lines} lines in {input_file_path}.")
print(f"The file contains {line_Writing_count} lines in {output_file_path}.")

print("Now I want to check reading from new trainlist file")
# Prompt user to continue or quit
if input("Press any key to continue, or 'q' to quit: ") == 'q':
    print("Exiting...")
    exit()

print("Continuing...")
train_path = "../../../../../mnt/disk1/data/DeepMine/wav"
data_list  = []
data_label = []
#with open(output_file_path, 'r') as file:
    # lines = file.readlines()[:20]  # Read only the first 4 lines
    #lines = file.readlines()

# lines = open(output_file_path).read().splitlines()
# dictkeys = list(set([x.split()[0] for x in lines]))
# dictkeys.sort()
# dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
# for index, line in enumerate(lines):
# 	speaker_label = dictkeys[line.split()[0]]
# 	file_name     = os.path.join(train_path, line.split()[1])
# 	data_label.append(speaker_label)
# 	data_list.append(file_name)
    

#second try : 
with open(output_file_path, 'r') as file:
    #lines = file.readlines()[:100]  # Read only the first 100 lines
    lines = file.readlines()
    
    #dictkeys = list(set([x.split()[0] for x in lines]))
    unique_speakers = set(line.split()[0] for line in lines)
    unique_speaker_count = len(unique_speakers)

    print(f"Number of unique speakers: {unique_speaker_count}")
    #print("Unique speakers:", unique_speakers)

    dictkeys = list(unique_speakers)
    dictkeys.sort()
    dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
    
    for line in lines:
        speaker_label = dictkeys[line.split()[0]]
        file_name = os.path.join(train_path, line.split()[1])
        file_name += ".wav"
        data_label.append(speaker_label)
        data_list.append(file_name)

print(f"speaker [0]= {data_label[0]},wav_path[0] = {data_list[0]} ")
print(f"speaker [10]= {data_label[10]},wav_path[10] = {data_list[10]} ")
print(f"speaker [99]= {data_label[99]},wav_path[99] = {data_list[99]} ")
#I want to read a wav file
num_frames = 200
audio, sr = soundfile.read(data_list[0])		
length = num_frames * 160 + 240
print(f"speaker [0]= {data_label[0]}, audio.shape[0]= {audio.shape[0]}, length = {length} ")
audio, sr = soundfile.read(data_list[1])		
print(f"speaker [1]= {data_label[1]}, audio.shape[0]= {audio.shape[0]}, sr = {sr}, length = {length} ")