#09-05-2024-May Panjshanbe 19-2-1403-Ordibehesht - "Ya la elaha ella allah "
#12-05-2024 23 Ordibehesht Yekshanbe "Ya zol jalal val ekram"

#import numpy as np
#import argparse, glob, os, torch, warnings, time
#import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import os

DevInputFile = "../../../../../mnt/disk1/data/DeepMine/key/text-dependent/trn/ENG/male/100-spk/3-sess/dev.trn"
DevOutputFile = '../../../ResultFile1-24-4-2024/eval_lists_trn.txt'
output_file_path = "../../../ResultFile1-24-4-2024/embeddings_speakers_results.txt"
train_path = "../../../../../mnt/disk1/data/DeepMine/wav"

def perform_calculation(file_path):
    return len(file_path)


with open(DevInputFile, 'r') as file:
    lines = file.readlines()

number_of_lines = len(lines)
print("The file contains " + str(number_of_lines) + " lines in " + DevInputFile + ".")
print(".....................................................")
#files = []
#spkrs = []
speakers_files = {}
embeddings_speakers = {}
#lines = open(eval_list).read().splitlines()
#lines = lines.splitlines()
for line in lines:
            parts = line.split()
            speaker = parts[0]
            file_paths = parts[1:]
            # Add file paths to speakers_files dictionary
            speakers_files[speaker] = file_paths
            # Perform calculations on each file path
            calculations =[]
            for fp in file_paths:
                    # fp = os.path.join(train_path, fp)
                    # fp += ".wav"
                    full_path = os.path.join(train_path, fp) + ".wav"
                    calculations.append(perform_calculation(full_path))
            if calculations:
                average_result = sum(calculations) / len(calculations)
            else:
                average_result = 0  # Default value if no calculations are available
		    #avgerage_result = sum(calculations) / len(calculations)
            embeddings_speakers[speaker] = average_result

print("Speakers and their files:")
for speaker, files in speakers_files.items():
    print(f"{speaker}: {files}")

print("\nSpeakers and their average calculation results:")
for speaker, avg in embeddings_speakers.items():
    print(f"{speaker}: Average Calculation Result = {avg:.2f}")


with open(output_file_path, 'w') as output_file:
    for speaker, avg in embeddings_speakers.items():
        output_file.write(f"{speaker}: Average Calculation Result = {avg:.2f}\n")

print(f"\nResults have been written to {output_file_path}.")