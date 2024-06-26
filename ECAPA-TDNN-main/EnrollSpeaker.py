#09-05-2024-May Panjshanbe 19-2-1403-Ordibehesht - "Ya la elaha ella allah "
#12-05-2024 23 Ordibehesht 1403 Yekshanbe "Ya zol jalal val ekram"
#"16-05-2024" 27 Ordibehesht 1403 Panjshanbe "Ya Malekolhagholmobin"

#import numpy as np
#import argparse, glob, os, torch, warnings, time
#import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import os
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

from model import ECAPA_TDNN

DevInputFile = "../../../../../mnt/disk1/data/DeepMine/key/text-dependent/trn/ENG/male/100-spk/3-sess/dev.trn"
DevOutputFile = '../../../ResultFile1-24-4-2024/eval_lists_trn.txt'
output_file_path = "../../../ResultFile1-24-4-2024/embeddings_speakers_results.txt"
train_path = "../../../../../mnt/disk1/data/DeepMine/wav"

def perform_calculation(file_path):
    return len(file_path)

def compute_file_embeddings(file_path, speaker_encoder):
    #full_path = os.path.join(file_path, fp) + ".wav"
    audio, _ = sf.read(full_path)

    # Full utterance
    data_1 = torch.FloatTensor(np.stack([audio], axis=0)).cuda()

    # Split utterance matrix
    max_audio = 300 * 160 + 240
    if audio.shape[0] <= max_audio:
        shortage = max_audio - audio.shape[0]
        audio = np.pad(audio, (0, shortage), 'wrap')
    feats = []
    startframe = np.linspace(0, audio.shape[0] - max_audio, num=5)
    for asf in startframe:
        feats.append(audio[int(asf):int(asf) + max_audio])
    feats = np.stack(feats, axis=0).astype(np.float)
    data_2 = torch.FloatTensor(feats).cuda()

    # Speaker embeddings
    with torch.no_grad():
        embedding_1 = speaker_encoder.forward(data_1, aug=False)
        embedding_1 = F.normalize(embedding_1, p=2, dim=1)
        embedding_2 = speaker_encoder.forward(data_2, aug=False)
        embedding_2 = F.normalize(embedding_2, p=2, dim=1)

    return [embedding_1, embedding_2]


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
C=1024
speaker_encoder = ECAPA_TDNN(C = C).cuda()
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
                    #calculations.append(perform_calculation(full_path))
                    calculations.append(compute_file_embeddings(full_path,speaker_encoder))
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