#In the name of GOD
#Import 
import numpy as np
#Function Definition
def split_and_print(line):
    # Split the line into different parts/columns using spaces as the delimiter
    parts = line.split()

    # Extract and print each part
    for part in parts:
        print(part)
#30-march-2023 - shanbe-- 11-farvardin-1403
#4-4-2024 - panjshanbe - 16-farvardin-1403
#7-4-2024 - Yekshanbe -
#8-4-2024 - Doshanbe -
#10-4-2024- Chaharshanbe-22-Farvardin
#15-4-2024- Dooshanbe - 27-farvardin-1403
#17-4-2024 - Chaharshanbe -29-farvardin-1403
#18-4-2024 - Panjshanbe - 30-farvardin-1403
#19-4-2024 -Jomee -31-farvardin-1403--ei hyva paljon
#20-4-2024 - Shanbe-1-ordibehesht-1403--ei hyva paljon
#21-4-2024 - Yekshanbe-2-Oridibehesht --ei hyva paljon
#22-4-2024 - Dooshanbe-3-Oridibehesht --ei hyva paljon
#24-4-2024-Chaharshanbe 5-Ordibehesht---> read SVSD challenge discription -hyvä
#today goals: 1-how to read files of dev.trn and dev_TC.ndx
#             2-read the wav file of address
#             3-read all files for train and use in ECAPA-TDNN model
#             4-Do the above work for TC,TW,IC and IW is always 0 and not mentioned
#             5-Do above for Test also
#             6-read model with 2 part: Speaker recognizer & Phrase recognizer a multi task learning and SSL pretrained model (WAV2VEc or WavLM)
#             7-add Phrase recognizer
#26-5-2024-Jomeh-7-ordibehesht  --->
print("Hello Practical Speech word date :4-April-2024")
print("Hello Practical Speech word date :10-April-2024")
print("Hello Practical Speech word date :15-April-2024")
print("Hello Practical Speech word date :17-April-2024")
print("Hello Practical Speech word date :18-April-2024 = 30-farvardin-1403")
print("Hello Practical Speech word date :19-April-2024 = 31-farvardin-1403")
print("Hello Practical Speech word date :20-April-2024 = 1-ordibehesht-1403")
print("Hello Practical Speech word date :21-April-2024 = 2-ordibehesht-1403")
print("Hello Practical Speech word date :22-April-2024 = 3-ordibehesht-1403")
print("Hello Practical Speech word date :24-April-2024 = 5-ordibehesht-1403")
print("Hello Practical Speech word date :26-April-2024 = 7-ordibehesht-1403")
#code address : /code1-29-3-2024/TD-SV-v1/ECAPA-TDNN-main
# trn address file: /mnt/disk1/data/DeepMine/key/text-dependent/trn/ENG/male/100-spk/3-sess
# ndx address file: /mnt/disk1/data/DeepMine/key/text-dependent/ndx/ENG/male/100-spk/3-sess
trn_dev_addr = "../../../../../mnt/disk1/data/DeepMine/key/text-dependent/trn/ENG/male/100-spk/3-sess/dev.trn"
ndx_dev_TC_addr = "../../../../../mnt/disk1/data/DeepMine/key/text-dependent/ndx/ENG/male/100-spk/3-sess/dev_TC.ndx" #Target Correct
# Spkr = []
# Utt = []

# Open the text file
with open(trn_dev_addr, 'r') as file:
    lines = file.readlines()[:3]  # Read only the first two lines

# Initialize an empty dictionary to store utterances by speaker
utterances_by_speaker = {}

# Process each line
for line in lines:
    # Split the line into parts
    parts = line.split()

    # Extract the speaker and utterance paths
    speaker = parts[0]
    print("speaker = "+speaker)
    utt_paths = parts[1:]
    print("utt_paths0 = "+utt_paths[0])
    print("utt_paths1 = "+utt_paths[1])
    print("utt_paths2 = "+utt_paths[2])

    # # Group the utterances for each speaker
    # if speaker in utterances_by_speaker:
    #     utterances_by_speaker[speaker].append(utt_paths)
    # else:
    #     utterances_by_speaker[speaker] = [utt_paths]
        
"""
# Convert the dictionary of utterances to NumPy arrays
for speaker in utterances_by_speaker:
    utterances_by_speaker[speaker] = np.array(utterances_by_speaker[speaker])

# Print the utterances by speaker
for speaker, utterances in utterances_by_speaker.items():
    print("Speaker:", speaker)
    for i, utt_paths in enumerate(utterances):
        print("Group", i+1, "utterances:", utt_paths)"""


# Command + / (Mac)
# with open(trn_dev_addr, 'r') as file:
#     line_count = 0
#     for line in file:
#         print(line)
#         split_and_print(line)
#         parts = line.split()
#             # Extract and print each part
#         for part in parts:
#             print(part)
#         line_count += 1
#         if line_count == 2:
#             break


with open(ndx_dev_TC_addr, 'r') as file:
    line_count = 0
    for line in file:
        print(line)
        split_and_print(line)
        line_count += 1
        if line_count == 2:
            break