#                                           In the name of GOD
#Author: Maryam Afshari
#24-4-2024-Chaharshanbe 5-Ordibehesht---> read SVSD challenge discription -hyvä -Vaasa

import numpy as np

trn_dev_addr = "../../../../../mnt/disk1/data/DeepMine/key/text-dependent/trn/ENG/male/100-spk/3-sess/dev.trn"
ndx_dev_TC_addr = "../../../../../mnt/disk1/data/DeepMine/key/text-dependent/ndx/ENG/male/100-spk/3-sess/dev_TC.ndx" #Target Correct

# Define the file path for reading and the output file path
input_file_path =  trn_dev_addr # Adjust this to your input file path
output_file_path = '../../../ResultFile1-24-4-2024/train_labels.txt'

# Open the input file and read the lines
with open(input_file_path, 'r') as file:
    lines = file.readlines()[:4]  # Read only the first 4 lines

# Open the output file to write the results
with open(output_file_path, 'w') as output_file:
    # Write the header line
    output_file.write('Train_file_id,Speaker_id,Phrase_id,label\n')
    
    # Process each line from the input file
    for line in lines:
        parts = line.strip().split()
        
        # The model_id and speaker information is taken directly from the first part
        model_id = parts[0]
        
        # Splitting the model_id to extract speaker_id and phrase_id
        # Assuming the format is always like Spk000099_09_01
        speaker_id, phrase_id, _ = model_id.split('_')
        
        # The label is fixed as 1 according to your requirement
        label = '1'
        
        # # Write the formatted output to the train_labels.txt
        # # Joining model_id, speaker_id, phrase_id, and label with comma separation
        # output_file.write(f"{model_id},{speaker_id},{phrase_id},{label}\n")

        # Process each utterance path (assuming at least one utterance path is present)
        for utt_path in parts[1:]:
            # Write the formatted output to the train_labels.txt
            # Joining model_id, speaker_id, phrase_id, Phrase_path, and label with comma separation
            output_file.write(f"{model_id},{speaker_id},{phrase_id},{utt_path},{label}\n")

print("Data for the first 4 lines has been written to train_labels.txt successfully.")
