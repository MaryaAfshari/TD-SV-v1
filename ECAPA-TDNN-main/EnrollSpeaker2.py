#"16-05-2024" 27 Ordibehesht 1403 Panjshanbe "Ya Malekolhagholmobin"

def parse_eval_file(file_path):
    speaker_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            model_id = parts[0]
            speaker_id, phrase_id, model_index = model_id.split('_')
            wav_files = parts[1:]
            if speaker_id not in speaker_data:
                speaker_data[speaker_id] = []
            # Store the entire structure for later use
            #speaker_data[speaker_id][phrase_id][model_index] = wav_files
            speaker_data[speaker_id].append((phrase_id, model_index, wav_files))
    return speaker_data

def write_speaker_data_to_file(speaker_data, output_file_path):
    with open(output_file_path, 'w') as outfile:
        for speaker_id, utterances in speaker_data.items():
            outfile.write(f'{speaker_id}:\n')
            for phrase_id, model_index, wav_files in utterances:
                wav_files_str = ' '.join(wav_files)
                outfile.write(f'  {phrase_id}_{model_index}: {wav_files_str}\n')


dev_input_file = "../../../../../mnt/disk1/data/DeepMine/key/text-dependent/trn/ENG/male/100-spk/3-sess/dev.trn"
dev_output_file = '../../../ResultFile1-24-4-2024/dev_speaker_data.txt'


speaker_data = parse_eval_file(dev_input_file)
write_speaker_data_to_file(speaker_data, dev_output_file)

print(f'Speaker data has been written to {dev_output_file}')