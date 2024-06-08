#Ya Latif
#Date: 18.3.1403 Khordad mah
#Date: 7.6.2024 June 
#Author: Maryam Afshari -Iranian
#part1- without phoneme aligner
import glob
import numpy as np
import os
import random
import soundfile as sf
import torch
from scipy import signal
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

phrases_to_phonemes = {
    "01": ["S", "E", "D", "AA", "Y", "E", "-", "M", "A", "N", "-", "N", "E", "SH", "AA", "N", "D", "A", "H", "A", "N", "D", "E", "Y", "E", "-", "H", "O", "V", "I", "Y", "Y", "A", "T", "E", "-", "M", "A", "N", "-", "AH", "A", "S", "T"],
    "02": ["S", "E", "D", "AA", "Y", "E", "-", "H", "A", "R", "-", "K", "A", "S", "-", "M", "O", "N", "H", "A", "S", "E", "R", "-", "B", "E", "-", "F", "A", "R", "D", "-", "AH", "A", "S", "T"],
    "03": ["H", "O", "V", "I", "Y", "Y", "A", "T", "E", "-", "M", "A", "N", "-", "R", "AA", "-", "B", "AA", "-", "S", "E", "D", "AA", "Y", "E", "-", "M", "A", "N", "-", "T", "A", "AH", "Y", "I", "D", "-", "K", "O", "N"],
    "04": ["S", "E", "D", "AA", "Y", "E", "-", "M", "A", "N", "-", "R", "A", "M", "Z", "E", "-", "AH", "O", "B", "U", "R", "E", "-", "M", "A", "N", "-", "AH", "A", "S", "T"],
    "05": ["B", "A", "N", "I", "AH", "AA", "D", "A", "M", "-", "AH", "A", "AH", "Z", "AA", "Y", "E", "-", "Y", "E", "K", "D", "I", "G", "A", "R", "A", "N", "D"],
    "06": ["M", "AY", "-", "V", "OY", "S", "-", "IH", "Z", "-", "M", "AY", "-", "P", "AE", "S", "W", "ER", "D"],
    "07": ["OW", "K", "EY", "-", "G", "UW", "G", "AH", "L"],
    "08": ["AA", "R", "T", "AH", "F", "IH", "SH", "AH", "L", "-", "IH", "N", "T", "EH", "L", "AH", "JH", "AH", "N", "S", "-", "IH", "Z", "-", "F", "AO", "R", "-", "R", "IY", "L"],
    "09": ["AE", "K", "SH", "AH", "N", "Z", "-", "S", "P", "IY", "K", "-", "L", "AW", "D", "ER", "-", "DH", "AE", "N", "-", "W", "ER", "D", "Z"],
    "10": ["DH", "EH", "R", "-", "IH", "Z", "-", "N", "OW", "-", "S", "AH", "CH", "-", "TH", "IH", "NG", "-", "AE", "Z", "-", "EY", "-", "F", "R", "IY", "-", "L", "AH", "N", "CH"]
}

def check_phrase_labels(train_list):
    """
    This function checks if all phrase labels in the train list are integers.
    If it finds a non-integer phrase label, it prints the label and exits the check.
    """
    lines = open(train_list).read().splitlines()
    lines = lines[1:]  # Skip the header row
    all_integers = True
    for line in lines:
        try:
            phrase_label = int(line.split()[2])  # Trying to convert to integer
        except ValueError:
            all_integers = False
            print(f"Non-integer phrase ID found: {line.split()[2]}")
            break
    if all_integers:
        print("All phrase labels are integers.")
    else:
        print("There are non-integer phrase labels in the file.")


def inspect_train_list_for_ft(train_list):
    """
    This function inspects the train list and prints lines that contain the phrase ID "FT".
    """
    # Inspect the train list entries
    with open(train_list, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines) - 1  # Subtract one for the header
    ft_count = 0

    # Print lines containing "FT"
    for line in lines[1:]:  # Skip the header row
        if "FT" in line:
            print(line)
            ft_count += 1
    
    if total_lines > 0:
        ft_percentage = (ft_count / total_lines) * 100
        print(f"'FT' phrase labels account for {ft_count} out of {total_lines} lines ({ft_percentage:.2f}%).")
    else:
        print("The train list is empty or only contains the header.")

class train_loader(Dataset):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames

        # Call the new functions to check and inspect the train list
        check_phrase_labels(train_list)
        inspect_train_list_for_ft(train_list)

        # Load and configure augmentation files
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))
        
        # Load data & labels
        self.data_list = []
        self.speaker_labels = []
        self.phrase_labels = []  # New phrase labels
        
        lines = open(train_list).read().splitlines()
        lines = lines[1:]  # Skip the header row
        dictkeys = list(set([x.split()[1] for x in lines]))  # Changed to index 1 for speaker-id
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
        
        for index, line in enumerate(lines):
            try:
                speaker_label = dictkeys[line.split()[1]]  # Changed to index 1 for speaker-id
                phrase_label = line.split()[2]  # Assuming phrase ID is the third column
                file_name = os.path.join(train_path, line.split()[0])  # Changed to index 0 for train-file-id
                file_name += ".wav"
                
                if phrase_label == "FT":
                    print(f"Ignoring entry with 'FT' phrase label: {line}")
                    continue  # Skip this entry

                self.speaker_labels.append(speaker_label)
                self.phrase_labels.append(phrase_label)
                self.data_list.append(file_name)
            except ValueError:
                print(f"Skipping line with non-integer phrase ID: {line}")

    def __getitem__(self, index):
        # Read the utterance and randomly select the segment
        file_name = self.data_list[index]
        audio, sr = sf.read(file_name)
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        start_frame = np.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = np.stack([audio], axis=0)
        
        # Data Augmentation
        augtype = 0
        if augtype == 0:  # Original
            audio = audio
        elif augtype == 1:  # Reverberation
            audio = self.add_rev(audio)
        elif augtype == 2:  # Babble
            audio = self.add_noise(audio, 'speech')
        elif augtype == 3:  # Music
            audio = self.add_noise(audio, 'music')
        elif augtype == 4:  # Noise
            audio = self.add_noise(audio, 'noise')
        elif augtype == 5:  # Television noise
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')

        # Convert phoneme symbols to numeric IDs
        #phoneme_symbols = phrases_to_phonemes[str(self.phrase_labels[index])]
        #phoneme_ids = torch.LongTensor([ord(ch) for ch in ''.join(phoneme_symbols)])
        
        # Convert phoneme symbols to numeric IDs
        phrase_label = str(self.phrase_labels[index])
        if phrase_label in phrases_to_phonemes:
            phoneme_symbols = phrases_to_phonemes[phrase_label]
            phoneme_ids = torch.LongTensor([ord(ch) for ch in ''.join(phoneme_symbols)])
        else:
            # Handle the case where the phrase label is not found
            phoneme_ids = torch.LongTensor([])  # Empty tensor or you can choose a default behavior

        return torch.FloatTensor(audio[0]), self.speaker_labels[index], self.phrase_labels[index], phoneme_ids

    def __len__(self):
        return len(self.data_list)

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = sf.read(rir_file)
        rir = np.expand_dims(rir.astype(np.float32), 0)
        rir = rir / np.sqrt(np.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = sf.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = np.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = np.stack([noiseaudio], axis=0)
            noise_db = 10 * np.log10(np.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio

def collate_fn(batch):
    # Assume batch is a list of tuples (data, speaker_labels, phrase_labels, phoneme_ids)
    data, speaker_labels, phrase_labels, phoneme_ids = zip(*batch)

    # Pad sequences
    data = pad_sequence(data, batch_first=True)
    speaker_labels = torch.tensor(speaker_labels)
    phrase_labels = torch.tensor(phrase_labels)
    phoneme_ids = pad_sequence(phoneme_ids, batch_first=True)

    return data, speaker_labels, phrase_labels, phoneme_ids

# Create a DataLoader using the custom collate function
def create_dataloader(train_list, train_path, musan_path, rir_path, num_frames, batch_size, n_cpu):
    dataset = train_loader(train_list, train_path, musan_path, rir_path, num_frames)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu, collate_fn=collate_fn)
