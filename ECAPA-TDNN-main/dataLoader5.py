import glob, numpy, os, random, soundfile, torch
from scipy import signal

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

# تعریف نگاشت برای phrase_label های رشته‌ای
phrase_label_mapping = {
    "01": 1,
    "02": 2,
    "03": 3,
    "04": 4,
    "05": 5,
    "06": 6,
    "07": 7,
    "08": 8,
    "09": 9,
    "10": 10,
    "FT": 99,
    "A": 11,
    "B": 12,
    "C": 13,
    "D": 14,
    "E": 15,
    "F": 16,
    "G": 17,
    "H": 18,
    "I": 19,
    "J": 20,
    "K": 21,
    "L": 22,
    "M": 23,
    "N": 24,
    "O": 25,
    "P": 26,
    "Q": 27,
    "R": 28,
    "S": 29,
    "T": 30,
    "U": 31,
    "V": 32,
    "W": 33,
    "X": 34,
    "Y": 35,
    "Z": 36
}

class train_loader(object):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames
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
                phrase_label_str = line.split()[2]
                phrase_label = phrase_label_mapping.get(phrase_label_str, None)
                if phrase_label is None:
                    raise ValueError(f"Unknown phrase label: {phrase_label_str}")
                file_name = os.path.join(train_path, line.split()[0])  # Changed to index 0 for train-file-id
                file_name += ".wav"
                
                self.speaker_labels.append(speaker_label)
                self.phrase_labels.append(phrase_label)
                self.data_list.append(file_name)
            except ValueError as e:
                print(f"Skipping line due to error: {e}")

    def __getitem__(self, index):
        # Read the utterance and randomly select the segment
        file_name = self.data_list[index]
        audio, sr = soundfile.read(file_name)
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)
        
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
        phoneme_symbols = phrases_to_phonemes[str(self.phrase_labels[index]).zfill(2)]
        phoneme_ids = torch.LongTensor([ord(ch) for ch in ''.join(phoneme_symbols)]).cuda()
        
        return torch.FloatTensor(audio[0]), self.speaker_labels[index], self.phrase_labels[index], phoneme_ids

    def __len__(self):
        return len(self.data_list)

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio
