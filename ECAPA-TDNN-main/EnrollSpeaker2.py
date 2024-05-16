#"16-05-2024" 27 Ordibehesht 1403 Panjshanbe "Ya Malekolhagholmobin"
import os, soundfile, torch, numpy
import torch.nn.functional as F
from model import ECAPA_TDNN

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

def process_wav_files_by_speaker(speaker_data, speaker_id):
    if speaker_id in speaker_data:
        embeddings = []
        total_length = 0 
        total_wav_files = 0
        speaker_encoder = ECAPA_TDNN(C = C).cuda()
        for phrase_id, model_index, wav_files in speaker_data[speaker_id]:
            for wav_file in wav_files:
                #audio, sr = load_audio(wav_file)
                audio = load_audio(wav_file)
                total_length = len(audio)
                print(f'Processing {wav_file} for speaker {speaker_id}, phrase {phrase_id}, model {model_index}, audio length: {total_length}')
                #print(f'Processing {wav_file} for speaker {speaker_id}, phrase {phrase_id}, model {model_index}, sample rate: {sr}')
                total_wav_files += 1 
                embedding = extract_embedding(speaker_encoder, audio)
                embeddings.append(embedding)
                # Implement your processing logic here
        print(f'Total number of wav files for speaker {speaker_id}: {total_wav_files}')
        if embeddings:
            avg_embedding = numpy.mean(embeddings, axis=0)
            print(f'Average embedding for speaker {speaker_id}: {avg_embedding}')
            return avg_embedding
        else:
            print(f'No embeddings found for speaker {speaker_id}')
            return None
    else:
        print(f'No data found for speaker {speaker_id}')
        return None

def load_audio(audio_path):
    #import librosa
    #audio, sr = librosa.load(audio_path, sr=16000)
    audio, _ = soundfile.read(os.path.join(wav_path, audio_path)+ ".wav")
    return audio 


def extract_embedding(model, audio):
    embedding_audio = {}
    speaker_encoder = model
    # Full utterance
    data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()
    # Spliited utterance matrix
    max_audio = 300 * 160 + 240
    if audio.shape[0] <= max_audio:
        shortage = max_audio - audio.shape[0]
        audio = numpy.pad(audio, (0, shortage), 'wrap')
    feats = []
    startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
    for asf in startframe:
        feats.append(audio[int(asf):int(asf)+max_audio])
    feats = numpy.stack(feats, axis = 0).astype(numpy.float)
    data_2 = torch.FloatTensor(feats).cuda()
	# Speaker embeddings
    with torch.no_grad():
        embedding_1 = speaker_encoder.forward(data_1, aug = False)
        embedding_1 = F.normalize(embedding_1, p=2, dim=1)
        embedding_2 = speaker_encoder.forward(data_2, aug = False)
        embedding_2 = F.normalize(embedding_2, p=2, dim=1)
    embedding_audio[audio] = [embedding_1, embedding_2]
    return embedding_audio

dev_input_file = "../../../../../mnt/disk1/data/DeepMine/key/text-dependent/trn/ENG/male/100-spk/3-sess/dev.trn"
dev_output_file = '../../../ResultFile1-24-4-2024/dev_speaker_data.txt'
wav_path = "../../../../../mnt/disk1/data/DeepMine/wav"
C = 1024

speaker_data = parse_eval_file(dev_input_file)
write_speaker_data_to_file(speaker_data, dev_output_file)

print(f'Speaker data has been written to {dev_output_file}')
print(process_wav_files_by_speaker(speaker_data, 'Spk000191'))