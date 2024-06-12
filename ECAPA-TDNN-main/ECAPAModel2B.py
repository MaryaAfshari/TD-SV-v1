import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import os
import sys
import tqdm
import time
import pickle

from tools2B import *
from loss2B import AAMsoftmax
from model2B import ECAPA_TDNN

class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step, **kwargs):
        super(ECAPAModel, self).__init__()
        ## ECAPA-TDNN
        self.speaker_encoder = ECAPA_TDNN(C=C).cuda()

        ## Classifier
        self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s).cuda()
        #add normal softmax  11 class------------ 0-10 phrase 
        #text_loss

        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

#train_network Method ...
    def train_network(self, epoch, loader):
        print("hello, this in train network ... ECAPAModel.py")
        self.train()
        ## Update the learning rate based on the current epoch
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        print("Loader Length = ", loader.__len__())

        for num, (data, speaker_labels, phrase_labels) in enumerate(loader, start=1):
            self.zero_grad()
            speaker_labels = torch.LongTensor(speaker_labels).cuda()
            # Print data for debugging
            # print(f"Data batch {num}:")
            # print(f"  - Data shape: {data.shape}")
            # print(f"  - Speaker labels: {speaker_labels}")
            # print(f"  - Phrase labels: {phrase_labels}")
            # Forward pass
            speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug=True)
            ##
            nloss, prec = self.speaker_loss.forward(speaker_embedding, speaker_labels)
            # Backward pass and optimization
            nloss.backward()
            self.optim.step()

            index += len(speaker_labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()

            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                             " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) + \
                             " Loss: %.5f, ACC: %2.2f%% \r" % (loss / (num), top1 / index * len(speaker_labels)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss / num, lr, top1 / index * len(speaker_labels)

#eval_network Method ...
    def eval_network(self, eval_list, eval_path):
        print("hello, this in eval network ... ECAPAModel2.py")
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        lines = lines[1:]  # Skip the header row
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
            file_name =os.path.join(eval_path, file)
            file_name += ".wav" 
            audio, _ = sf.read(file_name)
            # Full utterance
            data_1 = torch.FloatTensor(np.stack([audio], axis=0)).cuda()

            # Splitted utterance matrix
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
                embedding_1 = self.speaker_encoder.forward(data_1, aug=False)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.speaker_encoder.forward(data_2, aug=False)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]
        scores, labels = [], []

        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))

        # Compute EER and minDCF
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

        return EER, minDCF

#enroll_network Method ...
    def enroll_network(self, enroll_list, enroll_path, path_save_model):
        print("hello, this in enroll network ... ECAPAModel2.py")
        self.eval()
        enrollments = {}
        lines = open(enroll_list).read().splitlines()
        lines = lines[1:]  # Skip the header row
        for line in lines:
            parts = line.split()
            model_id = parts[0]
            enroll_files = parts[3:]  # Enrollment file IDs
            embeddings = []
            for file in enroll_files:
                file_name = os.path.join(enroll_path, file)
                file_name += ".wav" 
                audio, _ = sf.read(file_name)
                data = torch.FloatTensor(np.stack([audio], axis=0)).cuda()
                with torch.no_grad():
                    embedding = self.speaker_encoder.forward(data, aug=False)
                    embedding = F.normalize(embedding, p=2, dim=1)
                embeddings.append(embedding)
            enrollments[model_id] = torch.mean(torch.stack(embeddings), dim=0)#make avg of 3 enrollment file embeddings

        # with open("enrollments.pkl", "wb") as f:
        #     pickle.dump(enrollments, f)

        # Ensure the directory exists
        os.makedirs(path_save_model, exist_ok=True)

        # Save enrollments using the provided path
        with open(os.path.join(path_save_model, "enrollments.pkl"), "wb") as f:
            pickle.dump(enrollments, f)

#test_network Method ...
    def test_network(self, test_list, test_path, path_save_model):
        print("hello, this in test network ... ECAPAModel2.py")
        self.eval()
        # with open("enrollments.pkl", "rb") as f:
        #     enrollments = pickle.load(f)
        # Loading enrollments
        enrollments_path = os.path.join(path_save_model, "enrollments.pkl")
        print(f"Loading enrollments from {enrollments_path}")
        with open(os.path.join(path_save_model, "enrollments.pkl"), "rb") as f:
            enrollments = pickle.load(f)

        scores, labels = [], []
        lines = open(test_list).read().splitlines()
        lines = lines[1:]  # Skip the header row
        for line in lines:
            parts = line.split()
            model_id = parts[0]
            test_file = parts[1]
            trial_type = parts[2]
            # Assign labels based on trial-type
            if trial_type in ['TC', 'TW']:
                label = 1
            else:
                label = 0
            #label = int(parts[2])
            file_name = os.path.join(test_path, test_file)
            file_name += ".wav"
            #audio, _ = sf.read(os.path.join(test_path, test_file))
            audio, _ = sf.read(file_name)
            data = torch.FloatTensor(np.stack([audio], axis=0)).cuda()
            with torch.no_grad():
                test_embedding = self.speaker_encoder.forward(data, aug=False)
                test_embedding = F.normalize(test_embedding, p=2, dim=1)

            score = torch.mean(torch.matmul(test_embedding, enrollments[model_id].T)).detach().cpu().numpy()
            scores.append(score)
            labels.append(label)

        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

        return EER, minDCF

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
