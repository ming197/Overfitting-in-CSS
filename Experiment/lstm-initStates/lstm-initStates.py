import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
import myDataloader as db
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.numLayers = 1
        self.input_dim = 1
        self.output_dim = self.input_dim
        self.hidden_size = 128
        self.batchSize = db.batchSize  # or numData when numData < batchSize
        self.numData = db.numSequences
        if(self.numData <= self.batchSize):
            self.batchSize = self.numData

        self.lstm1 = nn.LSTM(self.input_dim, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_dim)

        h0 = torch.zeros(self.numLayers, self.numData, self.hidden_size).to(device)
        # print("h0 size: ", h0.size())
        self.h0 = nn.Parameter(h0, requires_grad=True)


    def forward(self, input, label):
        seqLength = input.shape[0]
        numDataInBatch = input.shape[1]
        for i in range(len(label)):
            idxSeq = label[i][0]
            temp = self.h0[:,idxSeq,:].view(1,1,-1)
            if(i == 0):
                h0 = temp
            else:
                h0 = torch.cat((h0, temp), 1)

        for t in range(seqLength):
            if(t ==0):
                h_t = h0
                c_t = torch.zeros(self.numLayers, numDataInBatch, self.hidden_size).to(device)
                print("h_t size: ", h_t.size())
                print("c_t size: ", c_t.size())
                input_t = torch.zeros(1, numDataInBatch, self.input_dim).to(device)  # firstDim = time = 1
            else:
                input_t = output

            output, (h_t, c_t) = self.lstm1(input_t, (h_t, c_t))

            output = self.linear(h_t)

            if (t == 0):
                outputs = output
            else:
                outputs = torch.cat((outputs,output),0)

        return outputs


if __name__ == "__main__":

    trainModel = False # True for train, False for evaluation
    # if(len(sys.argv) != 2 or (sys.argv[1] != 'train' and sys.argv[1] != 'eval')):
    #     print('Please specify the mode.\ne.g. python lstm-initStates.py train  or python lstm-initStates.py check')
    #     exit()

    # if(sys.argv[1] == 'train'):
    if True:
        learningRate = 0.001
        numEpoch = 100000

        model = Sequence().to(device)
        optimizer = Adam(params=model.parameters(), lr=learningRate)
        criterion = nn.MSELoss()


        fid = open('./result/loss.txt','w')
        minLoss = 0.001

        for epoch in range(numEpoch):
            for index, (nn_in, nn_label) in enumerate(db.data_loader):
                # nn_in: 735 * 241
                # 训练集和测试集的样本数，以及序列长度
                train_nums = 400
                test_nums = nn_in.size()[0] - train_nums
                train_seqLenth = db.seqLenth
                test_seqLenth = db.seqLenth
                # 拆分训练集数据和测试集数据
                nn_in_train, nn_in_test = nn_in.split([train_nums, test_nums], dim = 0)
                nn_in_train = nn_in_train.to(device)
                nn_in_test = nn_in_test.to(device)
                label_train = nn_label[:train_nums]
                label_test = nn_label[train_nums:]
                # transform
                train_data = nn_in_train.permute(1, 0)  # (train_seqLenth, train_nums)
                test_data = nn_in_test.permute(1, 0)
                train_data = train_data.view(train_seqLenth, -1, model.output_dim)
                test_data = test_data.view(test_seqLenth, -1, model.output_dim)
                
                # output of train and test
                output_train = model.forward(train_data, label_train)
                output_test = model.forward(test_data, label_test)
                # optimizer
                optimizer.zero_grad()
                loss_train = criterion(output_train, train_data)
                loss_test = criterion(output_test, test_data)
                loss_train = loss_train.to(device)
                loss_test = loss_test.to(device)
                loss_train.backward()
                optimizer.step()

            if epoch % 10 == 0:
                loss_val_train = loss_train.detach().cpu().numpy()
                loss_val_test = loss_test.detach().cpu().numpy()
                print(f'Epoch {epoch}, loss_train: {loss_val_train:.8f}, loss_test: {loss_val_test:.8f}')
                fid.write('%d %.8f %.8f\n'%(epoch, loss_val_train, loss_val_test))
                fid.flush()
                if (minLoss > loss_val_train):
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_train
                    }, './result/model_best.tar')

        fid.close()
    elif (sys.argv[1] == 'eval'):
        if(torch.cuda.is_available()):
            checkpoint = torch.load('./result/model_best.tar')
        else:
            checkpoint = torch.load('./result/model_best.tar', map_location=torch.device('cpu'))

        learningRate = 0.001
        numEpoch = 100000

        model = Sequence().to(device)
        optimizer = Adam(params=model.parameters(), lr=learningRate)
        criterion = nn.MSELoss()

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        model.train()

        # Save the initial hidden states (h0)
        fid = open("./result/initStates_from_best.txt", 'w')
        for idx in range(model.h0.shape[1]):
            for dim in range(model.h0.shape[2]):
                fid.write('%.8f\t' % model.h0[0][idx][dim].item())
            fid.write('\n')
        fid.close()


        # Save the model's output (y)
        Path("./result/output").mkdir(parents=True, exist_ok=True)
        for index, (nn_in, nn_label) in enumerate(db.data_loader):
            nn_in = nn_in.to(device)
            train_data = nn_in.permute(1, 0)  # (seqLength, batchsize)
            train_data = train_data.view(db.seqLenth, -1, model.output_dim)
            output = model.forward(train_data, nn_label)


            for idxBatch in range(output.shape[1]):
                print('Saving %d / %d' % (idxBatch+1,output.shape[1]))
                idxSeq = nn_label[idxBatch][0]

                plt.figure(figsize=(30, 10))
                plt.xlabel('time steps', fontsize=20)
                plt.ylabel('value', fontsize=20)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)

                def draw(yi, color):
                    pData = yi.detach().cpu().numpy()
                    plt.plot(pData, color, linewidth=2.0)

                draw(output[:,idxBatch,0], 'r')
                draw(train_data[:, idxBatch, 0], 'b')
                plt.draw()
                plt.savefig('./result/output/predict_idxSeq_%d.png' % idxSeq)
                plt.close()
                fid = open("./result/output/predict_idxSeq_%d.txt" % idxSeq, 'w')
                for ff in range(output.shape[0]):
                    fid.write('%.8f\t%.8f\n' % (output[ff,idxBatch,0],train_data[ff, idxBatch, 0]))

                fid.close()