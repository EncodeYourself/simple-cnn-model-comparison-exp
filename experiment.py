import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision.datasets import MNIST
from torch.utils.data.sampler import SubsetRandomSampler
from seaborn import heatmap

#%%


def unpack_mnist(dataset):
    return dataset.data, dataset.targets


#%%
class Dataset_with_indx(Dataset):
    def __init__(self, data, labels):
        self.data = data           
        self.labels = labels
        self.length = data.shape[0]
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, indx):
        return (indx, (self.data[indx]/255).unsqueeze(0), self.labels[indx])
        
    
    
class cnn_network(nn.Module):
    def __init__(self, n_classes = 10, fcl_n = 784):
        super(cnn_network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, 
            out_channels= 8, kernel_size=(3,3), 
            stride = (1,1), padding = (1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride = (2, 2))
        self.conv2 = nn.Conv2d(in_channels = 8, 
            out_channels= 16, kernel_size=(3,3), 
            stride = (1,1), padding = (1, 1))
        self.fcl = nn.Linear(fcl_n, n_classes)
    
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fcl(x)
        
        return x

    
class cnn_network2(nn.Module):
    def __init__(self, n_classes = 10, fcl_n = 784):
        super(cnn_network2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, 
            out_channels= 8, kernel_size=(3,3), 
            stride = (1,1), padding = (1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride = (2, 2))
        self.conv2 = nn.Conv2d(in_channels = 8, 
            out_channels= 16, kernel_size=(3,3), 
            stride = (1,1), padding = (1, 1))
        self.fcl = nn.Linear(fcl_n, n_classes)
        self.relu = nn.ReLU()
    
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fcl(x)
        
        return x

class cnn_network3(nn.Module):
    def __init__(self, n_classes = 10, fcl_n = 784):
        super(cnn_network3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, 
            out_channels= 8, kernel_size=(3,3), 
            stride = (1,1), padding = (1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride = (2, 2))
        self.conv2 = nn.Conv2d(in_channels = 8, 
            out_channels= 16, kernel_size=(3,3), 
            stride = (1,1), padding = (1, 1))
        self.fcl = nn.Linear(fcl_n, n_classes)
        self.relu = nn.ReLU()
    
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fcl(x)
        
        return x
        
    
    
    
class model_controller:
    def __init__(self, model, train_dataset, test_dataset, epochs, batch_size = 32, lr = .001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_dataset = train_dataset
        shape = self.train_dataset[0][1].shape[1:]
        self.n_classes = len(torch.unique(train_dataset[:][2]))
        torch.manual_seed(42)
        self.model = model(n_classes = self.n_classes, 
                           fcl_n= self.calc_fcl_n_num(shape)).to(self.device)
        self.train_full_length = len(self.train_dataset)
        self.epochs = epochs
        self.current_epoch = 0
        self.batch_size = batch_size
        self.test_loader = DataLoader(
            test_dataset, batch_size = self.batch_size, shuffle = True)
        self.test_loader_length = len(self.test_loader)
        
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)
        
        self.last_run_train_loss = np.empty((self.epochs))
        self.last_run_test_loss = np.empty(self.epochs)
        self.last_run_var = None
        self.last_mode = None
        self.conf_matrix = None
        self.last_acc = 0
        self.time = 0

    
    def calc_fcl_n_num(self, shape):
        
        '''Расчёт размера ввода для полносвязного слоя в конце сети''' 
        
        shape = np.array(shape)
        calc_n = lambda shape: np.floor(np.floor(shape/2)/2)
        result = int(np.prod(calc_n(shape))*16)
        
        return result
        
    
    def evaluate(self, selective = False):
        conf_matrix = np.zeros((self.n_classes, self.n_classes))
        correct = 0
        total   = 0
        
        with torch.no_grad():
            loss_values = np.empty(self.test_loader_length)
            
            for batch,(index, data, labels) in enumerate(tqdm(self.test_loader)):
                labels = labels.to(self.device)
                output = self.model(data.to(self.device))
                inst_loss = self.criterion(output, labels)
                batch_loss = torch.mean(inst_loss)
                loss_values[batch] = batch_loss
            
                _, pred = torch.max(output, 1)
                
                correct += sum(labels == pred)
                total += len(labels)
                
                for idx in range(len(labels)):
                    conf_matrix[labels[idx]][pred[idx]] += 1
            
            mean_loss = np.mean(loss_values)
            self.last_run_test_loss[self.current_epoch] = mean_loss
            print(mean_loss, self.current_epoch, self.last_mode )
        
        self.conf_matrix = conf_matrix
        self.last_acc = (correct / total).detach().cpu().numpy()
            
        
    def train(self):
        self.last_mode = 'Conventional'
        self.last_run_var = np.empty((self.epochs, self.train_full_length))
        
        self.model.train()
        loader = DataLoader(self.train_dataset, 
                            batch_size=self.batch_size, shuffle = True)
        loss_values = np.empty(len(loader))
        start_time = time.time()
        torch.manual_seed(42)
        for epoch in range(self.epochs):
            loss_values = torch.tensor([], requires_grad= True).to(self.device)
            self.optimizer.zero_grad()
            
            for batch_idx, (indx, data, labels) in enumerate(tqdm(loader)):
              output = self.model(data.to(self.device))
              inst_loss = self.criterion(output, labels.to(self.device))
              loss_values = torch.cat((loss_values, inst_loss))
              
            
            self.last_run_var[self.current_epoch] = loss_values.detach().cpu().numpy()
            loss_values = torch.mean(loss_values)
             
            self.last_run_train_loss[self.current_epoch] = loss_values.detach().cpu().numpy()
            loss_values.backward()  
            self.optimizer.step()
            
            self.evaluate(selective = False)
            self.current_epoch += 1
        
        complete_time = time.time()
        self.time = (complete_time - start_time) / 60
        self.current_epoch = 0
    
    
    def selective_train(self, proportion = .8):
        assert self.epochs % 4 == 0, 'epochs must be divisible by 4'
        self.last_mode = 'Selective'
        self.last_run_var = []
                            
        self.model.train()
        self.optimizer.zero_grad()
        
        start_time = time.time()
        cycles = int(self.epochs / 4)
        for cycle in range(cycles):
            limit = int(self.train_full_length * proportion)
            indices = None
            
            for iteration in range(4):
                loss_values = torch.tensor([], requires_grad= True).to(self.device)
                self.optimizer.zero_grad()
                if indices is None:
                    loader = DataLoader(self.train_dataset, 
                            batch_size = self.batch_size, 
                            shuffle = True)
                else:
                    loader = DataLoader(self.train_dataset, 
                            batch_size = self.batch_size, 
                            sampler = SubsetRandomSampler(indices))
                    
                for batch_idx, (index, data, labels) in enumerate(tqdm(loader)):
                    output = self.model(data.to(self.device))
                    inst_loss = self.criterion(output, labels.to(self.device))
                    loss_values = torch.cat((loss_values, inst_loss))
                    
                loss_values = loss_values.sort(descending = True)[0][:limit]
                indices = loss_values.sort(descending = True)[1][:limit]
                self.last_run_var.append(loss_values.detach().cpu().numpy().T)
                loss_values = torch.mean(loss_values)
                self.last_run_train_loss[self.current_epoch] = loss_values.detach().cpu().numpy()
                loss_values.backward()
            
                self.optimizer.step()
                
                limit = int(limit * proportion)
                self.evaluate(selective = True)
                
                self.current_epoch += 1
        complete_time = time.time()
        self.time = (complete_time - start_time) / 60

def loss_visualization(model1, model2):
    plt.rcParams["figure.figsize"] = [16,9]
    figure, axis = plt.subplots(3, 1)
    
    x = range(len(model1.last_run_train_loss))
    axis[0].set_title('loss vs epoch')
    axis[0].plot(x, model1.last_run_train_loss, label = f'{model1.last_mode} train')
    axis[0].plot(x, model1.last_run_test_loss, label = f'{model1.last_mode} test')
    axis[0].plot(x, model2.last_run_train_loss, label = f'{model2.last_mode} train')
    axis[0].plot(x, model2.last_run_test_loss, label = f'{model2.last_mode} test')
    axis[0].set_xticks(x)
    axis[0].legend()
    
    axis[1].set_title(f'{model1.last_mode} train loss distributions')
    axis[1].boxplot(model1.last_run_var.T)
    
    axis[2].set_title(f'{model2.last_mode} train loss distributions')
    axis[2].boxplot(model2.last_run_var.T)
    
    plt.show()

def confusion_matrices(model1, model2):
    
    plt.rcParams["figure.figsize"] = [30,10]
    figure, (ax1, ax2) = plt.subplots(1, 2)
    acc1 = str(model1.last_acc)[:4]
    ax1.set_title(f'{model1.last_mode} model, Accuracy == {acc1}%')
    heatmap(model1.conf_matrix, annot=True, fmt='g', cbar = False, ax = ax1)
    ax1.set_xlabel('Predicted');
    ax1.set_ylabel('True');
    
    acc2 = str(model2.last_acc)[:4]
    ax2.set_title(f'{model2.last_mode} model, Accuracy == {acc2}%')
    heatmap(model2.conf_matrix, annot=True, fmt='g', ax = ax2)
    ax2.set_xlabel('Predicted');
    ax2.set_ylabel('True');
    
    plt.show()
      
#%%


root = os.getcwd()
mnist = True

train_data = MNIST(root, train = True, download = True)
test_data = MNIST(root, train = False, download = True)


train_dataset = Dataset_with_indx(
    *unpack_mnist(train_data))
test_dataset = Dataset_with_indx(
    *unpack_mnist(test_data))



#%%
MC = model_controller(cnn_network, train_dataset, test_dataset, 90)
MC2 = model_controller(cnn_network2, train_dataset, test_dataset, 90)
#MC3 = model_controller(cnn_network3, train_dataset, test_dataset, 4)
MC.train()
MC2.train()

loss_visualization(MC, MC2)
#%%
