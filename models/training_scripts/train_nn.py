import sys
from pandas.core.indexing import is_label_like
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import os
import json
import numpy as np
from torch.autograd import Variable
import numpy as np
import mlflow

class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x



class ExperimentTrain:

    def __init__(self, mounted_input_path, mounted_output_path, model_name, device, model_args,
    train_batches_count, valid_batches_count):
        self.mounted_input_path = mounted_input_path
        self.mounted_output_path = mounted_output_path
        self.model_name = model_name
        self.train_batches_count = train_batches_count
        self.valid_batches_count = valid_batches_count
        self.device = device
        self.model_args = model_args
        self.model_details = dict()
    
    def __save_model_details(self):
        with open(f"{self.mounted_output_path}/models/{self.model_args['model']}_model_details.json", "w+") as f:
            json.dump(self.model_details, f)

    def __load_data_and_model(self, batch_index, epoch_index, is_train = True):
        data_type =  'train' if is_train else 'valid'
        data_input_path = f"{self.mounted_input_path}/input"
        image_tensor_path = f"{data_input_path}/tensor_{data_type}_batch_{batch_index}_images.pt"
        label_tensor_path = f"{data_input_path}/tensor_{data_type}_batch_{batch_index}_labels.pt"

        self.image_tensor = torch.load(image_tensor_path, map_location=self.device)
        self.label_tensor = torch.load(label_tensor_path, map_location=self.device)
        self.image_tensor.to(self.device)
        self.label_tensor.to(self.device)
        model_name = self.model_args['model']
        
        if epoch_index == 1 and batch_index == 0:
            model_input_path = f"{self.mounted_input_path}/models/{model_name}_model.pt"
        else:
            model_input_path = f"{self.mounted_output_path}/internal_output/{model_name}_model.pt"
        model_details_path = f"{self.mounted_output_path}/models/model_details.json"
        if os.path.exists(model_details_path):
            with open(model_details_path) as f:
                self.model_details = json.load(f)

        self.model = CNN()
        model_object = torch.load(model_input_path, map_location=self.device)
        self.model.load_state_dict(model_object["model_state"])
        self.model.to(self.device)

        self.criterion = model_object["criterion"]
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.model_args["lr"], momentum = self.model_args["momentum"])
        self.optimizer.load_state_dict(model_object["optimizer_state"])

    def __train_batch(self, batch_index, epoch_index):
        self.model.train()
        self.optimizer.zero_grad()
        model_name = self.model_args['model']
        if self.model_args["model"] == 'cnn':
            predicted_labels = self.model(self.image_tensor)
        else:
            print('\nInvalid model name')
            print('Exiting program')
        
        #self.label_tensor = self.label_tensor.to(self.device)
        predicted_labels = predicted_labels.type(torch.float).to(self.device)
        actual_labels = self.label_tensor.type(torch.LongTensor).to(self.device)
        print('actual output', predicted_labels[:5])
        print('expected outupt', self.label_tensor[:5])
        loss = self.criterion(predicted_labels, actual_labels)
        #print('loss', loss)
        if batch_index == 0:
            self.model_details[f"epoch_{epoch_index}"]["loss"] = loss.item()
        else:
            self.model_details[f"epoch_{epoch_index}"]["loss"] += loss.item()
        loss.backward()
        self.optimizer.step()
        model_object = {
            'model_state': self.model.state_dict(),
            'criterion': self.criterion,
            'optimizer_state': self.optimizer.state_dict()
        }


        if epoch_index == self.model_args["num_epochs"]:
            model_output_path = f"{self.mounted_output_path}/models/{model_name}_model.pt"
        else:
            model_output_path = f"{self.mounted_output_path}/internal_output/{model_name}_model.pt"

        if not(os.path.exists(f"{self.mounted_output_path}/internal_output")):
            os.mkdir(f"{self.mounted_output_path}/internal_output")

        if not(os.path.exists(f"{self.mounted_output_path}/models")):
            os.mkdir(f"{self.mounted_output_path}/models")
        torch.save(model_object, model_output_path)
        self.__save_model_details() 

    def __evaluate_model(self, epoch_index):
        mlflow.log_metric('Running validation set for epoch: ', epoch_index)
        log_path = f"{self.mounted_output_path}/models/{model_args['model']}_model_details.json"
        current_epoch_name = f"epoch_{epoch_index}"
        accu_val, total_acc, total_count = 0, 0, 0
        self.model.eval()
        with torch.no_grad():
            for i in range(self.valid_batches_count):
                self.__load_data_and_model(i, epoch_index, False)
                predicted_probs = self.model(self.image_tensor)
                predicted_labels = torch.max(predicted_probs, 1).indices
                actual_labels = torch.max(self.label_tensor, 1).indices
                total_acc += (predicted_labels == actual_labels).sum().item()
                total_count += self.label_tensor.size(0)
            accu_val = total_acc / total_count
            with open(log_path) as f:
                model_details = json.load(f)
                current_epoch_name = f"epoch_{epoch_index}"
                self.model_details[current_epoch_name]["loss"] = model_details[current_epoch_name]["loss"] / self.train_batches_count
                self.model_details[current_epoch_name]["accuracy"] = accu_val
        self.__save_model_details()
        return accu_val, self.model_details[current_epoch_name]["loss"]

    def start_experiment(self):
        for epoch in range(1, self.model_args["num_epochs"] + 1):
            mlflow.log_metric('\nRunning epoch', epoch)
            print('\nRunning epoch', epoch)
            self.model_details[f"epoch_{epoch}"] = dict()
            for i in range(self.train_batches_count):
                self.__load_data_and_model(i, epoch)
                self.__train_batch(i, epoch)
            accu_val, loss = self.__evaluate_model(epoch)
            mlflow.log_metric(f'Accuracy after epoch {epoch}', epoch)
            print(f'\tAccuracy after epoch {epoch}:', accu_val)
            mlflow.log_metric(f'Loss after epoch {epoch}:', epoch)
            print(f'\tLoss after epoch {epoch}:', loss)
            
if __name__ == "__main__":

    mounted_input_path = sys.argv[1]
    mounted_output_path = sys.argv[2]
    model_name = sys.argv[3]
    #is_first_batch = True if sys.argv[5] == "True" else False
    #is_last_batch = True if sys.argv[6] == "True" else False
    model_args_string = sys.argv[4]
    train_batches_count = int(sys.argv[5])
    valid_batches_count = int(sys.argv[6])

    model_args = json.loads(model_args_string)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        exit()
    exp = ExperimentTrain(mounted_input_path, mounted_output_path, model_name, device, model_args, train_batches_count, valid_batches_count)
    exp.start_experiment()
