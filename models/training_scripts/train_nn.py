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

#custom imports
from cnn import CNN
from alexnet import AlexNet

class ExperimentTrain:
    criterion = None
    model = None
    optimizer = None

    def __init__(self, mounted_input_path, mounted_output_path, model_name, device, model_args,
    train_batches_count, valid_batches_count):
        self.mounted_input_path = mounted_input_path
        self.mounted_output_path = mounted_output_path
        self.model_name = model_name
        self.train_batches_count = train_batches_count
        self.valid_batches_count = valid_batches_count
        self.device = device
        self.cpu = torch.device('cpu')
        self.model_args = model_args
        self.model_details = dict()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
    
    def __save_model_details(self):
        with open(f"{self.mounted_output_path}/models/{self.model_args['model']}_model_details.json", "w+") as f:
            json.dump(self.model_details, f)

    def __get_label_values(self, label_tensor):
        class_labels = torch.max(label_tensor, 1).indices
        return class_labels

    def __load_data_and_model(self, batch_index, is_train = True):
        data_type =  'train' if is_train else 'valid'
        data_input_path = f"{self.mounted_input_path}/input"
        image_tensor_path = f"{data_input_path}/tensor_{data_type}_batch_{batch_index}_images.pt"
        label_tensor_path = f"{data_input_path}/tensor_{data_type}_batch_{batch_index}_labels.pt"

        self.batch_data = torch.load(image_tensor_path, map_location=self.device)
        self.label_tensor = torch.load(label_tensor_path, map_location=self.device)
        if self.model_args["label_dimension"] == 1:
            self.label_tensor = self.__get_label_values(self.label_tensor)
        #print(type(self.label_tensor), self.label_tensor[:5])
        #self.image_partitions.to(self.device)
        self.label_tensor.to(self.device)
        model_name = self.model_args['model']
        
        model_details_path = f"{self.mounted_output_path}/models/model_details.json"
        if os.path.exists(model_details_path):
            with open(model_details_path) as f:
                self.model_details = json.load(f)

        if self.model == None:
            model_input_path = f"{self.mounted_input_path}/models/{model_name}_model.pt"
            if model_name == 'cnn':
                self.model = CNN()
            elif model_name == 'alexnet':
                self.model = AlexNet()
            model_object = torch.load(model_input_path, map_location=self.device)
            self.model.load_state_dict(model_object["model_state"])
            self.model.to(self.device)
            #print(type(self.model))

            self.criterion = model_object["criterion"]
            #print('loss loader', self.criterion)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.model_args['lr'], momentum=self.model_args['momentum'])
            #self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.model_args['lr'])
            self.optimizer.load_state_dict(model_object["optimizer_state"])

    def __train_batch(self, batch_index, epoch_index):
        self.model.train()
        self.optimizer.zero_grad()
        model_name = self.model_args['model']
        if self.model_args["model"] == 'cnn':
            predicted_labels = self.model(self.image_tensor)
        if self.model_args["model"] == 'alexnet':
            predicted_labels = self.model(self.batch_data, self.device)
        else:
            print('\nInvalid model name')
            print('Exiting program')
        
        #self.label_tensor = self.label_tensor.to(self.device)
        predicted_labels = predicted_labels.type(torch.float).to(self.device)
        actual_labels = self.label_tensor
        actual_labels.to(self.device)
        #print('\tpreds shape', self.log_softmax(predicted_labels.squeeze(1)).shape)
        #print('\tactual labels shape', actual_labels.shape)
        #print('actual output', predicted_labels[:5])
        #print('expected outupt', self.label_tensor[:5])
        loss = self.criterion(self.log_softmax(predicted_labels.squeeze(1)), actual_labels)
        #print('\tloss', loss)
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

        if not(os.path.exists(f"{self.mounted_output_path}/internal_output")):
            os.mkdir(f"{self.mounted_output_path}/internal_output")

        if not(os.path.exists(f"{self.mounted_output_path}/models")):
            os.mkdir(f"{self.mounted_output_path}/models")
        if epoch_index == self.model_args["num_epochs"]:
            torch.save(model_object, model_output_path)
        self.__save_model_details() 
    
    def __calculate_epoch_loss(self, epoch_index):
        current_epoch_name = f"epoch_{epoch_index}"
        log_path = f"{self.mounted_output_path}/models/{model_args['model']}_model_details.json"
        with open(log_path) as f:
            model_details = json.load(f)
            self.model_details[current_epoch_name]["loss"] = model_details[current_epoch_name]["loss"] / self.train_batches_count
        self.__save_model_details()
        return self.model_details[current_epoch_name]["loss"]

    def __evaluate_model(self):
        log_path = f"{self.mounted_output_path}/models/{model_args['model']}_model_details.json"
        accu_val, total_acc, total_count = 0, 0, 0
        self.model.eval()
        with torch.no_grad():
            for i in range(self.valid_batches_count):
                self.__load_data_and_model(i, False)
                if self.model_name == 'cnn':
                    predicted_probs = self.model(self.image_tensor)
                elif self.model_name == 'alexnet':
                    predicted_probs = self.model(self.batch_data, self.device)
                print(predicted_probs[:5], predicted_probs.shape)
                predicted_labels = torch.max(predicted_probs, 1).indices
                #print('predicted lbls', predicted_labels, predicted_labels.shape)
                #print('label tensor shape', self.label_tensor.shape)
                if self.model_args['label_dimension'] == 3:
                    actual_labels = torch.max(self.label_tensor, 1).indices
                else:
                    actual_labels = self.label_tensor
                #print('actual lbls', actual_labels, actual_labels.shape)
                total_acc += (predicted_labels == actual_labels).sum().item()
                total_count += actual_labels.size(0)
            accu_val = total_acc / total_count
            with open(log_path) as f:
                model_details = json.load(f)
                self.model_details["final_accuracy"] = accu_val
        self.__save_model_details()
        return accu_val

    def start_experiment(self):
        for epoch in range(1, self.model_args["num_epochs"] + 1):
            mlflow.log_metric('\nRunning epoch', epoch)
            print('\nRunning epoch', epoch)
            self.model_details[f"epoch_{epoch}"] = dict()
            for i in range(self.train_batches_count):
                self.__load_data_and_model(i)
                self.__train_batch(i, epoch)
            loss = self.__calculate_epoch_loss(epoch)
            mlflow.log_metric(f'Loss after epoch {epoch}:', epoch)
            print(f'\tLoss after epoch {epoch}:', loss)
        accu_val = self.__evaluate_model()
        mlflow.log_metric(f'Accuracy: ', epoch)
        print(f'\nFinal Accuracy:', accu_val)

            
if __name__ == "__main__":

    mounted_input_path = sys.argv[1]
    mounted_output_path = sys.argv[2]
    model_name = sys.argv[3]
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
