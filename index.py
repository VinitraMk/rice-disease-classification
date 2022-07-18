import os
from sklearn.utils import shuffle
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import DataLoader
from azureml.core import Workspace, Experiment as AzExperiment, Environment, ScriptRunConfig
from azureml.core.dataset import Dataset
from azureml.data import OutputFileDatasetConfig
from datetime import date
from azure.storage.blob import BlobServiceClient, BlobClient
import pandas as pd
import warnings
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from azureml.core.authentication import InteractiveLoginAuthentication
import mlflow
import shutil

#custom imports
from helper.preprocessor import Preprocessor
from helper.augmenter import Augmenter
from constants.types.model_enums import Model
from models.cnn import CNN
from models.alexnet import AlexNet
from experiments.RiceDataset import RiceDataset 
from helper.utils import get_config, get_filename, get_model_params, get_preproc_params, init_weights, read_json, save_fig, save_model, save_tensor, get_target_cols, get_azure_config

class Index:
    model_args = None
    preproc_args = None
    config = None
    preprocessor = None
    train_dataset = None
    valid_dataset = None
    test_dataset = None
    model_details = dict()
    valid_batches_count = 0
    train_batches_count = 0
    test_batches_count = 0
    current_model_obj = None
    model_state = None
    blob_service_client = None

    def __init__(self):
        if not(os.getenv('ROOT_DIR')):
            os.environ['ROOT_DIR'] = os.getcwd()

    def start_program(self):
        self.__define_args()
        self.__preprocess_data()
        self.__prepare_datasets()
        self.__make_model()
        self.__make_azure_resources()
        self.__build_data_and_upload()
        self.__start_training(self.test_loader)

    def __get_labels(self, predicted_labels):
        actual_labels = []
        for label in predicted_labels:
            actual_labels.append(self.idx_to_class[label])
        return actual_labels
        
    def __define_args(self):
        print('\nDefining configs')
        self.model_args = get_model_params()
        self.config = get_config()
        self.azure_config = get_azure_config()
        self.preproc_args = get_preproc_params()
        STORAGE_ACCOUNT_URL = 'https://mlintro1651836008.blob.core.windows.net/'
        self.blob_service_client = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=os.environ["AZURE_STORAGE_CONNECTIONKEY"])

    def __preprocess_data(self):
        print('\nSetting up preprocessor')
        self.augmenter = Augmenter()
        self.preprocessor = Preprocessor()
        class_to_idx, idx_to_class = self.preprocessor.get_class_mappings()
        self.idx_to_class = idx_to_class
        train, test, valid = self.preprocessor.get_data_paths()
        train_transforms, test_transforms = self.augmenter.get_transforms()
        self.train_dataset = RiceDataset(train, class_to_idx, train_transforms)
        self.valid_dataset = RiceDataset(valid, class_to_idx, test_transforms)
        self.test_dataset = RiceDataset(test, class_to_idx, test_transforms)
        #self.preprocessor.start_preprocessing(self.train_dataset)
    
    def __prepare_datasets(self):
        print('\nPreparing dataset')
        self.train_loader = DataLoader(self.train_dataset, batch_size = self.model_args["batch_size"], shuffle = self.model_args["shuffle_data"], collate_fn=self.preprocessor.collate_batch)
        self.train_batches_count = int(len(self.train_dataset) / self.model_args["batch_size"]) + 1
        self.valid_loader = DataLoader(self.valid_dataset, batch_size = self.model_args["batch_size"], shuffle = self.model_args["shuffle_data"], collate_fn=self.preprocessor.collate_batch)
        self.valid_batches_count = int(len(self.valid_dataset) / self.model_args["batch_size"]) + 1
        self.test_loader = DataLoader(self.test_dataset, batch_size = self.model_args["batch_size"], shuffle = self.model_args["shuffle_data"], collate_fn=self.preprocessor.collate_batch)
        self.test_batches_count = int(len(self.test_dataset) / self.model_args["batch_size"]) + 1

    def __make_model(self):
        print('\nMaking model')
        if (self.model_args["model"] == Model.CNN):
            self.model = CNN()
        elif (self.model_args["model"] == Model.ALEXNET):
            self.model = AlexNet()
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr = self.model_args['lr'], momentum = self.model_args["momentum"])
        model_obj = {
            'model_state': self.model.state_dict(),
            'criterion': criterion,
            'optimizer_state': optimizer.state_dict()
        }
        #print('model state', self.model.state_dict())
        model_path = f"{self.config['processed_io_path']}\\models"
        print('\tSaving model')
        save_model(model_obj, model_path, self.model_args["model"], True)
    
    def __make_azure_resources(self):
        print('\nConfiguring Azure Resources')
        # Configuring workspace
        print('\tConfiguring Workspace')
        today = date.today()
        todaystring = today.strftime("%d-%m-%Y")
        interactive_auth = InteractiveLoginAuthentication(tenant_id=self.azure_config["tenant_id"])
        self.azws = Workspace.from_config(self.config["resource_config_path"], auth=interactive_auth)
        mlflow.set_tracking_uri(self.azws.get_mlflow_tracking_uri())
        
        print('\tConfiguring Environment')
        self.azenv = Environment.get(workspace=self.azws, name="vinazureml-env")
        self.azexp = AzExperiment(workspace=self.azws, name=f'{todaystring}-experiments')
        print('\tGetting default blob store')
        self.def_blob_store = self.azws.get_default_datastore()

    def __copy_model_chkpoint(self):
        train_scripts_path = f"{self.config['processed_io_path']}/training_scripts"
        local_script_path = './models/training_scripts/train_nn.py'
        local_model_file = f'./models/{self.model_args["model"]}.py'
        shutil.copy(local_script_path, f'{train_scripts_path}/train_nn.py')
        shutil.copy(local_model_file, f'{train_scripts_path}/{self.model_args["model"]}.py')
        

    def __merge_output_with_training(self):
        new_train_dataset = torch.utils.data.ConcatDataset([self.split_train_dataset, self.new_dataset])
        self.train_loader = DataLoader(new_train_dataset, batch_size = self.model_args["batch_size"], shuffle = True, collate_fn=self.preprocessor.collate_batch)
        
    def __train_model_in_azure(self, is_first_batch = False, is_last_batch = False):
        print('\tBuilding config for experiment run')
        input_data = Dataset.File.from_files(path=(self.def_blob_store, '/input'))
        input_data = input_data.as_named_input('input').as_mount()
        output = OutputFileDatasetConfig(destination=(self.def_blob_store, '/output'))
        model_args_string = json.dumps(self.model_args)
        config = ScriptRunConfig(
            source_directory='./processed_io/training_scripts',
            script='train_nn.py',
            arguments=[input_data, output, self.model_args["model"], model_args_string,
            self.train_batches_count, self.valid_batches_count],
            compute_target="mikasa",
            environment=self.azenv
        )

        run = self.azexp.submit(config)
        run.wait_for_completion(show_output = False, raise_on_error = True)

    def __upload_batch_data(self):
        self.__copy_model_chkpoint()
        print('\tUploading scripts and data')
        if (self.model_args['refresh_data']):
            print('\t\tUploading data to blob storage')
            self.def_blob_store.upload(src_dir='./processed_io/input', target_path="input/input", overwrite=True, show_progress = True)
        if (self.model_args['refresh_model']):
            print('\t\tUploading model to blob storage')
            self.def_blob_store.upload(src_dir='./processed_io/models', target_path="input/models", overwrite=True, show_progress = False)

    def __download_output(self):
        config = get_config()
        MAIN_OUTPUT_CONTAINER = 'azureml-blobstore-31aeaa24-564c-4aa8-bdf4-fc4b5707bd1b/output'
        MODEL_CONTAINER = f'{MAIN_OUTPUT_CONTAINER}/models'
        #MODEL_DETAILS_CONTAINER = f'{MAIN_OUTPUT_CONTAINER}/internal_output'
        LOCAL_MODELDET_PATH = f"{config['experimental_output_path']}\\{self.model_args['model']}_model_details.json"
        LOCAL_MODEL_PATH = f"{config['internal_output_path']}\\{self.model_args['model']}_model.pt"
        blob_client_model = self.blob_service_client.get_blob_client(MODEL_CONTAINER, f'{self.model_args["model"]}_model.pt', snapshot=None)
        blob_client_modeldet = self.blob_service_client.get_blob_client(MODEL_CONTAINER, f'{self.model_args["model"]}_model_details.json', snapshot=None)
        print('\tDownloading model')
        self.__download_blob(LOCAL_MODEL_PATH, blob_client_model)
        print('\tDownloading model details')
        self.__download_blob(LOCAL_MODELDET_PATH, blob_client_modeldet)
        blob_client_modeldet.delete_blob()
        with open(LOCAL_MODELDET_PATH) as fp:
            self.model_details = json.load(fp)


    def __download_blob(self, local_filename, blob_client_instance):
        with open(local_filename, "wb") as my_blob:
            blob_data = blob_client_instance.download_blob()
            blob_data.readinto(my_blob)

    def __remove_old_inputs(self):
        print('\nDeleting old inputs')
        ROOT_CONTAINER = 'azureml-blobstore-31aeaa24-564c-4aa8-bdf4-fc4b5707bd1b'
        FOLDER_CONTAINER = f"{ROOT_CONTAINER}/input/input"
        LOCAL_PATH = self.config["internal_output_path"]
        blob_container_client = self.blob_service_client.get_container_client(ROOT_CONTAINER)
        all_blobs = blob_container_client.list_blobs(name_starts_with='input/input/tensor')
        for b in all_blobs:
            blob_name = b.name.split('/')[2]
            blob_client_model = self.blob_service_client.get_blob_client(FOLDER_CONTAINER, blob_name, snapshot=None)
            print(f'\tDeleting tensor {blob_name}')
            blob_client_model.delete_blob()
            local_blob_path = f"{self.config['internal_output_path']}\\{blob_name}"
            if os.path.exists(local_blob_path):
                os.remove(local_blob_path)

    def __plot_loss_accuracy(self, filename):
        #plot for loss and accuracy
        print("\nPlotting loss and accuracy for all epochs")
        x = np.arange(1, self.model_args["num_epochs"]+1)
        y1 = []
        avg_acc, avg_loss = 0, 0
        for i in range(1, self.model_args["num_epochs"]+1):
            y1.append(self.model_details[f"epoch_{i}"]["loss"])
            avg_loss+=self.model_details[f"epoch_{i}"]["loss"]
        avg_acc=self.model_details["final_accuracy"]
        avg_loss/=self.model_args["num_epochs"]
        print("\tFinal Accuracy:", avg_acc * 100)
        print("\tAverage loss:", avg_loss, "\n")
        model_logs = {
            'final_accuracy': avg_acc,
            'average_loss': avg_loss,
            'model_filename': filename,
            'model_params': json.dumps(self.model_args),
            'preprocessing_params': json.dumps(self.preproc_args)
        }
        exp_out_path = self.config['experimental_output_path']
        log_output_path = f"{exp_out_path}\\{filename}.json"
        with open(log_output_path, 'w+') as f:
            json.dump(model_logs, f)

        plt.plot(x, y1, color="red", marker='o', linewidth=1, markersize=5)
        save_fig("loss_accuracy_plot", plt)
        plt.clf()

    def __evaluate_model(self):
        print('\nGetting test data predictions')
        internal_path = self.config['internal_output_path']
        model_name = self.model_args['model']
        out_path = self.config['output_path']
        model_state_path = f"{internal_path}\\{model_name}_model.pt"
        model_object = torch.load(model_state_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(model_object['model_state'])
        self.model.eval()
        target_cols = get_target_cols()
        results_df = pd.DataFrame([], columns=['Image_id'] + self.model_args['output_labels'])
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if self.model_args["model"] == Model.CNN:
                    predicted_probs = self.model(batch[0])
                elif self.model_args["model"] == Model.ALEXNET:
                    device = torch.device('cpu')
                    predicted_probs = self.model(batch[0], device)
                    #predicted_labels = torch.max(predicted_probs, 1).indices
                #actual_labels = batch[0].type(torch.float)
                get_cols = lambda x: { 'blast': x[0].item(), 'brown': x[1].item(), 'healthy': x[2].item()}
                preds = []
                for x in predicted_probs:
                    #print('x', x)
                    preds.append(get_cols(x.squeeze(0)))
                preds = preds + preds
                predicted_df = pd.DataFrame(preds, columns=self.model_args['output_labels'])
                rgn_ids = list(batch[2])
                other_ids = [x.replace('_rgn','') for x in rgn_ids]
                predicted_ids = pd.DataFrame(rgn_ids + other_ids, columns=['Image_id'])
                predicted_df = pd.concat([predicted_ids, predicted_df], axis = 1)
                results_df = pd.concat([results_df, predicted_df])
            filename = get_filename(model_name)
            csv_output_path = f"{out_path}\\{filename}.csv"
            results_df.to_csv(csv_output_path, index = False)
            self.__plot_loss_accuracy(filename)

    def __build_data_and_upload(self):
        if self.model_args['refresh_data'] or self.model_args['refresh_model']:
            print('\nBuild data and upload')
        if self.model_args['refresh_data']:
            self.__remove_old_inputs()
            print('\tBuilding all input tensors for training')
            for i, batch  in enumerate(self.train_loader):
                print(f'\tBuilding batch {i} for training')
                save_tensor(batch[0], f'train_batch_{i}_images')
                save_tensor(batch[1], f'train_batch_{i}_labels')
            for i, batch in enumerate(self.valid_loader):
                print(f'\tBuilding batch {i} for validation')
                save_tensor(batch[0], f'valid_batch_{i}_images')
                save_tensor(batch[1], f'valid_batch_{i}_labels')
        if self.model_args['refresh_data'] or self.model_args['refresh_model']:
            self.__upload_batch_data()

    def __start_training(self, test_loader, is_pseudo_test = False):
        print("\nStarting training with training data")
        self.__train_model_in_azure()
        print('\nDownloading output')
        self.__download_output()
        self.__evaluate_model()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    index = Index()
    index.start_program()