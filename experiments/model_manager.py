import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
import torch.nn.functional as F
import wandb
import logging
from .models import *
from scipy import stats
import math
from .loss.loss_func import NLLSurvLoss
from sksurv.metrics import concordance_index_censored
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_auc_score, confusion_matrix,f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
from .utils import accuracy_confusionMatrix_plot, kfold_results_merge, move_to_device
# from .metrics.loss_func import NLLSurvLoss
# from .scheduler import *

DEBUG_BATCHES = 100000

class ModelManager():
    def __init__(self,
                 config
                 ):
        self.config = config
        self.device = config.model.device
        self.real_batch_size = config.data_loader.real_batch_size
        self.NUM_ACCUMULATION_STEPS = self.real_batch_size//config.data_loader.batch_size

        self.net = self.__build__()
        self.net.to(self.device)

        if self.config.data_loader.batch_size <= self.config.data_loader.real_batch_size:
            reduction = 'mean'
        else:
            reduction = 'none'
            self.num_replicas = self.config.data_loader.batch_size // self.config.data_loader.real_batch_size
        self.loss_function = self.__getLossFunction__()
        self.parameters_to_optimize = self.net.parameters()
        self.optimizer = self.__getOptimizer__()
        self.scheduler = self.__getScheduler__()
        try:
            self.AEM_lamda = self.config.trainer.AEM_lamda
        except:
            self.AEM_lamda = 0
        try:
            self.clip_grad_norm_max_norm = self.config.trainer.clip_grad_norm_max_norm
        except:
            self.clip_grad_norm_max_norm = None
        wandb.watch(self.net, log_freq=100)

    def __build__(self):
        if self.config.model.name == "BaselineModel":
            model = BaselineModel(**self.config.model.kwargs)
            return model
        if self.config.model.name == "ABMIL":
            model = ABMIL(**self.config.model.kwargs)
            return model
        if self.config.model.name == "ABMIL_Tangle":
            model = ABMIL_Tangle(**self.config.model.kwargs)
            return model
        if self.config.model.name == "ABMIL_Tweak":
            model = ABMIL_Tweak(**self.config.model.kwargs)
            return model
        if self.config.model.name == "MaxPooling":
            model = MaxPooling(**self.config.model.kwargs)
            return model
        if self.config.model.name == "Custom":
            model = Custom(**self.config.model.kwargs)
            return model
        if self.config.model.name == "Custom_Multimodal":
            model = Custom_Multimodal(**self.config.model.kwargs)
            return model
        if self.config.model.name == "TransMIL":
            model = TransMIL(**self.config.model.kwargs)
            return model
        if self.config.model.name == "MIL_Attention_FC_surv":
            model = MIL_Attention_FC_surv(**self.config.model.kwargs)
            return model
        if self.config.model.name == "LinearProbe":
            model = LinearProbe(**self.config.model.kwargs)
            return model
        else:
            raise Exception(f"{self.config.model.name} is not supported!")
        
    def __getLossFunction__(self):
        if self.config.loss.name == "NLLSurvLoss":
            return NLLSurvLoss(**self.config.loss.kwargs)
        elif self.config.loss.name == "MSELoss":
            return nn.MSELoss(**self.config.loss.kwargs)
        elif self.config.loss.name == "CrossEntropyLoss":
            if 'weight' in self.config.loss.kwargs:
                self.config.loss.kwargs['weight'] = torch.tensor(self.config.loss.kwargs['weight']).float().to(self.device) 
            return nn.CrossEntropyLoss(**self.config.loss.kwargs)
        else:
            raise Exception(f"{self.config.loss.name} is not supported!")

    def __getOptimizer__(self):
        if self.config.optimizer.name == "Adam":
            return optim.Adam(self.parameters_to_optimize,
                              lr=self.config.optimizer.learning_rate,
                              weight_decay=self.config.optimizer.weight_decay,
                            #   eps=1e-07,
                              )
        if self.config.optimizer.name == "AdamW":
            return optim.AdamW(self.parameters_to_optimize,
                              lr=self.config.optimizer.learning_rate,
                              weight_decay=self.config.optimizer.weight_decay,
                            #   eps=1e-07,
                              )        
        if self.config.optimizer.name == "RAdam":
            return optim.RAdam(self.parameters_to_optimize,
                              lr=self.config.optimizer.learning_rate,
                              weight_decay=self.config.optimizer.weight_decay,
                            #   eps=1e-07,
                              )        
        elif self.config.optimizer.name == 'SGD':
            return optim.SGD(params=self.parameters_to_optimize, lr=self.config.optimizer.learning_rate, 
                             weight_decay=self.config.optimizer.weight_decay, 
                             momentum=self.config.optimizer.momentum)        
        else:
            raise Exception(f"{self.config.optimizer.name} is not supported!")

    def __getScheduler__(self):
        if self.config.scheduler.name == "OneCycleLR":
            self.kwargs = {
                'max_lr': self.config.optimizer.learning_rate,
                'total_steps': self.config.scheduler.steps_per_epoch*self.config.trainer.epochs,
                'steps_per_epoch': self.config.scheduler.steps_per_epoch,
                'epochs': self.config.trainer.epochs,
                'pct_start': self.config.scheduler.pct_start,
                'final_div_factor': 1e8,  
                'anneal_strategy': 'linear',
                #'three_phase': False, # True 
            }
            return torch.optim.lr_scheduler.OneCycleLR(self.optimizer, **self.kwargs)
        elif self.config.scheduler.name == 'MultiStepLR':
            self.kwargs = {
                    'milestones':  self.config.scheduler.milestones,
                    'gamma': self.config.scheduler.gamma,
                    }
            return torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **self.kwargs)        
        elif self.config.scheduler.name == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.trainer.epochs, eta_min=1e-6, last_epoch=-1, verbose='deprecated')
        else:    
            raise Exception(f"{self.config.scheduler.name} is not supported!")

    def __reload_net__(self, path, device='cuda'):
        if device == 'cuda':
            logging.info(f'\nRestoring model weigths from: {path}')
            self.net = torch.load(path)
        else:
            logging.info(f'\nRestoring model weigths from: {path}')
            self.net = torch.load(path, map_location=torch.device('cpu'))

    def load_checkpoint(self, path, device='cpu'):
        if device == 'cuda':
            weights = torch.load(path)["state_dict"] 
            self.net.load_state_dict(weights, strict=False) 
        else:
            weights   =  torch.load(path, map_location=torch.device(device))["state_dict"] 
            self.net.load_state_dict(weights, strict=False) 

    def calculate_risk(self, h):
        r"""
        Take the logits of the model and calculate the risk for the patient 
        
        Args: 
            - h : torch.Tensor 
        
        Returns:
            - risk : torch.Tensor 
        
        """
        hazards = torch.sigmoid(h)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        return risk, survival.detach().cpu().numpy()
    
    def initialize_metrics_dict(self, task_type="Survival"):
        log_dict = {}
        if task_type == "Survival":
            log_dict["all_risk_scores"] = []
            log_dict["all_censorships"] = []
            log_dict["all_event_times"] = []
            log_dict["all_original_event_times"] = []
            log_dict["survival_predictions"] = []
        elif task_type == "Treatment_Response":
            log_dict["all_labels"] = []
            log_dict["treatment_response_predictions"] = []
            log_dict["treatment_response_logits"] = []
        else:
            raise Exception(f"{task_type} is not supported!")
        log_dict["patient_ids"] = []
        log_dict["dataset_name"] = []
        return log_dict

    def compute_metrics_dict(self, log_dict):
        metrics_dict = {}
        all_risk_scores = np.array(log_dict["all_risk_scores"])
        all_censorships = np.array(log_dict["all_censorships"])
        all_event_times = np.array(log_dict["all_event_times"])
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        metrics_dict["c-index"] = c_index
        return metrics_dict    

    def compute_metrics(self, log_df, task_type="Survival"):
        if task_type == "Survival":
            all_risk_scores = log_df["all_risk_scores"].values
            all_censorships = log_df["all_censorships"].values
            all_event_times = log_df["all_event_times"].values
            outputs = log_df["survival_predictions"].values
            c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
            loss = self.loss_function(torch.tensor(outputs.tolist()), torch.tensor(all_event_times).unsqueeze(-1), None, torch.tensor(all_censorships).unsqueeze(-1))
            metrics_dict = {"c-index": c_index, "Loss": loss}
        elif task_type == "Treatment_Response":
            all_labels = log_df["all_labels"].values
            all_predictions = log_df["treatment_response_predictions"].values
            all_logits = torch.tensor(log_df["treatment_response_logits"].tolist())
            # Calculate AUC

            logits_for_auc = torch.softmax(all_logits, dim=1).numpy()[:, 1]
            if len(logits_for_auc) > 1:
                auc = roc_auc_score(all_labels, logits_for_auc)    
                f1 = f1_score(all_labels, all_predictions, average='macro')
                accuracy = np.mean(all_labels == all_predictions)     
            else:
                auc = np.nan
                f1 = np.nan
                accuracy = np.nan

            # Calculate loss
            all_logits = all_logits.to(self.device)
            all_labels = torch.tensor(all_labels).long().to(self.device)
            loss = self.loss_function(all_logits, all_labels)
            metrics_dict = {"AUC": auc, "Loss": loss, "Accuracy": accuracy, "F1-Score": f1}
        else:
            raise Exception(f"{task_type} is not supported!")
        return metrics_dict

    def compute_metrics_df(self, log_df, task_type="Survival"):
        metrics_dict = {}        
        curr_metrics_dict = self.compute_metrics(log_df, task_type)
        metrics_dict.update(curr_metrics_dict)

        dataset_names = log_df["dataset_name"].unique()
        for dataset in dataset_names:
            dataset_df = log_df[log_df["dataset_name"]==dataset]
            curr_metrics_dict = self.compute_metrics(dataset_df, task_type)
            for key, value in curr_metrics_dict.items():
                metrics_dict[f"{dataset}_{key}"] = value
                metrics_dict[f"{dataset}_{key}"] = value
                        
        metrics_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics_dict.items()}
        return metrics_dict   

   

    def step(self, batch, log_dict, task_type="Survival", device="cuda"):
        batch_data = batch['input']
        labels = batch['label'] #  check this casting        
        #batch_data = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch_data.items()} 
        batch_data = move_to_device(batch_data, device)
        labels = labels.to(device)   
                 
        if len(labels.shape) == 1:
            labels = labels.reshape(-1,1)    
  
        
        result = self.net(batch_data)
        outputs = result['output']

        if task_type == "Survival":
            censorships = batch['censorship'] #  check this casting
            censorships = censorships.to(device)     
            if len(censorships.shape) == 1:
                censorships = censorships.reshape(-1,1) 
            risk, survival = self.calculate_risk(outputs) # output.detach()?
            log_dict["all_risk_scores"]+=(risk.flatten().tolist())
            log_dict["all_censorships"]+=(censorships.detach().view(-1).tolist())
            log_dict["all_event_times"]+=(labels.detach().view(-1).tolist())
            log_dict["all_original_event_times"]+=(batch["original_event_time"].detach().view(-1).tolist())
            log_dict["survival_predictions"] += outputs.detach().tolist()
            if len(risk.shape) == 1:
                risk = risk.reshape(-1,1)    
        elif task_type == "Treatment_Response":
            censorships = None   
            treatment_response_predictions = torch.argmax(outputs.detach().cpu(), dim=1).float()
            log_dict["treatment_response_predictions"] += treatment_response_predictions.numpy().tolist()
            log_dict["treatment_response_logits"] += outputs.detach().cpu().numpy().tolist()
            log_dict["all_labels"] += labels.detach().cpu().numpy().flatten().tolist()
        else:
            raise Exception(f"{task_type} is not supported!")
                    
        log_dict["patient_ids"]+=(batch['patient_id'])
        log_dict["dataset_name"]+=(batch['dataset_name'])

        output = {'outputs': outputs, 'labels': labels, 
                  'censorships': censorships, 'log_dict': log_dict}
        
        if self.AEM_lamda > 0:
            output['attention'] = result['attention']
        return output

    def train(self, 
                    train_dataloader, 
                    eval_dataloader=None, 
                    test_dataloader=None, 
                    task_type="Survival", 
                    checkpoint="example/nn_model.pt", 
                    device="cuda", 
                    debug=False, 
                    path="example", 
                    kfold="", 
                    config=None):
        cudnn.benchmark = False
        trainLoss = []
        validationLoss = []
        testLoss = []
        lowest_val_loss = np.inf
        STOP = False
        df_fold_suffix = f"_{kfold}"
        log_fold_string = f"/{kfold}"

        for epoch in range(self.config.trainer.epochs):
            if STOP:
                logging.info(f'\nSTOPPED at epoch {epoch}')
                break
            if kfold != "":
                logging.info(f'\nStarting training for {kfold}')
            logging.info('\nStarting epoch {}/{}, LR = {}'.format(epoch + 1, self.config.trainer.epochs,
                                                                  self.scheduler.get_last_lr()))
            tloss = []

            batch_numb = 0
            log_dict = {}
            self.net.train()
            train_dataloader.dataset.dataset.set_sample(config.data_loader.sample)
            for idx, batch in tqdm(enumerate(train_dataloader)):
                if debug and batch_numb == DEBUG_BATCHES:
                    print("DEBUG_BATCHES value reached")
                    break
                if idx == 0:
                    log_dict = self.initialize_metrics_dict(task_type)

                step_result = self.step(batch, log_dict, task_type, device)
                        
                outputs = step_result['outputs'] 
                labels = step_result['labels'] 
                censorships = step_result['censorships'] 
                log_dict = step_result['log_dict']



                if self.config.data_loader.batch_size <= self.config.data_loader.real_batch_size:
                    if task_type == "Survival":
                        loss = self.loss_function(outputs, labels, None, censorships)
                    elif task_type == "Treatment_Response":
                        if self.AEM_lamda > 0:
                            attention = step_result['attention']
                            bag_loss = self.loss_function(outputs, labels.squeeze(1))
                            div_loss = torch.sum(F.softmax(attention, dim=-1) * F.log_softmax(attention, dim=-1))
                            loss = self.AEM_lamda * div_loss + bag_loss
                        else:
                            loss = self.loss_function(outputs, labels.squeeze(1))
                    else:
                        raise Exception(f"{task_type} is not supported!")

                    loss = loss / self.NUM_ACCUMULATION_STEPS
                    tloss.append(loss.item()*self.NUM_ACCUMULATION_STEPS)
                    loss.backward()
                    if ((idx + 1) % self.NUM_ACCUMULATION_STEPS == 0):
                        if self.clip_grad_norm_max_norm is not None:
                            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 
                                                           max_norm=self.clip_grad_norm_max_norm)
                        self.optimizer.step()    
                        self.optimizer.zero_grad()
                        if self.config.scheduler.batch_step:                   
                            self.scheduler.step()
                else:
                    raise Exception(f"Batch size > real batch size is not supported!")               
                batch_numb += 1
            tloss = np.array(tloss)
            tloss = np.mean(tloss) # bisognerebbe cambiarlo in base alla reduction della loss
            trainLoss.append(tloss)
            train_df = pd.DataFrame(log_dict)            
            train_df.to_hdf(f"{path}/train_df{df_fold_suffix}.h5", key="df", mode="w")
            train_metrics_dict = self.compute_metrics_df(train_df, task_type)
            train_metrics_df = pd.DataFrame(train_metrics_dict, index=[0])
            train_metrics_df.to_csv(f"{path}/train_metrics{df_fold_suffix}.csv")
            if task_type == "Treatment_Response":
                train_confusion_matrix = accuracy_confusionMatrix_plot(log_dict, train_metrics_df)
            # self.KaplanMeier_plot(log_dict, train_dataloader.dataset.dataset.bins.astype(int))
            # self.predTime_vs_actualTime_confusionMatrix_plot(log_dict)
            to_log = {
                    f'Epoch': epoch + 1,
                    f'LR': self.optimizer.param_groups[0]['lr'],
                    # f'Train/Loss': tloss,
                    # f'Train/c-index': train_metrics_dict["c-index"],
                    # f'Valid/Loss': vloss,
                    # f'Valid/c-index': val_metrics_dict["c-index"],
                    }
            for key, value in train_metrics_dict.items():
                to_log[f'Train{log_fold_string}/{key}'] = value
                
            train_dataloader.dataset.dataset.set_sample(config.data_loader.test_sample)
            if eval_dataloader is not None:
                self.net.eval()
                vloss = []
                vlossWeights = []
                batch_numb = 0
                with torch.inference_mode():
                    for idx, batch in tqdm(enumerate(eval_dataloader)):
                        if debug and batch_numb == DEBUG_BATCHES:
                            break
                        if idx == 0:
                            log_dict = self.initialize_metrics_dict(task_type)
                            
                        step_result = self.step(batch, log_dict, task_type, device)
                        
                        outputs = step_result['outputs'] 
                        labels = step_result['labels'] 
                        censorships = step_result['censorships'] 
                        log_dict = step_result['log_dict']
                        if task_type == "Survival":
                            loss = self.loss_function(outputs, labels, None, censorships)
                        elif task_type == "Treatment_Response":
                            loss = self.loss_function(outputs, labels.squeeze(1))
                        else:
                            raise Exception(f"{task_type} is not supported!")
                        if self.config.data_loader.batch_size <= self.config.data_loader.real_batch_size:
                            vloss.append(loss.item())
                        else:
                            vloss.append(loss.detach().mean().item())
                        vlossWeights.append(batch["label"].size(dim=0))
                        batch_numb += 1
                vloss = np.array(vloss)
                vloss = np.average(vloss, weights=vlossWeights)
                # vloss = np.sum(vloss)
                validationLoss.append(vloss)
                val_df = pd.DataFrame(log_dict)                
                val_df.to_hdf(f"{path}/val_df{df_fold_suffix}.h5", key="df", mode="w")
                val_metrics_dict = self.compute_metrics_df(val_df, task_type)
                val_metrics_df = pd.DataFrame(val_metrics_dict, index=[0])
                val_metrics_df.to_csv(f"{path}/val_metrics{df_fold_suffix}.csv")
                if task_type == "Treatment_Response":
                    val_confusion_matrix = accuracy_confusionMatrix_plot(log_dict, val_metrics_df)
                for key, value in val_metrics_dict.items():
                    to_log[f'Valid{log_fold_string}/{key}'] = value


            if test_dataloader is not None:
                self.net.eval()
                ttloss = []
                ttlossWeights = []
                batch_numb = 0
                with torch.inference_mode():
                    for idx, batch in tqdm(enumerate(test_dataloader)):
                        if debug and batch_numb == DEBUG_BATCHES:
                            break
                        if idx == 0:
                            log_dict = self.initialize_metrics_dict(task_type)

                        step_result = self.step(batch, log_dict, task_type, device)
                        
                        outputs = step_result['outputs'] 
                        labels = step_result['labels'] 
                        censorships = step_result['censorships'] 
                        log_dict = step_result['log_dict']
                        
                        if task_type == "Survival":
                            loss = self.loss_function(outputs, labels, None, censorships)
                        elif task_type == "Treatment_Response":
                            loss = self.loss_function(outputs, labels.squeeze(1))
                        else:
                            raise Exception(f"{task_type} is not supported!")
                        if self.config.data_loader.batch_size <= self.config.data_loader.real_batch_size:
                            ttloss.append(loss.item())
                        else:
                            ttloss.append(loss.detach().mean().item())
                        ttlossWeights.append(batch["label"].size(dim=0))
                        batch_numb += 1
                ttloss = np.array(ttloss)
                ttloss = np.average(ttloss, weights=ttlossWeights)
                # ttloss = np.sum(ttloss)
                testLoss.append(ttloss)
                test_df = pd.DataFrame(log_dict)                
                test_df.to_hdf(f"{path}/test_df{df_fold_suffix}.h5", key="df", mode="w")
                test_metrics_dict = self.compute_metrics_df(test_df, task_type)
                test_metrics_df = pd.DataFrame(test_metrics_dict, index=[0])
                test_metrics_df.to_csv(f"{path}/test_metrics{df_fold_suffix}.csv")
                if task_type == "Treatment_Response":
                    test_confusion_matrix = accuracy_confusionMatrix_plot(log_dict, test_metrics_df)

                test_log = {
                    # f'Test/Loss': ttloss,
                    # f'Test/c-index': test_metrics_dict["c-index"],    
                    } 
                for key, value in test_metrics_dict.items():
                    test_log[f'Test{log_fold_string}/{key}'] = value
                
                to_log.update(test_log) 
                if task_type == "Treatment_Response":
                    plot_to_log = {
                        f"Train{log_fold_string}/Confusion_Matrix": wandb.Image(train_confusion_matrix),
                    }
                    if eval_dataloader is not None:
                        plot_to_log[f"Valid{log_fold_string}/Confusion_Matrix"] = wandb.Image(val_confusion_matrix)
                    if test_dataloader is not None:
                        plot_to_log[f"Test{log_fold_string}/Confusion_Matrix"] = wandb.Image(test_confusion_matrix)
                    to_log.update(plot_to_log) 
                                     
                   
            wandb.log(to_log)     
      
            if self.config.scheduler.batch_step==None or self.config.scheduler.batch_step==False:
                self.scheduler.step()
            # Early stopping
            if eval_dataloader is not None:
                if vloss < lowest_val_loss:
                    lowest_val_loss = vloss
                    patience_counter = 0
                    lowest_val_loss_epoch = epoch + 1
                    logging.info(
                        f"############################################ New lowest_val_loss reached: {lowest_val_loss} #########################")
                    if kfold != "":
                        checkpoint_splitted = checkpoint.split(".")
                        checkpoint = f"{checkpoint_splitted[0]}{df_fold_suffix}.pt"
                    torch.save(self.net, checkpoint)
                    wandb.run.summary["Lowest_Validation_Loss/Epoch"] = lowest_val_loss_epoch
                    # wandb.run.summary["Lowest_Validation_Loss/Validation_Loss"] = lowest_val_loss
                    # wandb.run.summary["Lowest_Validation_Loss/Validation_c-index"] = val_metrics_dict["c-index"],
                    for key, value in val_metrics_dict.items():
                        wandb.run.summary[f"Lowest_Validation_Loss/Valid{log_fold_string}/{key}"] = value
                    # # wandb.run.summary["Lowest_Validation_Loss/Validation_KM"] = val_metrics_dict["KM"],
                    train_df.to_hdf(f"{path}/best_train_df{df_fold_suffix}.h5", key="df", mode="w")
                    train_metrics_df.to_csv(f"{path}/best_train_metrics{df_fold_suffix}.csv")
                    val_df.to_hdf(f"{path}/best_val_df{df_fold_suffix}.h5", key="df", mode="w")
                    val_metrics_df.to_csv(f"{path}/best_val_metrics{df_fold_suffix}.csv")                
                elif patience_counter == self.config.trainer.patience:
                    logging.info(f"End of training phase - Patience threshold reached\nWeights Restored from Lowest val_loss epoch: {lowest_val_loss_epoch}\nlowest_val_loss: {lowest_val_loss}")
                    STOP = True
                else:
                    patience_counter += 1

    def evaluate(self, test_dataloader, task_type="Survival", checkpoint=None, device="cuda", best=False, path="", kfold=""):
        cudnn.benchmark = False
        logging.info("test")   
        df_fold_suffix = f"_{kfold}"
        log_fold_string = f"/{kfold}"     
        if best:
            if kfold != "":
                checkpoint_splitted = checkpoint.split(".")
                checkpoint = f"{checkpoint_splitted[0]}{df_fold_suffix}.pt"
            net = torch.load(checkpoint)
            logging.info("\n Evalate best model")
        else:
            net = self.net
            logging.info("\n Evalate last model")

        net = net.to(device)
        net.eval()
        tloss = []
        tlossWeights = []
        with torch.inference_mode():
            for idx, batch  in enumerate(test_dataloader):
                if idx == 0:
                        log_dict = self.initialize_metrics_dict(task_type)
                # batch_data = torch.squeeze(batch_data, 0)
                step_result = self.step(batch, log_dict, task_type, device)
                
                outputs = step_result['outputs'] 
                labels = step_result['labels'] 
                censorships = step_result['censorships'] 
                log_dict = step_result['log_dict']


                if task_type == "Survival":
                    loss = self.loss_function(outputs, labels, None, censorships)
                elif task_type == "Treatment_Response":
                    loss = self.loss_function(outputs, labels.squeeze(1))
                else:
                        raise Exception(f"{task_type} is not supported!")

                if self.config.data_loader.batch_size <= self.config.data_loader.real_batch_size:
                    tloss.append(loss.item())
                else:
                    tloss.append(loss.detach().mean().item())
                tlossWeights.append(batch["label"].size(dim=0))

        tloss = np.array(tloss)
        tloss = np.average(tloss, weights=tlossWeights)
        test_df = pd.DataFrame(log_dict)
        test_metrics_dict = self.compute_metrics_df(test_df, task_type)

        if best:
            # wandb.run.summary["Lowest_Validation_Loss/Test_Loss"] = tloss
            # wandb.run.summary["Lowest_Validation_Loss/Test_c-index"] = test_metrics_dict["c-index"]
            for key, value in test_metrics_dict.items():
                    wandb.run.summary[f"Lowest_Validation_Loss/Test{log_fold_string}/{key}"] = value
            test_df.to_hdf(f"{path}/best_test_df{df_fold_suffix}.h5", key="df", mode="w")
            test_metrics_df = pd.DataFrame(test_metrics_dict, index=[0])
            test_metrics_df.to_csv(f"{path}/best_test_metrics{df_fold_suffix}.csv")
            if task_type == "Treatment_Response":
                test_confusion_matrix = accuracy_confusionMatrix_plot(log_dict, test_metrics_df)
        else:
            for key, value in test_metrics_dict.items():
                    wandb.run.summary[f"Last_Epoch_Model/Test{log_fold_string}/{key}"] = value
            test_df.to_hdf(f"{path}/last_epoch_test_df{df_fold_suffix}.h5", key="df", mode="w")
            test_metrics_df = pd.DataFrame(test_metrics_dict, index=[0])
            test_metrics_df.to_csv(f"{path}/last_epoch_test_metrics{df_fold_suffix}.csv")
            if task_type == "Treatment_Response":
                test_confusion_matrix = accuracy_confusionMatrix_plot(log_dict, test_metrics_df)

    def log_aggregated(self, result_id_path):
        out = kfold_results_merge(result_id_path)
        to_log = {
         "Last_Epoch_Model/Test/Aggregated/AUC": out['AUC'],
         "Last_Epoch_Model/Test/Aggregated/Accuracy": out['Accuracy'],
         "Last_Epoch_Model/Test/Aggregated/F1-Score": out['F1-Score'],
         "Last_Epoch_Model/Test/Aggregated/Confusion_Matrix": wandb.Image(out['Confusion_Matrix']),
         }
        wandb.log(to_log)

