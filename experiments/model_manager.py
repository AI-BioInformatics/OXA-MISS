import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
import torch.nn.functional as F
import wandb
import logging, datetime
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
import io, copy
from PIL import Image
from .utils import accuracy_confusionMatrix_plot, kfold_results_merge, move_to_device, import_class_from_path
import os
# from .metrics.loss_func import NLLSurvLoss
# from .scheduler import *
from adam_atan2_pytorch import AdoptAtan2
DEBUG_BATCHES = 8

class ModelManager():
    def __init__(self,
                 config, 
                 ModelClass,
                 results_store
                 ):
        self.results_store = results_store
        self.config = config
        self.device = config.model.device
        self.real_batch_size = config.data_loader.real_batch_size
        self.NUM_ACCUMULATION_STEPS = self.real_batch_size//config.data_loader.batch_size
        self.attention_dir_check = False
        
        model_kwargs = config.model.kwargs
        self.net = ModelClass(**model_kwargs) 
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
        if self.config.optimizer.name == "AdoptAtan2":
            return AdoptAtan2(self.net.parameters(), 
                              lr = self.config.optimizer.learning_rate)
        elif self.config.optimizer.name == "Adam":
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
            self.net = torch.load(path, weights_only=False)
        else:
            logging.info(f'\nRestoring model weigths from: {path}')
            self.net = torch.load(path, map_location=torch.device('cpu'), weights_only=False)

    def load_checkpoint(self, path, device='cpu'):
        if device == 'cuda':
            weights = torch.load(path, weights_only=False).state_dict()#["state_dict"] 
            self.net.load_state_dict(weights, strict=False) 
        else:
            weights   =  torch.load(path, map_location=torch.device(device), weights_only=False)["state_dict"] 
            self.net.load_state_dict(weights, strict=False) 
    def get_dataset_name(self):
        split_path = self.config.data_loader.KFold.splits if isinstance(self.config.data_loader.KFold.splits, str) else self.config.data_loader.KFold.splits[0]
        if "chemorefractory" in split_path.lower():
            return "Decider"
        else:
            return split_path.strip("/").split("/")[-1]

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
                if np.unique(all_labels).size>1:
                    auc = roc_auc_score(all_labels, logits_for_auc)    
                else:
                    auc = np.nan
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

    def save_XA_attentions(self, batch, step_result, save_path, partition="test", epoch="last"):
        if "XA_attentions" not in step_result:
            return
        attentions = step_result['XA_attentions']
        slides_str_descriptors = batch['slides_str_descriptor']
        patient_ids = batch['patient_id']
        dataset_names = batch['dataset_name']
        if not self.attention_dir_check:
            if not os.path.exists(f"{save_path}/attention"):
                os.makedirs(f"{save_path}/attention")
            self.attention_dir_check = True
        for i, patient_id in enumerate(patient_ids):
            dataset_name = dataset_names[i]
            for key, val in attentions.items():
                if val is not None:
                    slides = slides_str_descriptors[i].split("|")
                    start = 0
                    for slide in slides:
                        slidename, num_patches = slide.split("#")
                        num_patches = int(num_patches)
                        if key == "att_genomics_to_patches":
                            temp = val[i,:,:,start:start+num_patches].detach().cpu()
                        elif key == "att_patches_to_genomics":
                            temp = val[i,:,start:start+num_patches,:].detach().cpu()
                        else:
                            continue
                        
                        torch.save(temp, f"{save_path}/attention/{dataset_name}_{patient_id}_{slidename}_{partition}_{epoch}_{key}.pt")
                        start += num_patches

    def adjust_status(self, data, scenario):
        # return data
        for status_key, modality_key in zip(["WSI_status", "genomics_status", "cnv_status"], ["WSI", "Genomics", "CNV"]):
            if modality_key in self.config.model.kwargs.input_modalities:
                # controllo se lo scenario è uno scenario in cui manca una sola modalità e se essa corrisponde a quella dello status_key corrente
                # ----> devo impostaere le altre modalita a true, altrimenti rischio di avere tutte le modalità false
                # QUINDI STO ASSUMENDO CHE DI DEFAULT AVREI TUTTE LE MODALITA' DISPONIBILI
                if  '_miss_' in scenario: 
                    if modality_key.lower() in scenario:
                        data[status_key] = torch.tensor([data['missing_modality_test_scenarios'][scenario]])
                    else:
                        data[status_key] = torch.tensor([True])
                    
                        
                # se mancano tutte le modalità
                elif scenario.startswith('missing_all_'):
                    rate = scenario.split('_')[-1]
                    modality = modality_key.lower()
                    scenario_modality = f'missing_all_{modality}_{rate}'
                    prev = data[status_key].item() 
                    data[status_key] = torch.tensor([data['missing_modality_test_scenarios'][scenario_modality]])
                    if not prev and data[status_key].item():
                        print('debug') 
        return data
    
    def step(self, batch, log_dict, task_type="Survival", device="cuda", model=None, eval_missing_modality_scenario=None, is_eval=False):
        batch_data = batch['input']
        if eval_missing_modality_scenario:
            is_eval = True
            batch_data = copy.deepcopy(batch['input'])
            batch_data = self.adjust_status(batch_data, eval_missing_modality_scenario)
            
        labels = batch['label'] #  check this casting        
        #batch_data = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch_data.items()} 
        batch_data = move_to_device(batch_data, device)
        labels = labels.to(device)   
                 
        if len(labels.shape) == 1:
            labels = labels.reshape(-1,1)    
  
        model = model if model != None else self.net
        # torch.cuda.synchronize()
        if model.__class__.__name__ == "MUSE":
            if not is_eval:
                result = model(batch_data, labels)
            else:
                result = model.inference(batch_data)
        else:
            result = model(batch_data) # per TITANS funziona model(batch_data['patch_features'].squeeze(0).long())
        # torch.cuda.synchronize()
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
        if "XA_attentions" in result:
            output['XA_attentions'] = result['XA_attentions']   
        
        if model.__class__.__name__ in ["MUSE", "ProSurv"] and not is_eval:
            output['partial_loss'] = result["partial_loss"]

        if self.AEM_lamda > 0 and batch_data["WSI_status"].item() is True and 'attention' in result:            
            output['attention'] = result['attention']

        return output

    def train(self, 
                    train_dataloader, 
                    eval_dataloader=None, 
                    test_dataloader=None, 
                    task_type="Survival", 
                    checkpoint_last_epoch='{path}/model_last_epoch.pt',
                    checkpoint_model_lowest_loss = '{path}/model_lowest_loss.pt', 
                    checkpoint_model_highest_metric = '{path}/model_highest_metric.pt',
                    device="cuda", 
                    debug=False, 
                    path="example", 
                    kfold="", 
                    config=None):
        # cudnn.benchmark = False
        trainLoss = []
        validationLoss = []
        testLoss = []
        lowest_val_loss = np.inf
        highest_val_metric_monitor = -1
        STOP = False
        df_fold_suffix = f"_{kfold}" if kfold else ""
        log_fold_string = f"/{kfold}" if kfold else ""

        model_last_epoch = None
        model_highest_metric = None
        model_lowest_loss = None

        if kfold != "":
            checkpoint_splitted_last = checkpoint_last_epoch.split(".")
            checkpoint_last_epoch = f"{checkpoint_splitted_last[0]}{df_fold_suffix}.pt"



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

            if hasattr(self.config.trainer, 'robust_training'):
                if self.config.trainer.robust_training is True:
                    logging.info(f'Robust training is enabled for training')
                    train_dataloader.dataset.dataset.set_robust_training_on()
            
            for idx, batch in tqdm(enumerate(train_dataloader)):
                if debug and batch_numb == DEBUG_BATCHES:
                    print("DEBUG_BATCHES value reached")
                    break
                if idx == 0:
                    log_dict = self.initialize_metrics_dict(task_type)

                step_result = self.step(batch, log_dict, task_type, device)
                # self.save_XA_attentions(batch, step_result, path, partition="train", epoch="last") # remove it! it is here for debugging
                        
                outputs = step_result['outputs'] 
                labels = step_result['labels'] 
                censorships = step_result['censorships'] 
                log_dict = step_result['log_dict']



                if self.config.data_loader.batch_size <= self.config.data_loader.real_batch_size:
                    if task_type == "Survival":
                        loss = self.loss_function(outputs, labels, None, censorships)
                        if self.net.__class__.__name__ in ["MUSE", "ProSurv"]:
                            loss += step_result["partial_loss"]
                            
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
                    # torch.cuda.synchronize()
                    loss.backward()
                    # torch.cuda.synchronize()
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
            model_last_epoch = copy.deepcopy(self.net)
            model_last_epoch.to('cpu')
            tloss = np.array(tloss)
            tloss = np.mean(tloss) # bisognerebbe cambiarlo in base alla reduction della loss
            trainLoss.append(tloss)
            train_df = pd.DataFrame(log_dict)            
            # train_df.to_hdf(f"{path}/train_df{df_fold_suffix}.h5", key="df", mode="w")
            train_metrics_dict = self.compute_metrics_df(train_df, task_type)
            train_metrics_df = pd.DataFrame(train_metrics_dict, index=[0])
            # train_metrics_df.to_csv(f"{path}/train_metrics{df_fold_suffix}.csv")
            if task_type == "Treatment_Response":
                train_confusion_matrix = accuracy_confusionMatrix_plot(log_dict, train_metrics_df)
            # self.KaplanMeier_plot(log_dict, train_dataloader.dataset.dataset.bins.astype(int))
            # self.predTime_vs_actualTime_confusionMatrix_plot(log_dict)
            to_log = {
                    f'Epoch': epoch + 1,
                    f'LR': self.optimizer.param_groups[0]['lr'],
                    f'dataset_name': self.get_dataset_name(),
                    f'modality_setting': self.config.data_loader.missing_modalities_tables.missing_mod_rate
                    # f'Train/Loss': tloss,
                    # f'Train/c-index': train_metrics_dict["c-index"],
                    # f'Valid/Loss': vloss,
                    # f'Valid/c-index': val_metrics_dict["c-index"],
                    }
            for key, value in train_metrics_dict.items():
                to_log[f'Train{log_fold_string}/{key}'] = value
                
            train_dataloader.dataset.dataset.set_sample(config.data_loader.test_sample)
            if hasattr(self.config.trainer, 'robust_training'):
                if self.config.trainer.robust_training is True:
                    logging.info(f'Robust training is disabled for evaluation')
                    train_dataloader.dataset.dataset.set_robust_training_off()

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
                            
                        step_result = self.step(batch, log_dict, task_type, device, is_eval=True)
                        
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
                # val_df.to_hdf(f"{path}/val_df{df_fold_suffix}.h5", key="df", mode="w")
                val_metrics_dict = self.compute_metrics_df(val_df, task_type)
                if task_type == "Treatment_Response":
                    val_metric_monitor = (val_metrics_dict['AUC'] + val_metrics_dict['F1-Score']) / 2 
                else:
                    val_metric_monitor = val_metrics_dict["c-index"]
                                
                val_metrics_df = pd.DataFrame(val_metrics_dict, index=[0])
                # val_metrics_df.to_csv(f"{path}/val_metrics{df_fold_suffix}.csv")
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

                        step_result = self.step(batch, log_dict, task_type, device, is_eval=True)
                        # self.save_XA_attentions(batch, step_result, path, partition="test", epoch="last")
                        
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
                # test_df.to_hdf(f"{path}/test_df{df_fold_suffix}.h5", key="df", mode="w")
                test_metrics_dict = self.compute_metrics_df(test_df, task_type)
                test_metrics_df = pd.DataFrame(test_metrics_dict, index=[0])
                # test_metrics_df.to_csv(f"{path}/test_metrics{df_fold_suffix}.csv")
                if task_type == "Treatment_Response":
                    test_confusion_matrix = accuracy_confusionMatrix_plot(log_dict, test_metrics_df)

                test_log = {} 
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

                # Stesso codice ma adattato per testare anche i missing mod scenarios
                if config.missing_modality_test.active and config.missing_modality_test.test_scenarios_on_each_epoch:
                    for scenario in config.missing_modality_test.scenarios:
                        eval_missing_modality_scenario_suffix = f"_{scenario}"
                        emms_suffix = eval_missing_modality_scenario_suffix
                        
                        ttloss = []
                        ttlossWeights = []
                        batch_numb = 0
                        with torch.inference_mode():
                            for idx, batch in tqdm(enumerate(test_dataloader)):
                                if debug and batch_numb == DEBUG_BATCHES:
                                    break
                                if idx == 0:
                                    log_dict = self.initialize_metrics_dict(task_type)
                                
                                step_result = self.step(batch, log_dict, task_type, device, 
                                                        eval_missing_modality_scenario=scenario)
                                # self.save_XA_attentions(batch, step_result, path, partition="test", epoch="last")
                                
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
                        # test_df.to_hdf(f"{path}/test_df{df_fold_suffix}{emms_suffix}.h5", key="df", mode="w")
                        test_metrics_dict = self.compute_metrics_df(test_df, task_type)
                        test_metrics_df = pd.DataFrame(test_metrics_dict, index=[0])
                        # test_metrics_df.to_csv(f"{path}/test_metrics{df_fold_suffix}{emms_suffix}.csv")
                        if task_type == "Treatment_Response":
                            test_confusion_matrix = accuracy_confusionMatrix_plot(log_dict, test_metrics_df)

                        test_log = {
                            # f'Test/Loss': ttloss,
                            # f'Test/c-index': test_metrics_dict["c-index"],    
                            } 
                        for key, value in test_metrics_dict.items():
                            test_log[f'Test{log_fold_string}/Missing_modalities_scenarios/{scenario}/{key}'] = value
                        
                        to_log.update(test_log) 
                        if task_type == "Treatment_Response":
                            plot_to_log = {
                                # f"Train{log_fold_string}/Confusion_Matrix": wandb.Image(train_confusion_matrix),
                            }
                            # if eval_dataloader is not None:
                            #     plot_to_log[f"Valid{log_fold_string}/Confusion_Matrix"] = wandb.Image(val_confusion_matrix)
                            if test_dataloader is not None:
                                plot_to_log[f"Test{log_fold_string}/Missing_modalities_scenarios/{scenario}/Confusion_Matrix"] = wandb.Image(test_confusion_matrix)
                            to_log.update(plot_to_log) 
                                     
                   
            wandb.log(to_log)     
      
            if self.config.scheduler.batch_step==None or self.config.scheduler.batch_step==False:
                self.scheduler.step()
            # Early stopping
            if eval_dataloader is not None:

                if val_metric_monitor > highest_val_metric_monitor:
                    highest_val_metric_monitor = val_metric_monitor
                    
                    highest_val_metric_monitor_epoch = epoch + 1
                    logging.info(
                        f"############################################ New highest_val_metric_monitor reached: {highest_val_metric_monitor} #########################")
                    
                    model_highest_metric = copy.deepcopy(self.net)
                    model_highest_metric.to('cpu')
                    wandb.run.summary["Highest_Metric/Epoch"] = highest_val_metric_monitor_epoch
                    for key, value in val_metrics_dict.items():
                        wandb.run.summary[f"Highest_Metric/Valid{log_fold_string}/{key}"] = value
                        
                    
                    # train_df.to_hdf(f"{path}/best_train_df_highest_metric{df_fold_suffix}.h5", key="df", mode="w")
                    # train_metrics_df.to_csv(f"{path}/best_train_metrics_highest_metric{df_fold_suffix}.csv")

                    # val_df.to_hdf(f"{path}/best_val_df_highest_metric{df_fold_suffix}.h5", key="df", mode="w")
                    # val_metrics_df.to_csv(f"{path}/best_val_metrics_highest_metric{df_fold_suffix}.csv")  

                    # test_df.to_hdf(f"{path}/best_test_df_highest_metric{df_fold_suffix}.h5", key="df", mode="w")
                    # test_metrics_df.to_csv(f"{path}/best_test_metrics_highest_metric{df_fold_suffix}.csv") 

                if vloss < lowest_val_loss:
                    lowest_val_loss = vloss
                    patience_counter = 0
                    lowest_val_loss_epoch = epoch + 1
                    logging.info(
                        f"############################################ New lowest_val_loss reached: {lowest_val_loss} #########################")
                    # if kfold != "":
                    #     checkpoint_splitted = checkpoint.split(".")
                    #     checkpoint = f"{checkpoint_splitted[0]}{df_fold_suffix}.pt"
                    # torch.save(self.net, checkpoint)
                    model_lowest_loss = copy.deepcopy(self.net)
                    model_lowest_loss.to('cpu')
                    wandb.run.summary["Lowest_Validation_Loss/Epoch"] = lowest_val_loss_epoch
                    # wandb.run.summary["Lowest_Validation_Loss/Validation_Loss"] = lowest_val_loss
                    # wandb.run.summary["Lowest_Validation_Loss/Validation_c-index"] = val_metrics_dict["c-index"],
                    for key, value in val_metrics_dict.items():
                        wandb.run.summary[f"Lowest_Validation_Loss/Valid{log_fold_string}/{key}"] = value
                    # # wandb.run.summary["Lowest_Validation_Loss/Validation_KM"] = val_metrics_dict["KM"],
                    # train_df.to_hdf(f"{path}/best_train_df_lowest_loss{df_fold_suffix}.h5", key="df", mode="w")
                    # train_metrics_df.to_csv(f"{path}/best_train_metrics_lowest_loss{df_fold_suffix}.csv")

                    # val_df.to_hdf(f"{path}/best_val_df_lowest_loss{df_fold_suffix}.h5", key="df", mode="w")
                    # val_metrics_df.to_csv(f"{path}/best_val_metrics_lowest_loss{df_fold_suffix}.csv")      

                    # test_df.to_hdf(f"{path}/best_test_df_lowest_loss{df_fold_suffix}.h5", key="df", mode="w")
                    # test_metrics_df.to_csv(f"{path}/best_test_metrics_lowest_loss{df_fold_suffix}.csv")    

                elif patience_counter == self.config.trainer.patience:
                    logging.info(f"End of training phase - Patience threshold reached\nWeights Restored from Lowest val_loss epoch: {lowest_val_loss_epoch}\nlowest_val_loss: {lowest_val_loss}")
                    STOP = True
                else:
                    patience_counter += 1

        if self.config.model.save_checkpoints:
            torch.save(model_last_epoch, checkpoint_last_epoch)
        
            if eval_dataloader is not None:
                for checkpoint, model in zip([checkpoint_model_lowest_loss, checkpoint_model_highest_metric], [model_lowest_loss, model_highest_metric]):
                    if kfold != "":
                        checkpoint_splitted = checkpoint.split(".")
                        checkpoint = f"{checkpoint_splitted[0]}{df_fold_suffix}.pt"
                    torch.save(model, checkpoint)

    def evaluate(self, test_dataloader, 
                task_type="Survival", 
                checkpoint_last_epoch='{path}/model_last_epoch.pt',
                checkpoint_model_lowest_loss = '{path}/model_lowest_loss.pt', 
                checkpoint_model_highest_metric = '{path}/model_highest_metric.pt',
                device="cuda", 
                best=False, 
                path="", 
                kfold="",
                log_aggregated=False,
                log_on_telegram=True,
                Save_XA_attention_files=False,
                eval_missing_modality_scenario=None,
                print_demo_results=False,
                is_demo_test=False,
                repo_path=None):
        if not eval_missing_modality_scenario:
            eval_missing_modality_scenario_suffix = ""
        else:
            eval_missing_modality_scenario_suffix = f"_{eval_missing_modality_scenario}"

        emms_suffix = eval_missing_modality_scenario_suffix
        # cudnn.benchmark = False
        logging.info("test")   
        df_fold_suffix = f"_{kfold}"
        log_fold_string = f"/{kfold}"     
        models = []
        summary_paths = []
        
        if best:
            if kfold != "":
                checkpoint_splitted_last = checkpoint_last_epoch.split(".")
                checkpoint_last_epoch = f"{checkpoint_splitted_last[0]}{df_fold_suffix}.pt"

                checkpoint_splitted_lowest_l = checkpoint_model_lowest_loss.split(".")
                checkpoint_model_lowest_loss = f"{checkpoint_splitted_lowest_l[0]}{df_fold_suffix}.pt"

                checkpoint_splitted_highest_m = checkpoint_model_highest_metric.split(".")
                checkpoint_model_highest_metric = f"{checkpoint_splitted_highest_m[0]}{df_fold_suffix}.pt"
            if os.path.exists(checkpoint_model_lowest_loss):
                model_lowest_l = torch.load(checkpoint_model_lowest_loss, weights_only=False)
            else:
                model_lowest_l = torch.load(checkpoint_last_epoch, weights_only=False)
            models.append(model_lowest_l)
            summary_paths.append('Lowest_Validation_Loss_Model/Test')
            
            if os.path.exists(checkpoint_model_highest_metric):
                model_highest_m = torch.load(checkpoint_model_highest_metric, weights_only=False)
            else:
                model_highest_m = torch.load(checkpoint_last_epoch, weights_only=False)
            # model_highest_m = torch.load(checkpoint_model_highest_metric, weights_only=False)
            
            models.append(model_highest_m)
            summary_paths.append('Highest_Validation_Metric_Model/Test')
                
            last_model = torch.load(checkpoint_last_epoch, weights_only=False)
            logging.info("\n Evaluate best model")
        else:
            last_model = self.net
            logging.info("\n Evaluate last model")

        models.append(last_model)
        summary_paths.append('Last_Epoch_Model/Test')

        if emms_suffix:
            for i, (model, summary_path) in enumerate(zip(models, summary_paths)):
                summary_paths[i] += f"/Missing_modalities_scenarios/{eval_missing_modality_scenario}"        

        scenario = eval_missing_modality_scenario if eval_missing_modality_scenario else "base"
        for model, summary_path in zip(models, summary_paths):
            model = model.to(device)
            model.eval()
            tloss = []
            tlossWeights = []
            with torch.inference_mode():
                for idx, batch  in enumerate(test_dataloader):
                    if idx == 0:
                            log_dict = self.initialize_metrics_dict(task_type)
                    # batch_data = torch.squeeze(batch_data, 0)
                    step_result = self.step(batch, log_dict, task_type, device, model, eval_missing_modality_scenario, is_eval=True)
                    # step_result = self.step(batch, log_dict, task_type, device, model)
                    if Save_XA_attention_files:
                        self.save_XA_attentions(batch, step_result, path, partition="test", epoch="best" if best else "last")
                    
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

            self.results_store.add_result(
                scenario=scenario,
                model_version=summary_path.split("/")[0],
                fold_result=test_metrics_dict,
                fold_num=int(kfold.split('_')[1]) if kfold else 0
            )

            for key, value in test_metrics_dict.items():
                wandb.run.summary[f"{summary_path}{log_fold_string}/{key}"] = value

        model_name = model.__class__.__name__ 

        if log_aggregated:
            test_scenario = scenario
            aggregated_metrics = self.results_store.compute_aggregated_metrics(test_scenario, task_type)
            self._log_aggregated_metrics(aggregated_metrics, test_scenario, task_type)
            
            if log_on_telegram:
                from .telegram_logger import send_telegram_message
                import asyncio
            columns=["ID", "model_name", "dataset_name", "model_version", "End Time", "seed", "modality_setting", "internal_val_size", "input_modalities","test_scenario"]
            for key, value in self.config.model.kwargs.items():
                if key not in ["input_modalities"]:
                    columns.append(key)
            if task_type == 'Treatment_Response':
                csv_path = f"{repo_path}/experiments/test_results_csv/TS_{model_name}.csv"
                if not os.path.exists(csv_path):
                    columns += ["F1-Score_mean", "F1-Score_std", 'AUC_mean', 'AUC_std', 'Accuracy_mean', 'Accuracy_std', 'Mean_F1-Score_AUC']
                    df_old_records = pd.DataFrame(columns=columns)
                else:
                    df_old_records = pd.read_csv(csv_path)
                max_f1 = df_old_records["F1-Score_mean"].max()
                max_auc = df_old_records["AUC_mean"].max()
                max_acc = df_old_records["Accuracy_mean"].max()
                # df_old_records["Mean_AUC_F1"] = (df_old_records["AUC_mean"] + df_old_records["F1-Score_mean"]) / 2
                max_mean_auc_f1 = df_old_records["Mean_F1-Score_AUC"].max()
                df_old_records_sorted = df_old_records.sort_values(by="Mean_F1-Score_AUC", ascending=False)
                # for i, log in enumerate(aggregated_metrics):
                for model_version, metrics in aggregated_metrics.items():
                    overperformed_metrics = []

                    current_f1_mean = metrics['F1-Score_mean']
                    current_auc_mean = metrics['AUC_mean']
                    current_acc_mean = metrics['Accuracy_mean']
                    current_f1_std = metrics['F1-Score_std']
                    current_auc_std = metrics['AUC_std']
                    current_acc_std = metrics['Accuracy_std']

                    if current_f1_mean > max_f1: overperformed_metrics.append('F1-Score')
                    if current_auc_mean > max_auc: overperformed_metrics.append('AUC')
                    if current_acc_mean > max_acc: overperformed_metrics.append('Accuracy')

                    current_mean_auc_f1 = (current_auc_mean + current_f1_mean) / 2
                    position = (df_old_records_sorted["Mean_F1-Score_AUC"] > current_mean_auc_f1).sum() + 1
                    if log_on_telegram and (position <= 10 or current_mean_auc_f1 > max_mean_auc_f1-0.035 or overperformed_metrics):
                        missing_mod_test_part = "\n*Missing modality scenario*: " + str(eval_missing_modality_scenario) if eval_missing_modality_scenario else ""
                        telegram_message = f"*Run id*: {wandb.run.id} \n*Seed*: {self.config.seed} \n*Dataset*: {self.get_dataset_name()} \n*Task*: {task_type}  \n*Model Class*: {self.net.__class__.__name__}\n*Model version*: {model_version}{missing_mod_test_part}\n\n*F1s mean*: {current_f1_mean:.3f} ± {current_f1_std:.3f}\n*AUC mean*: {current_auc_mean:.3f} ± {current_auc_std:.3f}\n*Accuracy mean*: {current_acc_mean:.3f} ± {current_acc_std:.3f}\n\n"                
                        telegram_message += f"*overperformed metrics*: {overperformed_metrics if overperformed_metrics else 'none'}\n\n"
                        telegram_message += f"*Mean F1s-AUC*: {current_mean_auc_f1:.3f} -> {position}° best run"
                        asyncio.run(send_telegram_message(telegram_message))
        
                
            # elif task_type == 'Survival':
            else:
                csv_path = f"{repo_path}/experiments/test_results_csv/Surv_{model_name}.csv"
                if is_demo_test:
                    csv_path = csv_path.replace(".csv", "_demo_test.csv")
                if not os.path.exists(csv_path):
                    columns += ["c-index_mean", "c-index_std", "c-index_list"]
                    df_old_records = pd.DataFrame(columns=columns)
                else:
                    df_old_records = pd.read_csv(csv_path)
                    if 'c-index_list' not in df_old_records.columns:
                        df_old_records['c-index_list'] = None
                max_c_index_mean = df_old_records["c-index_mean"].max() if "c-index_mean" in df_old_records.columns and df_old_records.size else 0 
                df_old_records_sorted = df_old_records.sort_values(by="c-index_mean", ascending=False) if "c-index_mean" in df_old_records.columns and df_old_records.size else None
                for model_version, metrics in aggregated_metrics.items():
                    overperformed_metrics = []

                    current_c_index_mean = metrics['c-index_mean']
                    current_c_index_std = metrics['c-index_std']

                    if current_c_index_mean > max_c_index_mean: overperformed_metrics.append('C-index')
                    
                    if df_old_records_sorted is not None and df_old_records_sorted.size:
                        position = (df_old_records_sorted["c-index_mean"] > current_c_index_mean).sum() + 1
                    else:
                        position = 1
                    if log_on_telegram and (position <= 10 or current_c_index_mean > max_c_index_mean-0.035 or overperformed_metrics):
                        missing_mod_test_part = "\n*Missing modality scenario*: " + str(eval_missing_modality_scenario) if eval_missing_modality_scenario else ""
                        telegram_message = f"*Run id*: {wandb.run.id} \n*Seed*: {self.config.seed} \n*Dataset*: {self.get_dataset_name()} \n*Task*: {task_type} \n*Model Class*: {self.net.__class__.__name__}\n*Model version*: {model_version}{missing_mod_test_part}\n\n*C-index mean*: {current_c_index_mean:.3f} ± {current_c_index_std:.3f}\n\n"                
                        telegram_message += f"*overperformed metrics*: {overperformed_metrics if overperformed_metrics else 'none'}\n\n"
                        telegram_message += f"*Mean C-index*: {current_c_index_mean:.3f} -> {position}° best run"
                        asyncio.run(send_telegram_message(telegram_message))
            
            for model_version, metrics in aggregated_metrics.items():
                new_row = {
                    "ID": wandb.run.id,
                    "model_name": model_name,
                    "dataset_name" : self.get_dataset_name(),
                    "model_version": model_version.split('/')[0],
                    "End Time": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                    "seed": self.config.seed,
                    "internal_val_size": self.config.data_loader.KFold.internal_val_size,
                    'modality_setting': self.config.data_loader.missing_modalities_tables.missing_mod_rate,
                    'test_scenario': eval_missing_modality_scenario if eval_missing_modality_scenario else self.config.data_loader.missing_modalities_tables.missing_mod_rate,
                }
                
                if task_type == "Treatment_Response":
                    new_row["Mean_F1-Score_AUC"] = metrics["Mean_F1-Score_AUC"]
                    

                if hasattr(self.config.model.kwargs, "input_modalities"):
                    new_row["input_modalities"] = str(self.config.model.kwargs.input_modalities).replace(" ", "")

                if new_row["model_name"].startswith('Custom_Multimodal') or new_row["model_name"].startswith('OXA_MISS'):
                    new_row['use_WSI_level_embs'] = False
                    new_row['WSI_level_embs_fusion_type'] = None
                    new_row['WSI_level_encoder_sizes'] = None
                    new_row['WSI_level_encoder_dropout'] = None
                    new_row['WSI_level_encoder_LayerNorm'] = None
                    new_row['genomics_group_name'] = self.config.model.kwargs.genomics_group_name
                    new_row['genomics_group_dropout'] = self.config.model.kwargs.genomics_group_dropout
                    new_row['cnv_group_name'] = self.config.model.kwargs.cnv_group_name
                    new_row['cnv_group_dropout'] = self.config.model.kwargs.cnv_group_dropout
                    new_row['inner_dim'] = self.config.model.kwargs.inner_dim
                    new_row['num_latent_queries'] = self.config.model.kwargs.num_latent_queries
                    new_row['wsi_dropout'] = self.config.model.kwargs.wsi_dropout
                    new_row['use_layernorm'] = self.config.model.kwargs.use_layernorm
                    new_row['dropout'] = self.config.model.kwargs.dropout

                    
                # else:
                #     # per ogni valore in kwargs, lo aggiungo
                #     for key, value in self.config.model.kwargs.items():
                #         if key not in ["input_modalities"]:
                #             new_row[key] = value

                if task_type == "Treatment_Response":
                    task_metrics = ["AUC", "F1-Score", "Accuracy"]
                else:
                    task_metrics = ["c-index"]
                                
                for metric in task_metrics:
                    for suffix in ["_mean", "_std", "_list"]:
                        key = metric+suffix
                        new_row[key] = np.round(metrics[key], 3).tolist()
                df_old_records = pd.concat([df_old_records, pd.DataFrame([new_row])], ignore_index=True)
            df_old_records.to_csv(csv_path, index=False)
                    

            if print_demo_results:
                # Stampo il df in console, solo le colonne: ID,model_name,dataset_name,model_version,modality_setting, test_scenario,c-index, c-index_mean, c-index_std, c-index_list
                # c-index_mean, c-index_std arrotondati al terzo decimale
                print("\n\n\n\n\n\n\n\n\n")
                print("Demo results:\n")
                df_old_records_filtered = df_old_records[df_old_records.ID == wandb.run.id]
                print(df_old_records_filtered[["ID", "model_name", "dataset_name", "model_version", "modality_setting", "test_scenario", "c-index_mean", "c-index_std", "c-index_list"]].round(3).to_string(index=False))
                print("\n\n\n")
            

            '''
            Demo results:

                ID model_name dataset_name                   model_version modality_setting  c-index     test_scenario  c-index_mean  c-index_std                        c-index_list
            n69bgwyx   OXA_MISS         KIRC    Lowest_Validation_Loss_Model         complete    0.770          complete         0.802        0.082  [0.792, 0.75, 0.684, 0.875, 0.909]
            n69bgwyx   OXA_MISS         KIRC Highest_Validation_Metric_Model         complete    0.805          complete         0.827        0.113    [0.75, 0.909, 1.0, 0.792, 0.684]
            n69bgwyx   OXA_MISS         KIRC                Last_Epoch_Model         complete    0.904          complete         0.958        0.084         [1.0, 0.789, 1.0, 1.0, 1.0]
            n69bgwyx   OXA_MISS         KIRC    Lowest_Validation_Loss_Model         complete    0.607      wsi_miss_100         0.691        0.175      [0.625, 0.75, 1.0, 0.579, 0.5]
            n69bgwyx   OXA_MISS         KIRC Highest_Validation_Metric_Model         complete    0.652      wsi_miss_100         0.741        0.184      [0.875, 0.579, 0.75, 0.5, 1.0]
            n69bgwyx   OXA_MISS         KIRC                Last_Epoch_Model         complete    0.834      wsi_miss_100         0.918        0.144       [1.0, 0.958, 1.0, 1.0, 0.632]
            n69bgwyx   OXA_MISS         KIRC    Lowest_Validation_Loss_Model         complete    0.749 genomics_miss_100         0.712        0.187 [0.708, 0.364, 0.875, 0.737, 0.875]
            n69bgwyx   OXA_MISS         KIRC Highest_Validation_Metric_Model         complete    0.770 genomics_miss_100         0.747        0.215   [1.0, 0.708, 0.364, 0.789, 0.875]
            n69bgwyx   OXA_MISS         KIRC                Last_Epoch_Model         complete    0.802 genomics_miss_100         0.853        0.121      [0.75, 1.0, 0.727, 1.0, 0.789]
            
            '''

    def _log_aggregated_metrics(self, aggregated_metrics, scenario, task_type):
        """Log aggregated metrics for a specific scenario"""
        for model_version, metrics in aggregated_metrics.items():
            path_prefix = f"{model_version}/Aggregated"
            if scenario != "base":
                path_prefix += f"/Missing_modalities_scenarios/{scenario}"
                
            to_log = {}
            for key, value in metrics.items():
                if key not in ["model_version", "Confusion_Matrix"]:
                    to_log[f"{path_prefix}/{key}"] = value
                    
            if "Confusion_Matrix" in metrics:
                to_log[f"{path_prefix}/Confusion_Matrix"] = wandb.Image(metrics["Confusion_Matrix"])
                
            wandb.log(to_log)
