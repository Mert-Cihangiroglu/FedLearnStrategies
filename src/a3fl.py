#import sys
#sys.path.append("../")
import time
#import wandb

import torch
from torch.utils.data import DataLoader, TensorDataset

import torchvision
from torchvision import datasets
from torchvision import datasets, transforms

from collections import defaultdict, OrderedDict
import random
import numpy as np
#from models.resnet import ResNet18, layer2module
import copy
import os
import math
from tqdm import tqdm
from utils.backdoor_util import save_img
from models import Cifar10Model, Cifar100Model, GTSRBModel

class Attacker:
    def __init__(self, triggerX, triggerY):
        self.previous_global_model = None
        self.triggerX = triggerX 
        self.triggerY = triggerY 
        self.setup()
        

    def setup(self):
        self.handcraft_rnds = 0
        self.trigger = torch.ones((1,3,32,32), requires_grad=False, device = 'cuda')*0.5
        self.mask = torch.zeros_like(self.trigger)
        self.mask[:, :, self.triggerX:self.triggerX+5, self.triggerY:self.triggerY+5] = 1
        self.mask = self.mask.cuda()
        self.trigger0 = self.trigger.clone()

    def get_model(self, dataset_name):
        if dataset_name == "CIFAR10" or dataset_name == "CINIC10" or dataset_name == "SVHN":
            return Cifar10Model()
        elif dataset_name == "CIFAR100":
            return Cifar100Model()
        elif dataset_name == "GTSRB":
            return GTSRBModel()

    def init_badnets_trigger(self):
        print('Setup baseline trigger pattern.')
        self.trigger[:, 0, :,:] = 1
        return
    
    def get_adv_model(self, model_, dl, trigger, mask):
        adv_model = copy.deepcopy(model_)
        model = copy.deepcopy(model_)
        adv_model.train()
        model.train()
        for l in list(adv_model.parameters()):
            l.requires_grad = True
        for l in list(model.parameters()):
            l.requires_grad = True
        ce_loss = torch.nn.CrossEntropyLoss()
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)
        opt = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)
        for _ in tqdm(range(5), leave=False):
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = ce_loss(outputs, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        for _ in tqdm(range(5), leave=False):
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = trigger*mask +(1-mask)*inputs
                outputs = adv_model(inputs)
                loss = ce_loss(outputs, labels)
                adv_opt.zero_grad()
                loss.backward()
                adv_opt.step()

        sim_sum = 0.
        sim_count = 0.
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        for name in dict(adv_model.named_parameters()):
            if 'conv' in name:
                sim_count += 1
                sim_sum += cos_loss(dict(adv_model.named_parameters())[name].grad.reshape(-1),\
                                    dict(model.named_parameters())[name].grad.reshape(-1))
        return adv_model, sim_sum/sim_count

    def search_trigger(self, dataset, model_dict, dl, target_class, type_, adversary_id = 0, epoch = 0):
        trigger_optim_time_start = time.time()
        K = 0
        model = self.get_model(dataset).cuda()
        model.load_state_dict(model_dict)
        model.eval()
        adv_models = []
        adv_ws = []

        def val_asr(model, dl, t, m):
            ce_loss = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
            correct = 0.
            num_data = 0.
            total_loss = 0.
            with torch.no_grad():
                for inputs, labels in dl:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    inputs = t*m +(1-m)*inputs
                    labels[:] = target_class
                    output = model(inputs)
                    loss = ce_loss(output, labels)
                    total_loss += loss
                    pred = output.data.max(1)[1] 
                    correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
                    num_data += output.size(0)
            asr = correct/num_data
            return asr, total_loss
        
        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = 0.01
        
        K = 20
        t = self.trigger.clone()
        m = self.mask.clone()
        def grad_norm(gradients):
            grad_norm = 0
            for grad in gradients:
                grad_norm += grad.detach().pow(2).sum()
            return grad_norm.sqrt()
        ga_loss_total = 0.
        normal_grad = 0.
        ga_grad = 0.
        count = 0
        trigger_optim = torch.optim.Adam([t], lr = alpha*10, weight_decay=0)
        for iter in tqdm(range(K), leave=False):
            if iter % 10 == 0:
                asr, loss = val_asr(model, dl, t, m)
            if iter % 1 == 0 and iter != 0:
                if len(adv_models)>0:
                    for adv_model in adv_models:
                        del adv_model
                adv_models = []
                adv_ws = []
                for _ in range(1):
                    adv_model, adv_w = self.get_adv_model(model, dl, t,m) 
                    adv_models.append(adv_model)
                    adv_ws.append(adv_w)
            

            for inputs, labels in dl:
                count += 1
                t.requires_grad_()
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = t*m +(1-m)*inputs
                labels[:] = target_class
                outputs = model(inputs) 
                loss = ce_loss(outputs, labels)
                
                if len(adv_models) > 0:
                    for am_idx in range(len(adv_models)):
                        adv_model = adv_models[am_idx]
                        adv_w = adv_ws[am_idx]
                        outputs = adv_model(inputs)
                        nm_loss = ce_loss(outputs, labels)
                        if loss == None:
                            loss = 0.01*adv_w*nm_loss/1
                        else:
                            loss += 0.01*adv_w*nm_loss/1
                if loss != None:
                    loss.backward()
                    normal_grad += t.grad.sum()
                    new_t = t - alpha*t.grad.sign()
                    t = new_t.detach_()
                    t = torch.clamp(t, min = -2, max = 2)
                    t.requires_grad_()
        t = t.detach()
        #print('Before Trigger: ', self.trigger)
        self.trigger = t
        self.mask = m
        #print('Trigger: ', self.trigger)
        trigger_optim_time_end = time.time()
            

    def poison_input(self, inputs, labels, target_class, eval=False):
        if eval:
            bkd_num = inputs.shape[0]
        else:
            bkd_num = int(0.25 * inputs.shape[0])
        inputs[:bkd_num] = self.trigger*self.mask + inputs[:bkd_num]*(1-self.mask)
        labels[:bkd_num] = target_class
        return inputs, labels

    def save_random_samples(self, dataloader):
        ldr_iterator = iter(dataloader)
        for i in range(5): 
            try:
                target_img, target_label = next(ldr_iterator)
                target_img, target_label = target_img.cuda(), target_label.cuda()
                target_img, _ = self.poison_input(target_img, target_label, 0, eval=False)
            except StopIteration:
                ldr_iterator = iter(self.local_dataloader)
                target_img, target_label = next(ldr_iterator)
            save_img(target_img[i], f'CIFAR10_{i}','a3fl')