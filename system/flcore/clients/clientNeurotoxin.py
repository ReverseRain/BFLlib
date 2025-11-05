import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientNeurotoxin(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.old_model=copy.deepcopy(self.model)

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        self.old_model=copy.deepcopy(self.model)

        for epoch in range(max_local_epochs):
            gm = torch.cat([p.view(-1) for p in self.model.parameters()], dim=0).detach()
            k = int(len(self.download_gradient) * 1e-2)
            print("k = ",k)
            _, topk_indices = torch.topk(self.download_gradient.abs(), k)

            S = torch.zeros_like(self.download_gradient, dtype=torch.bool)
            S[topk_indices] = True  
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
                gi = torch.cat([p.view(-1) for p in self.model.parameters()], dim=0).detach()
                gi=gi-gm

                
                gi[S] = 0.0
                gm=gm+gi
                # gm=gi
                
                self.overwrite_grad(self.model.parameters,gm)

            
        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        old_model = torch.cat([p.view(-1) for p in self.old_model.parameters()], dim=0).detach()
        new_model = torch.cat([p.view(-1) for p in model.parameters()], dim=0).detach()
        local_model = torch.cat([p.view(-1) for p in self.model.parameters()], dim=0).detach()

        self.download_gradient = (new_model-old_model) - (local_model-old_model)*self.weight
        
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
    
    def overwrite_grad(self,pp, newgrad):
        pointer=0
        for param in pp():
            num_params = param.numel()
            
            param_data = newgrad[pointer : pointer + num_params].view_as(param.data)
            param.data=(param_data)

            pointer += num_params
