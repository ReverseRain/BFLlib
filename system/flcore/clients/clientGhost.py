import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientGhost(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.old_head=copy.deepcopy(self.model.head)

        self.opt_head = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate)
        self.opt_base = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        self.other_client_update=torch.zeros_like(self.get_head_parameters(self.model.head))

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        self.old_head=copy.deepcopy(self.model.head)

        # Update the feature extractor first
        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = False

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                features = self.model.base(x)
                output = self.model.head(features)

                bias_col = torch.ones((features.size(0), 1), device=features.device, dtype=features.dtype)
                features_padded = torch.cat([features, bias_col], dim=1)
                

                loss = self.loss(output, y) + 0.3*torch.sum(torch.norm(features_padded @ self.other_client_update.T,dim=-1))
                self.opt_base.zero_grad()
                loss.backward()
                self.opt_base.step()

        # Update the cliassification head
        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

        for epoch in range(max_local_epochs):
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
                self.opt_head.zero_grad()
                loss.backward()
                self.opt_head.step()

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        old_head = self.get_head_parameters(self.old_head)
        new_head = self.get_head_parameters(model.head)
        local_head = self.get_head_parameters(self.model.head)

        self.other_client_update = ((new_head-old_head) - (local_head-old_head)*self.weight)
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def get_head_parameters(self,model):
        with torch.no_grad():
            W = model.weight          # (out, in)
            b = model.bias            # (out,)

            W_bias = torch.cat([W, b.unsqueeze(1)], dim=1)
            return W_bias
