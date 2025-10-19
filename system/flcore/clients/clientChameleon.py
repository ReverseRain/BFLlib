import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientChameleon(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.opt_head = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate)
        self.opt_base = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        supcon_loss=SupConLoss()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

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
                feature = self.model.base(x)
                loss = supcon_loss(feature, y,
                                        fac_label=self.poison_flag)
                self.opt_base.zero_grad()
                loss.backward()
                self.opt_base.step()

        # self.model.cpu()
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
        
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: 
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=5):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None, fac_label=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            
            mask_scale = mask.clone().detach()
            mask_cross_feature = torch.ones_like(mask_scale).to(device)
            
            for ind, label in enumerate(labels.view(-1)):
                if label == fac_label:
                    mask_scale[ind, :] = mask[ind, :]

        else:
            mask = mask.float().to(device)

        contrast_feature = features
        anchor_feature = features
        # if self.contrast_mode == 'one':
        #     anchor_feature = features[:, 0]
        # elif self.contrast_mode == 'all':
        #     anchor_feature = features
        # else:
        #     raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) * mask_cross_feature 
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask
        mask_scale = mask_scale * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(1e-12+exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos_mask = (mask_scale * log_prob).sum(1)
        mask_check = mask.sum(1)
        for ind, mask_item in enumerate(mask_check):
            if mask_item == 0:
                continue
            else:
                mask_check[ind] = 1 / mask_item
        mask_apply = mask_check
        mean_log_prob_pos = mean_log_prob_pos_mask * mask_apply
        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(batch_size).mean()

        return loss
