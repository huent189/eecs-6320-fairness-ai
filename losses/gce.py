import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

        
class GeneralizedCELoss(nn.Module):
    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = p * targets + (1 - p) * (1 - targets)
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach()**self.q)*self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')
        loss = F.binary_cross_entropy(p, targets, reduction='none') * loss_weight
        
        return loss