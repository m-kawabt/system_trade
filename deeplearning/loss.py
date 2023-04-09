import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLoss(nn.Module):
    def __init__(self, device):
        super(MyLoss, self).__init__()
        self.device = device

    def forward(self, predictions, targets):
        """
        predictions : shape [batch_size, 2, 2]
        targets : shape [batch_size, 1]
        """
        # batch_size = predictions.size(0)
        trade_or_notrade_p = predictions[:,0]
        is_trade_mask = torch.abs(targets).gt(0.5).to(device=self.device)
        trade_or_notrade_t = torch.where(is_trade_mask, 1, 0).to(device=self.device)
        is_trade_loss = F.cross_entropy(trade_or_notrade_p, trade_or_notrade_t)

        long_or_short_p = predictions[:,1]

        long_or_short_t = torch.where(targets[is_trade_mask].gt(0), 0, 1)
        longshort_loss = F.cross_entropy(long_or_short_p[is_trade_mask], long_or_short_t)
        
        return is_trade_loss, longshort_loss