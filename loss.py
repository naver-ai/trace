# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class TRACELoss(nn.Module):
    def __init__(self, neg_pos_ratio):
        super(TRACELoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, predictions, targets, weights):
        """TRACE Loss
        Args:
            predictions (tuple): A tuple containing textness preds, 8-directional link preds.
                cls shape: torch.size(batch_size,height,width,2)
                link shape: torch.size(batch_size,height,width,16)

            ground_truth (tensor): Ground truth location and 8-directional links for a batch.
                cls shape: torch.size(batch_size,height,width,1)
                link shape: torch.size(batch_size,height,width,8)
        """
        num_channel = predictions.size(3)

        """ loss for each corners """
        loss = 0
        for i in range(num_channel):
            pred_cls_batch = predictions[:, :, :, i].contiguous().view(-1, 1)
            gt_cls_batch = targets[:, :, :, i].contiguous().view(-1, 1)
            weights_batch = weights[:, :, :, i].contiguous().view(-1, 1)

            pos_idx = (gt_cls_batch > 0).data
            loss_temp = (weights_batch * F.huber_loss(pred_cls_batch, gt_cls_batch, reduction="none")).data
            loss_temp[pos_idx] = 0
            _, loss_idx = loss_temp.sort(0, descending=True)
            _, idx_rank = loss_idx.sort(0)

            # OHEM
            sel_idx, num_sel, num_pos = self.getOHEMIndex(pos_idx, self.neg_pos_ratio, idx_rank)

            # Calc loss
            loss += (
                weights_batch[sel_idx] * F.huber_loss(pred_cls_batch[sel_idx], gt_cls_batch[sel_idx], reduction="none")
            ).sum() / num_sel

        return loss

    def getOHEMIndex(self, pos_idx, neg_pos_ratio, idx_rank):
        num_pos = pos_idx.long().sum()
        num_neg = min((neg_pos_ratio * num_pos, pos_idx.size(0) - 1)) if num_pos > 0 else 10000
        neg_idx = idx_rank < num_neg
        sel_idx = (pos_idx + neg_idx).gt(0)

        return sel_idx, (num_pos + num_neg).float(), num_pos.float()
