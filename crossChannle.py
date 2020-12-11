#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pdb
import torch
from torch import nn
import torch.nn.functional as F


class CrossChannleModule(nn.Module):
    def __init__(self, in_channels):
        super(CrossChannleModule, self).__init__()
	
        bottleneck = in_channels // 8 	

        self.fc_in = nn.Sequential(nn.Conv1d(in_channels, bottleneck, 1, 1, 0, bias=True), 
                                   nn.ReLU(True),
                                   nn.Conv1d(bottleneck,  in_channels,1, 1, 0, bias=True))

        self.fc_out = nn.Sequential(nn.Conv1d(in_channels,bottleneck, 1, 1, 0, bias=True),
                                    nn.ReLU(True),
                                    nn.Conv1d(bottleneck, in_channels,1, 1, 0, bias=True))

    def forward(self, x):
        '''
        param x: (bt, c, h, w)
        return:
        '''
        bt, c, h, w = x.shape
        residual = x
        x = x.view(bt, c, -1) # b x c x h*w
        x_v = x.permute(0, 2, 1).contiguous() # b x h*w x c
        x_v = self.fc_in(x_v) # b x h*w x c
        x_m = x_v.mean(1).view(-1, 1, c ) # b x 1 x c
        score = -(x_m - x_m.permute(0, 2, 1).contiguous())**2 # b x c x c
        attn = F.softmax(score, dim=1) # b x c x c

        out = self.fc_out(torch.bmm(x_v, attn)) # b x h*w x c
        out = out.permute(0, 2, 1).contiguous().view(bt, c, h, w)  # b x c x h x w
        out = F.relu(residual + out)
        return out 

if __name__ == '__main__':
    def test():
        pdb.set_trace()
        input_tensor = torch.rand(2, 8, 3, 3)
        cc = CrossChannleModule(3*3)
        out = cc(input_tensor)
    test()

