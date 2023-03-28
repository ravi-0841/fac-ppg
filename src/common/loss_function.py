# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Modified from https://github.com/NVIDIA/tacotron2"""

import torch
from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self, mel_weight=1, gate_weight=0.005):
        super(Tacotron2Loss, self).__init__()
        self.w_mel = mel_weight
        self.w_gate = gate_weight

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return self.w_mel * mel_loss + self.w_gate * gate_loss


class SpectrogramMSELoss(nn.Module):
    def __init__(self):
        super(SpectrogramMSELoss, self).__init__()
        pass
    
    def forward(self, model_output, target):
        target.requires_grad = False
        loss = nn.MSELoss()(model_output, target)
        return loss


class SpectrogramL1Loss(nn.Module):
    def __init__(self):
        super(SpectrogramL1Loss, self).__init__()
    
    def forward(self, model_output, target):
        target.requires_grad = False
        loss = nn.L1Loss()(model_output, target)
        return loss


class TruncatedSpectrogramL1Loss(nn.Module):
    def __init__(self):
        super(TruncatedSpectrogramL1Loss, self).__init__()
    
    def forward(self, model_output, target):
        target.requires_grad = False
        loss = nn.L1Loss()(model_output, target)
        return loss


class MaskedSpectrogramL1Loss(nn.Module):
    def __init__(self):
        super(MaskedSpectrogramL1Loss, self).__init__()
    
    def forward(self, output, target, length):
        target.requires_grad = False
        loss = 0
        batch_size = output.shape[0]
        pad_seq_len = output.shape[2]
        real_imag_channel = output.shape[1]
        fft_len = output.shape[3]
        for i in range(batch_size):
            mask = torch.zeros((batch_size, real_imag_channel, pad_seq_len, fft_len))
            mask[:, :, :length[i], :] = 1
            loss += torch.sum(torch.abs(output[i] - target[i]) * mask.to("cuda"))
            loss /= torch.sum(mask)
        return loss / batch_size


class MaskedSpectrogramL2Loss(nn.Module):
    def __init__(self):
        super(MaskedSpectrogramL1Loss, self).__init__()
    
    def forward(self, output, target, length):
        target.requires_grad = False
        loss = 0
        batch_size = output.shape[0]
        pad_seq_len = output.shape[2]
        real_imag_channel = output.shape[1]
        fft_len = output.shape[3]
        for i in range(batch_size):
            mask = torch.zeros((batch_size, real_imag_channel, pad_seq_len, fft_len))
            mask[:, :, :length[i], :] = 1
            loss += torch.sum(torch.abs(output[i] - target[i]) * mask.to("cuda"))
            loss /= torch.sum(mask)
        return loss / batch_size


class MaskedSpectrogramL1LossReduced(nn.Module):
    def __init__(self):
        super(MaskedSpectrogramL1LossReduced, self).__init__()
    
    def forward(self, output, target, length):
        target.requires_grad = False
        loss = 0
        batch_size = output.shape[0]
        fft_len = output.shape[1]
        pad_seq_len = output.shape[2]
        for i in range(batch_size):
            mask = torch.zeros((batch_size, fft_len, pad_seq_len))
            mask[:, :, :length[i]] = 1
            loss += torch.sum(torch.abs(output[i] - target[i]) * mask.to("cuda"))
            loss /= torch.sum(mask)
        return loss / batch_size


class MaskedSpectrogramL2LossReduced(nn.Module):
    def __init__(self):
        super(MaskedSpectrogramL2LossReduced, self).__init__()
    
    def forward(self, output, target, length):
        target.requires_grad = False
        loss = 0
        batch_size = output.shape[0]
        fft_len = output.shape[1]
        pad_seq_len = output.shape[2]
        for i in range(batch_size):
            mask = torch.zeros((batch_size, fft_len, pad_seq_len))
            mask[:, :, :length[i]] = 1
            loss += torch.sum(torch.square(output[i] - target[i]) * mask.to("cuda"))
            loss /= torch.sum(mask)
        return loss / batch_size


# class ExpectedKLDivergence(nn.Module):
#     def __init__(self, alpha=0.1, beta=0.8):
#         super(ExpectedKLDivergence, self).__init__()
        
#         self.alpha = torch.tensor(alpha).float()
#         self.beta = torch.tensor(beta).float()
    
#     def compute_expected_KL(self, past_posterior, curr_posterior):
#         divergence = past_posterior * (curr_posterior*(torch.log(curr_posterior) - torch.log(self.beta)) + (1.0 - curr_posterior)*(torch.log(1.0 - curr_posterior) - torch.log(1.0 - self.beta)))
#         divergence += (1.0 - past_posterior) * (curr_posterior*(torch.log(curr_posterior) - torch.log(1.0 - self.beta)) + (1.0 - curr_posterior)*(torch.log(1.0 - curr_posterior) - torch.log(self.beta)))
#         return divergence
    
#     def forward(self, posterior, length):
#         # Expect posterior to be of shape [batch_size, padded_seq_length]
#         # posterior += 1e-10
#         total_loss = 0
#         (batch_len, _) = (posterior.shape[0], posterior.shape[1])
#         for b in range(batch_len):
#             actual_seq_len = length[b]
#             loss = posterior[b,0] * (torch.log(posterior[b,0]) - torch.log(self.alpha)) + (1.0 - posterior[b,0]) * (torch.log(1.0 - posterior[b,0]) - torch.log(1.0 - self.alpha))
#             for s in range(1, actual_seq_len):
#                 loss += self.compute_expected_KL(posterior[b,s-1], posterior[b,s])
            
#             total_loss += loss

#         total_loss = total_loss / batch_len
#         return total_loss


class ExpectedKLDivergence(nn.Module):
    def __init__(self, alpha=0.1, beta=0.9): #0.8
        super(ExpectedKLDivergence, self).__init__()
        
        self.alpha = torch.tensor(alpha).float()
        self.beta = torch.tensor(beta).float()
    
    def compute_expected_KL(self, past_posterior, curr_posterior):
        divergence = past_posterior[1]*(curr_posterior[1]*(torch.log(curr_posterior[1]) - torch.log(self.beta)) + curr_posterior[0]*(torch.log(curr_posterior[0]) - torch.log(1.0 - self.beta)))
        divergence += past_posterior[0]*(curr_posterior[1]*(torch.log(curr_posterior[1]) - torch.log(1.0 - self.beta)) + curr_posterior[0]*(torch.log(curr_posterior[0]) - torch.log(self.beta)))
        return divergence
    
    def forward(self, posterior, length):
        # Expect posterior to be of shape [batch_size, padded_seq_length, 2]
        # posterior += 1e-10
        total_loss = 0
        (batch_len, _) = (posterior.shape[0], posterior.shape[1])
        for b in range(batch_len):
            actual_seq_len = length[b]
            loss = posterior[b,0,1]*(torch.log(posterior[b,0,1]) - torch.log(self.alpha)) + posterior[b,0,0]*(torch.log(posterior[b,0,0]) - torch.log(1.0 - self.alpha))
            for s in range(1, actual_seq_len):
                loss += self.compute_expected_KL(posterior[b,s-1], posterior[b,s])
            
            total_loss += loss

        total_loss = total_loss / batch_len
        return total_loss


class SparsityKLDivergence(nn.Module):
    def __init__(self, sparse_prob=0.01):
        super(SparsityKLDivergence, self).__init__()
        
        self.sparse_prob = torch.tensor(sparse_prob).float()
    
    def forward(self, posterior):
        loss = torch.sum(posterior[:,:,0]*torch.log(posterior[:,:,0]/(1 - self.sparse_prob)) + posterior[:,:,1]*torch.log(posterior[:,:,1]/self.sparse_prob), dim=1)
        return torch.mean(loss)


class VecExpectedKLDivergence(nn.Module):
    def __init__(self, alpha=0.1, beta=0.95):
        super(VecExpectedKLDivergence, self).__init__()

        self.alpha = torch.tensor(alpha).float()
        self.beta = torch.tensor(beta).float()

    def compute_vectorized_KL(self, past_posterior, curr_posterior):
        divergence = past_posterior[:,1]*(curr_posterior[:,1]*(torch.log(curr_posterior[:,1]) - torch.log(self.beta)) + curr_posterior[:,0]*(torch.log(curr_posterior[:,0]) - torch.log(1.0 - self.beta)))
        divergence += past_posterior[:,0]*(curr_posterior[:,1]*(torch.log(curr_posterior[:,1]) - torch.log(1.0 - self.beta)) + curr_posterior[:,0]*(torch.log(curr_posterior[:,0]) - torch.log(self.beta)))
        return torch.sum(divergence)

    def forward(self, posterior, length):
        # Expect posterior to be of the shape [batch_size, padded_seq_length, 2]
        total_loss = 0
        batch_len = posterior.shape[0]
        for b in range(batch_len):
            real_seq_len = length[b]
            loss = posterior[b,0,1]*(torch.log(posterior[b,0,1]) - torch.log(self.alpha)) + posterior[b,0,0]*(torch.log(posterior[b,0,0]) - torch.log(1.0 - self.alpha))
            loss += self.compute_vectorized_KL(posterior[b,0:real_seq_len-1,:], posterior[b,1:real_seq_len,:])
            total_loss += loss

        total_loss /= batch_len
        return total_loss


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()
    
    def forward(self, distribution):
        # distribution -> [batch, n_categories]
        loss = -1*torch.sum(torch.mul(distribution, 
                                      torch.log2(distribution + 1e-15)), dim=-1)
        return torch.mean(loss)


































