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
import numpy as np


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


class RateLoss(nn.Module):
    def __init__(self):
        super(RateLoss, self).__init__()
        
    def forward(self, x, hparams, WSOLA, 
                model_saliency, rate_distribution, 
                mask_sample, intent_cats, 
                additional_criterion, uniform=False):
        if np.random.rand() <= hparams.exploitation_prob:
            index = torch.argmax(rate_distribution, dim=-1) #exploit
        else:
            # index = torch.multinomial(rate_distribution, 1) #explore using predictive distribution
            if uniform:
                index = torch.multinomial(torch.ones((rate_distribution.shape[1])), 
                                        x.shape[0], 
                                        replacement=True)
                index = index.to("cuda")
            else:
                index = torch.multinomial(rate_distribution, 1, 
                                          replacement=True) #explore
            

        rate = 0.5 + 0.1*index #0.2*index
        mod_speech, mod_e, _ = WSOLA(mask=mask_sample[:,:,0], 
                                     rate=rate, speech=x)
    
        mod_speech = mod_speech.to("cuda")
        mod_e = mod_e.to("cuda")
        _, _, mod_mask, mod_saliency = model_saliency(mod_speech, mod_e)
        
        ## directly maximize score of intended index
        # intent_saliency_indices = torch.argmax(intent_saliency, dim=-1)
        # loss_rate_l1 = -1 * mod_saliency.gather(1,intent_saliency_indices.view(-1,1)).view(-1)
        loss_rate_l1 = 1 - mod_saliency.gather(1,intent_cats.view(-1,1)).view(-1)
        
        ## Minimizing loss on intended saliency
        # loss_rate_l1 = torch.sum(torch.abs(mod_saliency - intent_saliency), dim=-1)

        corresp_probs = rate_distribution.gather(1,index.view(-1,1)).view(-1)
        log_corresp_prob = torch.log(corresp_probs)
        unbiased_multiplier = torch.mul(corresp_probs.detach(), log_corresp_prob)
        loss_rate_l1 = torch.mean(torch.mul(loss_rate_l1.detach(), 
                                            unbiased_multiplier))
        loss_rate_ent = -1*additional_criterion(rate_distribution)
        loss_rate = loss_rate_l1 + hparams.lambda_entropy * loss_rate_ent
        return loss_rate


class PitchRateLoss(nn.Module):
    def __init__(self):
        super(PitchRateLoss, self).__init__()
        
    def forward(self, x, hparams, WSOLA, OLA, 
                model_saliency, rate_distribution, 
                pitch_distribution, mask_sample, 
                intent_cats, additional_criterion, 
                uniform=False):
        if np.random.rand() <= hparams.exploitation_prob:
            index_rate = torch.argmax(rate_distribution, dim=-1) #exploit
            index_pitch = torch.argmax(pitch_distribution, dim=-1) #exploit
        else:
            # index = torch.multinomial(rate_distribution, 1) #explore using predictive distribution
            if uniform:
                index_rate = torch.multinomial(torch.ones((rate_distribution.shape[1])), 
                                        x.shape[0], 
                                        replacement=True)
                index_rate = index_rate.to("cuda")
                index_pitch = torch.multinomial(torch.ones((pitch_distribution.shape[1])), 
                                        x.shape[0], 
                                        replacement=True)
                index_pitch = index_pitch.to("cuda")
            else:
                index_rate = torch.multinomial(rate_distribution, 1, 
                                          replacement=True) #explore
                index_pitch = torch.multinomial(pitch_distribution, 1, 
                                          replacement=True) #explore
            

        rate = 0.5 + 0.1*index_rate
        pitch = 0.5 + 0.1*index_pitch
        
        pitch_mod_speech = OLA(factor=pitch, speech=x)
        mod_speech, mod_e, _ = WSOLA(mask=mask_sample[:,:,0], 
                                     rate=rate, speech=pitch_mod_speech)

        mod_speech = mod_speech.to("cuda")
        mod_e = mod_e.to("cuda")
        _, _, mod_mask, mod_saliency = model_saliency(mod_speech, mod_e)
        
        ## directly maximize score of intended index
        # loss_l1 = 1 - mod_saliency.gather(1,intent_cats.view(-1,1)).view(-1)
        
        ## Minimizing loss on intended saliency
        intent_saliency = nn.functional.one_hot(intent_cats, num_classes=5).to("cuda")
        loss_l1 = torch.sum(torch.abs(mod_saliency - intent_saliency), dim=-1)

        corresp_probs_rate = rate_distribution.gather(1,index_rate.view(-1,1)).view(-1)
        corresp_probs_pitch = pitch_distribution.gather(1,index_pitch.view(-1,1)).view(-1)
        
        log_corresp_prob_rate = torch.log(corresp_probs_rate)
        log_corresp_prob_pitch = torch.log(corresp_probs_pitch)
        
        unbiased_multiplier_rate = torch.mul(corresp_probs_rate.detach(), log_corresp_prob_rate)
        unbiased_multiplier_pitch = torch.mul(corresp_probs_pitch.detach(), log_corresp_prob_pitch)
        
        loss_saliency = torch.mean(torch.mul(loss_l1.detach(), 
                                            unbiased_multiplier_rate))
        loss_saliency += torch.mean(torch.mul(loss_l1.detach(), 
                                            unbiased_multiplier_pitch))
        
        loss_ent = -1*additional_criterion(rate_distribution) + -1*additional_criterion(pitch_distribution)
        loss = loss_saliency + hparams.lambda_entropy * loss_ent
        return loss


class BlockPitchRateLoss(nn.Module):
    def __init__(self):
        super(BlockPitchRateLoss, self).__init__()
        
    def forward(self, x, hparams, WSOLA, OLA, 
                model_saliency, rate_distribution, 
                pitch_distribution, mask_sample, 
                intent_cats, additional_criterion, 
                uniform=False):
        if np.random.rand() <= hparams.exploitation_prob:
            index_rate = torch.argmax(rate_distribution, dim=-1) #exploit
            index_pitch = torch.argmax(pitch_distribution, dim=-1) #exploit
        else:
            # index = torch.multinomial(rate_distribution, 1) #explore using predictive distribution
            if uniform:
                index_rate = torch.multinomial(torch.ones((rate_distribution.shape[1])), 
                                        x.shape[0], 
                                        replacement=True)
                index_rate = index_rate.to("cuda")
                index_pitch = torch.multinomial(torch.ones((pitch_distribution.shape[1])), 
                                        x.shape[0], 
                                        replacement=True)
                index_pitch = index_pitch.to("cuda")
            else:
                index_rate = torch.multinomial(rate_distribution, 1, 
                                          replacement=True) #explore
                index_pitch = torch.multinomial(pitch_distribution, 1, 
                                          replacement=True) #explore
            

        rate = 0.25 + 0.15*index_rate
        # pitch = 0.25 + 0.15*index_pitch
        # pitch = 0.5 + 0.125*index_pitch
        pitch = 0.5 + 0.1*index_pitch
        
        pitch_mod_speech = OLA(mask=mask_sample[:,:,0], 
                               factor=pitch, speech=x)
        mod_speech, mod_e, _ = WSOLA(mask=mask_sample[:,:,0], 
                                     rate=rate, speech=pitch_mod_speech)

        mod_speech = mod_speech.to("cuda")
        mod_e = mod_e.to("cuda")
        _, _, mod_mask, mod_saliency = model_saliency(mod_speech, mod_e)
        
        ## directly maximize score of intended index
        # loss_l1 = 1 - mod_saliency.gather(1,intent_cats.view(-1,1)).view(-1)
        
        ## Minimizing loss on intended saliency
        intent_saliency = nn.functional.one_hot(intent_cats, num_classes=5).to("cuda")
        loss_l1 = torch.sum(torch.abs(mod_saliency - intent_saliency), dim=-1)

        corresp_probs_rate = rate_distribution.gather(1,index_rate.view(-1,1)).view(-1)
        corresp_probs_pitch = pitch_distribution.gather(1,index_pitch.view(-1,1)).view(-1)
        
        log_corresp_prob_rate = torch.log(corresp_probs_rate)
        log_corresp_prob_pitch = torch.log(corresp_probs_pitch)
        
        unbiased_multiplier_rate = torch.mul(corresp_probs_rate.detach(), log_corresp_prob_rate)
        unbiased_multiplier_pitch = torch.mul(corresp_probs_pitch.detach(), log_corresp_prob_pitch)
        
        loss_saliency = torch.mean(torch.mul(loss_l1.detach(), 
                                            unbiased_multiplier_rate))
        loss_saliency += torch.mean(torch.mul(loss_l1.detach(), 
                                            unbiased_multiplier_pitch))
        
        loss_ent = -1*additional_criterion(rate_distribution) + -1*additional_criterion(pitch_distribution)
        loss = loss_saliency + hparams.lambda_entropy*loss_ent
        return loss


class EnergyPitchRateLoss(nn.Module):
    def __init__(self):
        super(EnergyPitchRateLoss, self).__init__()
        
    def forward(self, x, hparams, WSOLA, MOLA, 
                model_saliency, rate_distribution, 
                pitch_distribution, energy_distribution, 
                mask_sample, intent_cats, additional_criterion, 
                uniform=False):
        if np.random.rand() <= hparams.exploitation_prob:
            index_rate = torch.argmax(rate_distribution, dim=-1) #exploit
            index_pitch = torch.argmax(pitch_distribution, dim=-1) #exploit
            index_energy = torch.argmax(energy_distribution, dim=-1) #exploit
        else:
            # index = torch.multinomial(rate_distribution, 1) #explore using predictive distribution
            if uniform:
                index_rate = torch.multinomial(torch.ones((rate_distribution.shape[1])), 
                                        x.shape[0], 
                                        replacement=True)
                index_rate = index_rate.to("cuda")
                
                index_pitch = torch.multinomial(torch.ones((pitch_distribution.shape[1])), 
                                        x.shape[0], 
                                        replacement=True)
                index_pitch = index_pitch.to("cuda")
                
                index_energy = torch.multinomial(torch.ones((energy_distribution.shape[1])), 
                                        x.shape[0], 
                                        replacement=True)
                index_energy = index_energy.to("cuda")
            else:
                index_rate = torch.multinomial(rate_distribution, 1, 
                                          replacement=True) #explore
                index_pitch = torch.multinomial(pitch_distribution, 1, 
                                          replacement=True) #explore
                index_energy = torch.multinomial(energy_distribution, 1, 
                                          replacement=True) #explore
            

        rate = 0.5 + 0.1*index_rate
        pitch = 0.5 + 0.1*index_pitch
        energy = 0.5 + 0.1*index_energy
        
        energy_pitch_mod_speech = MOLA(factor_pitch=pitch, 
                                       factor_energy=energy, 
                                       speech=x)
        mod_speech, mod_e, _ = WSOLA(mask=mask_sample[:,:,0], rate=rate, 
                                     speech=energy_pitch_mod_speech)
    
        mod_speech = mod_speech.to("cuda")
        mod_e = mod_e.to("cuda")
        _, _, mod_mask, mod_saliency = model_saliency(mod_speech, mod_e)
        
        ## directly maximize score of intended index
        loss_l1 = 1 - mod_saliency.gather(1,intent_cats.view(-1,1)).view(-1)
        
        ## Minimizing loss on intended saliency
        corresp_probs_rate = rate_distribution.gather(1,index_rate.view(-1,1)).view(-1)
        corresp_probs_pitch = pitch_distribution.gather(1,index_pitch.view(-1,1)).view(-1)
        corresp_probs_energy = energy_distribution.gather(1,index_energy.view(-1,1)).view(-1)
        
        log_corresp_prob_rate = torch.log(corresp_probs_rate)
        log_corresp_prob_pitch = torch.log(corresp_probs_pitch)
        log_corresp_prob_energy = torch.log(corresp_probs_energy)
        
        unbiased_multiplier_rate = torch.mul(corresp_probs_rate.detach(), log_corresp_prob_rate)
        unbiased_multiplier_pitch = torch.mul(corresp_probs_pitch.detach(), log_corresp_prob_pitch)
        unbiased_multiplier_energy = torch.mul(corresp_probs_energy.detach(), log_corresp_prob_energy)
        
        loss_saliency = torch.mean(torch.mul(loss_l1.detach(), 
                                            unbiased_multiplier_rate))
        loss_saliency += torch.mean(torch.mul(loss_l1.detach(), 
                                            unbiased_multiplier_pitch))
        loss_saliency += torch.mean(torch.mul(loss_l1.detach(), 
                                            unbiased_multiplier_energy))
        
        loss_ent = (-1*additional_criterion(rate_distribution) 
                    + -1*additional_criterion(pitch_distribution) 
                    + -1*additional_criterion(energy_distribution))
        loss = loss_saliency + hparams.lambda_entropy * loss_ent
        return loss


class PitchRateLossCS(nn.Module):
    def __init__(self):
        super(PitchRateLossCS, self).__init__()
        
    def forward(self, x, hparams, WSOLA, OLA, 
                model_saliency, rate_distribution, 
                pitch_distribution, mask_sample, 
                intent_cats, intent_saliency, 
                additional_criterion, uniform=False):
        if np.random.rand() <= hparams.exploitation_prob:
            index_rate = torch.argmax(rate_distribution, dim=-1) #exploit
            index_pitch = torch.argmax(pitch_distribution, dim=-1) #exploit
        else:
            # index = torch.multinomial(rate_distribution, 1) #explore using predictive distribution
            if uniform:
                index_rate = torch.multinomial(torch.ones((rate_distribution.shape[1])), 
                                        x.shape[0], 
                                        replacement=True)
                index_rate = index_rate.to("cuda")
                index_pitch = torch.multinomial(torch.ones((pitch_distribution.shape[1])), 
                                        x.shape[0], 
                                        replacement=True)
                index_pitch = index_pitch.to("cuda")
            else:
                index_rate = torch.multinomial(rate_distribution, 1, 
                                          replacement=True) #explore
                index_pitch = torch.multinomial(pitch_distribution, 1, 
                                          replacement=True) #explore
            

        rate = 0.5 + 0.1*index_rate
        pitch = 0.5 + 0.1*index_pitch
        
        pitch_mod_speech = OLA(factor=pitch, speech=x)
        mod_speech, mod_e, _ = WSOLA(mask=mask_sample[:,:,0], 
                                     rate=rate, speech=pitch_mod_speech)
        
    
        mod_speech = mod_speech.to("cuda")
        mod_e = mod_e.to("cuda")
        (_, _, 
        mod_mask, mod_saliency) = model_saliency(mod_speech, mod_e, 
                                                intent_saliency.unsqueeze(1).to("cuda"))
        
        ## directly maximize score of intended index
        loss_rate_l1 = 1 - mod_saliency.gather(1,intent_cats.view(-1,1)).view(-1)
        
        ## Minimizing loss on intended saliency

        corresp_probs_rate = rate_distribution.gather(1,index_rate.view(-1,1)).view(-1)
        corresp_probs_pitch = pitch_distribution.gather(1,index_pitch.view(-1,1)).view(-1)

        log_corresp_prob_rate = torch.log(corresp_probs_rate)
        log_corresp_prob_pitch = torch.log(corresp_probs_pitch)
        
        unbiased_multiplier_rate = torch.mul(corresp_probs_rate.detach(), log_corresp_prob_rate)
        unbiased_multiplier_pitch = torch.mul(corresp_probs_pitch.detach(), log_corresp_prob_pitch)
        
        loss_rate_saliency = torch.mean(torch.mul(loss_rate_l1.detach(), 
                                            unbiased_multiplier_rate))
        loss_rate_saliency += torch.mean(torch.mul(loss_rate_l1.detach(), 
                                            unbiased_multiplier_pitch))
        
        loss_rate_ent = -1*additional_criterion(rate_distribution) + -1*additional_criterion(pitch_distribution)
        
        loss_rate = loss_rate_saliency + hparams.lambda_entropy * loss_rate_ent
        
        return loss_rate






























































