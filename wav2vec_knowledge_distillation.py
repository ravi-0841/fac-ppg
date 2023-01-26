import torch
import torch.nn as nn
import fairseq
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Config, Wav2Vec2Model

# cp_path = "/home/ravi/Desktop/Meta_Internship/models/wav2vec_small.pt"

# _, cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])

# # Pre-trained model for distillation
# state_dict = torch.load(cp_path, map_location=torch.device("cpu"))
# model = Wav2Vec2Model(cfg.model)
# model.load_state_dict(state_dict["model"])
# model = model.eval()

# data = np.random.randn(2,80000)
# #target = np.random.randn(249, 1, 768)

# data = torch.from_numpy(data).float()
# #target = torch.from_numpy(target).float().to("cuda")
# target = model(data, features_only=True)["layer_results"][-1][-1]
# target = target.detach()

# data = data.to("cuda")
# target = target.to("cuda")

# # New initialized model
# config = Wav2Vec2Config(cfg)
# model1 = Wav2Vec2Model(config)
# model1 = model1.to("cuda")
# optimizer = torch.optim.Adam(model1.parameters(), lr=0.00001)
# model1.train()
# criterion = torch.nn.L1Loss(reduction="mean")

# for i in range(100):
#     optimizer.zero_grad()
#     with torch.enable_grad():
#         feats = model1(data, features_only=True)
#         feats = feats["layer_results"][-1][-1]
#         loss = criterion(feats, target)
#     loss.backward()
#     optimizer.step()
#     print("Epoch: {}, Current loss is: {}".format(i, loss.item()))


class Wav2Vec2_encoder(nn.Module):
    
    def __init__(self, config_path, device="cuda"):
        super(Wav2Vec2_encoder, self).__init__()
        
        self.device = device
        self.config_path = config_path
        _, cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.config_path])
        self.config = Wav2Vec2Config(cfg)
        self.model = Wav2Vec2Model(self.config)
        self.model = self.model.to(self.device)
        
    def forward(self, x):
        # x -> [batch_size, #samples]
        # x = x.to(self.device)
        out = self.model(x, features_only=True)
        return out["layer_results"][-1][-1]


class Wav2Vec2_pretrained(nn.Module):
    
    def __init__(self, config_path, device="cuda"):
        super(Wav2Vec2_pretrained, self).__init__()
        
        self.device = device
        self.config_path = config_path
        _, cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.config_path])
        
        state_dict = torch.load(self.config_path, map_location=torch.device("cpu"))
        self.model = Wav2Vec2Model(cfg.model)
        self.model.load_state_dict(state_dict["model"])
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        
    def forward(self, x):
        # x -> [batch_size, #samples]
        # x = x.to(self.device)
        out = self.model(x, features_only=True)
        out = out["layer_results"][-1][-1]
        return out.detach()














