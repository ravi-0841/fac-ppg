from typing import List, Tuple

import torch
import torch.nn as nn


class GLSTM(nn.Module):
    def __init__(
        self,
        in_features=None,
        out_features=None,
        mid_features=None,
        hidden_size=448,
        groups=2,
    ) -> None:
        super(GLSTM, self).__init__()

        hidden_size_t = hidden_size // groups

        self.lstm_list1 = nn.ModuleList(
            [
                nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True)
                for i in range(groups)
            ]
        )
        self.lstm_list2 = nn.ModuleList(
            [
                nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True)
                for i in range(groups)
            ]
        )

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.groups = groups
        self.mid_features = mid_features

    def forward(self, x) -> torch.Tensor:
        out = x
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1).contiguous()

        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.stack(
            [self.lstm_list1[i](out[i])[0] for i in range(self.groups)], dim=-1
        )
        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        out = self.ln1(out)

        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.cat(
            [self.lstm_list2[i](out[i])[0] for i in range(self.groups)], dim=-1
        )
        out = self.ln2(out)

        out = out.view(out.size(0), out.size(1), x.size(1), -1).contiguous()
        out = out.transpose(1, 2).contiguous()

        return out


class GluConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride) -> None:
        super(GluConv2d, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x) -> torch.Tensor:
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out


class GluConvTranspose2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, output_padding=(0, 0)
    ) -> None:
        super(GluConvTranspose2d, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=output_padding,
        )
        self.conv2 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=output_padding,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x) -> torch.Tensor:
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out


class GCRN_encoder(nn.Module):
    def __init__(self) -> None:
        super(GCRN_encoder, self).__init__()

        self.conv1 = GluConv2d(
            in_channels=2, out_channels=16, kernel_size=(2, 3), stride=(1, 2)
        )
        self.conv2 = GluConv2d(
            in_channels=16, out_channels=20, kernel_size=(2, 3), stride=(1, 2)
        )
        self.conv3 = GluConv2d(
            in_channels=20, out_channels=28, kernel_size=(2, 3), stride=(1, 2)
        )
        self.conv4 = GluConv2d(
            in_channels=28, out_channels=36, kernel_size=(2, 3), stride=(1, 2)
        )
        self.conv5 = GluConv2d(
            in_channels=36, out_channels=64, kernel_size=(2, 3), stride=(1, 2)
        )

        self.glstm = GLSTM(groups=2)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm2d(28)
        self.bn4 = nn.BatchNorm2d(36)
        self.bn5 = nn.BatchNorm2d(64)

        self.elu = nn.ELU(inplace=True)

    def forward(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        out = x
        e1 = self.elu(self.bn1(self.conv1(out)))
        e2 = self.elu(self.bn2(self.conv2(e1)))
        e3 = self.elu(self.bn3(self.conv3(e2)))
        e4 = self.elu(self.bn4(self.conv4(e3)))
        e5 = self.elu(self.bn5(self.conv5(e4)))

        out = e5

        out = self.glstm(out)
        return out, [e5, e4, e3, e2, e1]


class GCRN_decoder(nn.Module):
    def __init__(self) -> None:
        super(GCRN_decoder, self).__init__()

        self.glstm = GLSTM(groups=2)

        self.conv5_t_1 = GluConvTranspose2d(
            in_channels=128, out_channels=36, kernel_size=(2, 3), stride=(1, 2)
        )
        self.conv4_t_1 = GluConvTranspose2d(
            in_channels=72, out_channels=28, kernel_size=(2, 3), stride=(1, 2)
        )
        self.conv3_t_1 = GluConvTranspose2d(
            in_channels=56, out_channels=20, kernel_size=(2, 3), stride=(1, 2)
        )
        self.conv2_t_1 = GluConvTranspose2d(
            in_channels=40,
            out_channels=16,
            kernel_size=(2, 3),
            stride=(1, 2),
            output_padding=(0, 1),
        )
        self.conv1_t_1 = GluConvTranspose2d(
            in_channels=32, out_channels=1, kernel_size=(2, 3), stride=(1, 2)
        )

        self.conv5_t_2 = GluConvTranspose2d(
            in_channels=128, out_channels=36, kernel_size=(2, 3), stride=(1, 2)
        )
        self.conv4_t_2 = GluConvTranspose2d(
            in_channels=72, out_channels=28, kernel_size=(2, 3), stride=(1, 2)
        )
        self.conv3_t_2 = GluConvTranspose2d(
            in_channels=56, out_channels=20, kernel_size=(2, 3), stride=(1, 2)
        )
        self.conv2_t_2 = GluConvTranspose2d(
            in_channels=40,
            out_channels=16,
            kernel_size=(2, 3),
            stride=(1, 2),
            output_padding=(0, 1),
        )
        self.conv1_t_2 = GluConvTranspose2d(
            in_channels=32, out_channels=1, kernel_size=(2, 3), stride=(1, 2)
        )

        self.bn5_t_1 = nn.BatchNorm2d(36)
        self.bn4_t_1 = nn.BatchNorm2d(28)
        self.bn3_t_1 = nn.BatchNorm2d(20)
        self.bn2_t_1 = nn.BatchNorm2d(16)
        self.bn1_t_1 = nn.BatchNorm2d(1)

        self.bn5_t_2 = nn.BatchNorm2d(36)
        self.bn4_t_2 = nn.BatchNorm2d(28)
        self.bn3_t_2 = nn.BatchNorm2d(20)
        self.bn2_t_2 = nn.BatchNorm2d(16)
        self.bn1_t_2 = nn.BatchNorm2d(1)

        self.elu = nn.ELU(inplace=True)

        self.fc1 = nn.Linear(in_features=257, out_features=257)
        self.fc2 = nn.Linear(in_features=257, out_features=257)

    def forward(self, x, e) -> torch.Tensor:

        # e - [conv5, conv4, conv3, conv2, conv1]
        x = self.glstm(x)
        out = torch.cat((x, e[0]), dim=1)

        d5_1 = self.elu(torch.cat((self.bn5_t_1(self.conv5_t_1(out)), e[1]), dim=1))
        d4_1 = self.elu(torch.cat((self.bn4_t_1(self.conv4_t_1(d5_1)), e[2]), dim=1))
        d3_1 = self.elu(torch.cat((self.bn3_t_1(self.conv3_t_1(d4_1)), e[3]), dim=1))
        d2_1 = self.elu(torch.cat((self.bn2_t_1(self.conv2_t_1(d3_1)), e[4]), dim=1))
        d1_1 = self.elu(self.bn1_t_1(self.conv1_t_1(d2_1)))

        d5_2 = self.elu(torch.cat((self.bn5_t_2(self.conv5_t_2(out)), e[1]), dim=1))
        d4_2 = self.elu(torch.cat((self.bn4_t_2(self.conv4_t_2(d5_2)), e[2]), dim=1))
        d3_2 = self.elu(torch.cat((self.bn3_t_2(self.conv3_t_2(d4_2)), e[3]), dim=1))
        d2_2 = self.elu(torch.cat((self.bn2_t_2(self.conv2_t_2(d3_2)), e[4]), dim=1))
        d1_2 = self.elu(self.bn1_t_2(self.conv1_t_2(d2_2)))

        out1 = self.fc1(d1_1)
        out2 = self.fc2(d1_2)
        out = torch.cat([out1, out2], dim=1)

        return out


class GCRN_embed_decoder(nn.Module):
    def __init__(self) -> None:
        super(GCRN_embed_decoder, self).__init__()

        self.glstm = GLSTM(groups=2)

        self.conv5_t_1 = GluConvTranspose2d(
            in_channels=128, out_channels=36, kernel_size=(2, 3), stride=(1, 2)
        )
        self.conv4_t_1 = GluConvTranspose2d(
            in_channels=72, out_channels=28, kernel_size=(2, 3), stride=(1, 2)
        )
        self.conv3_t_1 = GluConvTranspose2d(
            in_channels=56, out_channels=20, kernel_size=(2, 3), stride=(1, 2)
        )
        self.conv2_t_1 = GluConvTranspose2d(
            in_channels=40,
            out_channels=16,
            kernel_size=(2, 3),
            stride=(1, 2),
            output_padding=(0, 1),
        )
        self.conv1_t_1 = GluConvTranspose2d(
            in_channels=32, out_channels=1, kernel_size=(2, 3), stride=(1, 2)
        )

        self.conv5_t_2 = GluConvTranspose2d(
            in_channels=128, out_channels=36, kernel_size=(2, 3), stride=(1, 2)
        )
        self.conv4_t_2 = GluConvTranspose2d(
            in_channels=72, out_channels=28, kernel_size=(2, 3), stride=(1, 2)
        )
        self.conv3_t_2 = GluConvTranspose2d(
            in_channels=56, out_channels=20, kernel_size=(2, 3), stride=(1, 2)
        )
        self.conv2_t_2 = GluConvTranspose2d(
            in_channels=40,
            out_channels=16,
            kernel_size=(2, 3),
            stride=(1, 2),
            output_padding=(0, 1),
        )
        self.conv1_t_2 = GluConvTranspose2d(
            in_channels=32, out_channels=1, kernel_size=(2, 3), stride=(1, 2)
        )

        self.bn5_t_1 = nn.BatchNorm2d(36)
        self.bn4_t_1 = nn.BatchNorm2d(28)
        self.bn3_t_1 = nn.BatchNorm2d(20)
        self.bn2_t_1 = nn.BatchNorm2d(16)
        self.bn1_t_1 = nn.BatchNorm2d(1)

        self.bn5_t_2 = nn.BatchNorm2d(36)
        self.bn4_t_2 = nn.BatchNorm2d(28)
        self.bn3_t_2 = nn.BatchNorm2d(20)
        self.bn2_t_2 = nn.BatchNorm2d(16)
        self.bn1_t_2 = nn.BatchNorm2d(1)

        self.elu = nn.ELU(inplace=True)

        self.fc1 = nn.Linear(in_features=257, out_features=257)
        self.fc2 = nn.Linear(in_features=257, out_features=257)

    def forward(self, x) -> torch.Tensor:

        x = self.glstm(x)

        out = torch.repeat_interleave(x, 2, dim=1)  # torch.cat((x, x), dim=1)
        d5_1_conv = self.bn5_t_1(self.conv5_t_1(out))
        d5_1 = self.elu(
            torch.repeat_interleave(d5_1_conv, 2, dim=1)
        )  # self.elu(torch.cat((d5_1_conv, d5_1_conv), dim=1))

        d4_1_conv = self.bn4_t_1(self.conv4_t_1(d5_1))
        d4_1 = self.elu(
            torch.repeat_interleave(d4_1_conv, 2, dim=1)
        )  # self.elu(torch.cat((d4_1_conv, d4_1_conv), dim=1))

        d3_1_conv = self.bn3_t_1(self.conv3_t_1(d4_1))
        d3_1 = self.elu(
            torch.repeat_interleave(d3_1_conv, 2, dim=1)
        )  # self.elu(torch.cat((d3_1_conv, d3_1_conv), dim=1))

        d2_1_conv = self.bn2_t_1(self.conv2_t_1(d3_1))
        d2_1 = self.elu(
            torch.repeat_interleave(d2_1_conv, 2, dim=1)
        )  # self.elu(torch.cat((d2_1_conv, d2_1_conv), dim=1))

        d1_1 = self.elu(self.bn1_t_1(self.conv1_t_1(d2_1)))

        d5_2_conv = self.bn5_t_2(self.conv5_t_2(out))
        d5_2 = self.elu(
            torch.repeat_interleave(d5_2_conv, 2, dim=1)
        )  # self.elu(torch.cat((d5_2_conv, d5_2_conv), dim=1))

        d4_2_conv = self.bn4_t_2(self.conv4_t_2(d5_2))
        d4_2 = self.elu(
            torch.repeat_interleave(d4_2_conv, 2, dim=1)
        )  # self.elu(torch.cat((d4_2_conv, d4_2_conv), dim=1))

        d3_2_conv = self.bn3_t_2(self.conv3_t_2(d4_2))
        d3_2 = self.elu(
            torch.repeat_interleave(d3_2_conv, 2, dim=1)
        )  # self.elu(torch.cat((d3_2_conv, d3_2_conv), dim=1))

        d2_2_conv = self.bn2_t_2(self.conv2_t_2(d3_2))
        d2_2 = self.elu(
            torch.repeat_interleave(d2_2_conv, 2, dim=1)
        )  # self.elu(torch.cat((d2_2_conv, d2_2_conv), dim=1))

        d1_2 = self.elu(self.bn1_t_2(self.conv1_t_2(d2_2)))

        out1 = self.fc1(d1_1)
        out2 = self.fc2(d1_2)
        out = torch.cat([out1, out2], dim=1)

        return out


class GCRN(nn.Module):
    def __init__(self, mixing_alpha=0) -> None:
        super(GCRN, self).__init__()

        self.mixing_alpha = mixing_alpha

        self.embed_projection1 = nn.Linear(in_features=448, out_features=768)
        self.embed_projection2 = nn.Linear(in_features=768, out_features=448)

        self.encoder = GCRN_encoder()
        self.decoder = GCRN_decoder()
        self.nlu = nn.Tanh()

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:

        lstm_out, conv_out = self.encoder(x)
        embed = self.nlu(
            self.embed_projection1(
                lstm_out.permute(0, 2, 1, 3).flatten(start_dim=2, end_dim=3)
            )
        )
        out = self.decoder(lstm_out, conv_out)

        return out, embed


class GCRN_embedding_decoder(nn.Module):
    def __init__(self, mixing_alpha=0) -> None:
        super(GCRN_embedding_decoder, self).__init__()

        self.mixing_alpha = mixing_alpha

        self.embed_projection1 = nn.Linear(in_features=448, out_features=768)
        self.embed_projection2 = nn.Linear(in_features=768, out_features=448)

        self.encoder = GCRN_encoder()
        self.decoder = GCRN_embed_decoder()
        self.nlu = nn.Tanh()

    def forward(self, x, input_embed) -> Tuple[torch.Tensor, torch.Tensor]:

        lstm_out, _ = self.encoder(x)
        # print("lstm_out shape: ", lstm_out.shape)
        embed = self.nlu(
            self.embed_projection1(
                lstm_out.permute(0, 2, 1, 3).flatten(start_dim=2, end_dim=3)
            )
        )

        input_embed = self.nlu(self.embed_projection2(input_embed[:, :, -1, :]))
        embed_shape = input_embed.shape
        input_embed = torch.reshape(
            input_embed, (embed_shape[0], embed_shape[1], 64, 7)
        )
        input_embed = input_embed.permute(0, 2, 1, 3)

        # print("input_embed shape: ", input_embed.shape)
        
        decoder_input = (
            self.mixing_alpha * lstm_out
            + (1 - self.mixing_alpha) * input_embed[:, :, : lstm_out.shape[2], :]
        )
        # decoder_input = input_embed

        out = self.decoder(decoder_input)

        return out, embed
