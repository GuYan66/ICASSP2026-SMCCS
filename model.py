import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from resnet import resnet50, resnet18, Discrimination
import numpy as np
import math
import matplotlib.pyplot as plt

class DG_module(nn.Module):
    def __init__(self, channel, reduction=16):
        super(DG_module, self).__init__()

        self.FC11 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC11.apply(weights_init_kaiming)
        self.FC12 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC12.apply(weights_init_kaiming)
        self.FC13 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC13.apply(weights_init_kaiming)
        self.FC1 = nn.Conv2d(channel // 4, channel, kernel_size=1)
        self.FC1.apply(weights_init_kaiming)

        self.FC21 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC21.apply(weights_init_kaiming)
        self.FC22 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC22.apply(weights_init_kaiming)
        self.FC23 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC23.apply(weights_init_kaiming)
        self.FC2 = nn.Conv2d(channel // 4, channel, kernel_size=1)
        self.FC2.apply(weights_init_kaiming)

        self.FC31 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC31.apply(weights_init_kaiming)
        self.FC32 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC32.apply(weights_init_kaiming)
        self.FC33 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC33.apply(weights_init_kaiming)
        self.FC3 = nn.Conv2d(channel // 4, channel, kernel_size=1)
        self.FC3.apply(weights_init_kaiming)

        self.FC41 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC41.apply(weights_init_kaiming)
        self.FC42 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC42.apply(weights_init_kaiming)
        self.FC43 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC43.apply(weights_init_kaiming)
        self.FC4 = nn.Conv2d(channel // 4, channel, kernel_size=1)
        self.FC4.apply(weights_init_kaiming)

        self.FC51 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC51.apply(weights_init_kaiming)
        self.FC52 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC52.apply(weights_init_kaiming)
        self.FC53 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC53.apply(weights_init_kaiming)
        self.FC5 = nn.Conv2d(channel // 4, channel, kernel_size=1)
        self.FC5.apply(weights_init_kaiming)

        self.dropout = nn.Dropout(p=0.01)

    def forward(self, x):
        x1 = (self.FC11(x) + self.FC12(x) + self.FC13(x)) / 3
        x1 = self.FC1(F.relu(x1))
        x2 = (self.FC21(x) + self.FC22(x) + self.FC23(x)) / 3
        x2 = self.FC2(F.relu(x2))
        x3 = (self.FC31(x) + self.FC32(x) + self.FC33(x)) / 3
        x3 = self.FC3(F.relu(x3))
        x4 = (self.FC41(x) + self.FC42(x) + self.FC43(x)) / 3
        x4 = self.FC4(F.relu(x4))
        x5 = (self.FC51(x) + self.FC52(x) + self.FC53(x)) / 3
        x5 = self.FC5(F.relu(x5))
        out = torch.cat((x, x1, x2, x3, x4, x5), 0)
        out = self.dropout(out)
        return out

class SAI(nn.Module):
    def __init__(self, feature_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert feature_dim % num_heads == 0, f"Feature dimension {feature_dim} must be divisible by the number of attention heads {num_heads}"

        self.ir2vis_attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
        self.vis2ir_attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, ir_feat, vis_feat):
        B, C, H, W = ir_feat.shape
        
        ir_seq = ir_feat.flatten(2).permute(0, 2, 1)
        vis_seq = vis_feat.flatten(2).permute(0, 2, 1)
        
        ir_attended, _ = self.ir2vis_attn(
            query=ir_seq,
            key=vis_seq,
            value=vis_seq
        )
        ir_attended = ir_attended.permute(0, 2, 1).view(B, C, H, W)
        
        vis_attended, _ = self.vis2ir_attn(
            query=vis_seq,
            key=ir_seq,
            value=ir_seq
        )
        vis_attended = vis_attended.permute(0, 2, 1).view(B, C, H, W)
        
        gate = self.spatial_gate(ir_feat + vis_feat)
    
        ir_final = gate * ir_attended + (1 - gate) * ir_feat
        vis_final = gate * vis_attended + (1 - gate) * vis_feat
        
        return ir_final, vis_final


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class att_resnet(nn.Module):
    def __init__(self, class_num, arch='resnet50'):
        super(att_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
        self.SA = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.classifier = ClassBlock(2048, class_num)

    def forward(self, x):
        f = self.base.layer4(x)
        x = torch.mul(x, self.sigmoid(torch.mean(f, dim=1, keepdim=True)))
        f = torch.squeeze(self.base.avgpool(f))
        out, feat = self.classifier(f)
        return x, out, feat


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        
        x = self.base.layer2(x)
        
        x = self.base.layer3(x)
    
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        f = x
        x = self.classifier(x)
        return x, f


class classifier(nn.Module):
    def __init__(self, num_part, class_num):
        super(classifier, self).__init__()
        input_dim = 1024
        self.part = num_part
        self.l2norm = Normalize(2)
        for i in range(num_part):
            name = 'classifier_' + str(i)
            setattr(self, name, ClassBlock(input_dim, class_num))

    def forward(self, x, feat_all, out_all):
        start_point = len(feat_all)
        for i in range(self.part):
            name = 'classifier_' + str(i)
            cls_part = getattr(self, name)
            out_all[i + start_point], feat_all[i + start_point] = cls_part(torch.squeeze(x[:, :, i]))
            feat_all[i + start_point] = self.l2norm(feat_all[i + start_point])

        return feat_all, out_all


class embed_net(nn.Module):
    def __init__(self, class_num, part, arch='resnet50'):
        super(embed_net, self).__init__()
        self.DG = DG_module(1024)
        self.part = part
        self.base_resnet = base_resnet(arch=arch)
        
        with torch.no_grad():
            dummy_input = torch.rand(2, 3, 384, 192)
            dummy_output = self.base_resnet(dummy_input)
            self.feature_dim = dummy_output.size(1)
        
        num_heads = 8
        if self.feature_dim % num_heads != 0:
            num_heads = self.find_divisible_head(self.feature_dim)
            print(f"Adjusting the number of attention heads: {num_heads}")

        
        self.sai = SAI(self.feature_dim, num_heads=num_heads)

        self.att_v = att_resnet(class_num)
        self.att_n = att_resnet(class_num)
        self.classifier = classifier(part, class_num)
        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

        
        self.D_shared_pseu = Discrimination()
        self.D_special = Discrimination()
        self.id_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def find_divisible_head(self, dim):
        for num_heads in [8, 4, 2, 1]:
            if dim % num_heads == 0:
                return num_heads
        return 1

        
    def forward(self, x1, x2, modal=0, cam_ids=None):
        if cam_ids is not None:
            sub = (cam_ids == 1)
            sub = sub.float().cuda()
            sub_nb = sub
        else:
            sub_nb = torch.zeros(x1.size(0) * 2).cuda()

        if modal == 0:
            x = torch.cat((x1, x2), 0)
            x = self.base_resnet(x)
            
            sh_f = x

            x1, x2 = torch.chunk(x, 2, 0)
            x1, x2 = self.sai(x1, x2)
            
            x1, out_v, feat_v = self.att_v(x1)
            x2, out_n, feat_n = self.att_n(x2)
            x = torch.cat((x1, x2), 0)
            
            sp_f=x

            if sub_nb.dim() > 1:
                sub_nb = sub_nb.squeeze()
            sub_nb = sub_nb.long()
            sp_logits = self.D_special(sp_f)
            
            unad_loss_b = self.id_loss(sp_logits.float(), sub_nb.long())
            unad_loss = unad_loss_b

            pseu_sh_logits = self.D_shared_pseu(sh_f)
            
            p_sub = sub_nb.chunk(2)[0].repeat_interleave(2)
            pp_sub = torch.roll(p_sub, -1)
            pseu_loss = self.id_loss(pseu_sh_logits.float(), pp_sub)
            self.unad_loss = unad_loss
            self.pseu_loss = pseu_loss
            
            feat_globe = torch.cat((feat_v, feat_n), 0)
            out_globe = torch.cat((out_v, out_n), 0)

        elif modal == 1:
            x = self.base_resnet(x1)
            x, _, _ = self.att_v(x)
        elif modal == 2:
            x = self.base_resnet(x2)
            x, _, _ = self.att_n(x)
        x_=x.detach()
        x_zq=self.DG(self.avgpool2(x_))
        xp=self.avgpool2(x_zq)
        feat_zq = xp.view(xp.size(0), xp.size(1))
        x = self.avgpool(x)

        feat = {}
        out = {}
        feat, out = self.classifier(x, feat, out)

        if self.training:
            return feat_zq,feat, out, feat_globe, out_globe
        else:
            for i in range(self.part):
                if i == 0:
                    featf = feat[i]
                else:
                    featf = torch.cat((featf, feat[i]), 1)
            return featf
