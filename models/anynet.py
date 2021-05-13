from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodules import post_3dconvs,feature_extraction_conv
import sys

from seg_oprs.loss_oprs import ProbOhemCrossEntropy2d

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class BiSeNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BiSeNetHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnRelu(in_planes, 256, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        else:
            self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(256, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(64, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fm = self.conv_3x3(x)
        output = self.conv_1x1(fm)
        # if self.scale > 1:
        output = F.upsample(output, self.scale,
                                   mode='bilinear')
        return output

class AttentionRefinement(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d):
        super(AttentionRefinement, self).__init__()
        self.conv_3x3 = ConvBnRelu(in_planes, out_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes, 1, 1, 0,
                       has_bn=True, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        fm = self.conv_3x3(x)
        fm_se = self.channel_attention(fm)
        fm = fm * fm_se

        return fm

class FeatureFusion(nn.Module):
    def __init__(self, in_planes, out_planes,
                 reduction=1, norm_layer=nn.BatchNorm2d):
        super(FeatureFusion, self).__init__()
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes // reduction, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=True, has_bias=False),
            ConvBnRelu(out_planes // reduction, out_planes, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.channel_attention(fm)
        output = fm + fm * fm_se
        return output

class AnyNet(nn.Module):
    def __init__(self, args):
        super(AnyNet, self).__init__()

        self.init_channels = args.init_channels
        self.maxdisplist = args.maxdisplist
        self.spn_init_channels = args.spn_init_channels
        self.nblocks = args.nblocks
        self.layers_3d = args.layers_3d
        self.channels_3d = args.channels_3d
        self.growth_rate = args.growth_rate
        self.with_spn = args.with_spn
        self.is_training = args.is_training

        self.ffm = FeatureFusion((4+2) * args.init_channels, (4+2) * args.init_channels, 1, norm_layer=nn.BatchNorm2d)

        self.refines = nn.ModuleList([ConvBnRelu((16 + 8) * args.init_channels, 8 * args.init_channels, 3, 1, 1,
                              has_bn=True, norm_layer=nn.BatchNorm2d,
                              has_relu=True, has_bias=False),
                   ConvBnRelu((4 + 8) * args.init_channels, 4 * args.init_channels, 3, 1, 1,
                              has_bn=True, norm_layer=nn.BatchNorm2d,
                              has_relu=True, has_bias=False)])

        conv_channel = 80
        out_planes = args.seg_classes # num_classes
        norm_layer = nn.BatchNorm2d

        up_scale = (1,1)
        if self.is_training:
            up_scale = (256, 512)
        else:
            up_scale = (368, 1232)

        self.seg_head = nn.ModuleList([BiSeNetHead(8 * args.init_channels, out_planes, up_scale,
                             True, norm_layer),
                 BiSeNetHead(4 * args.init_channels, out_planes, up_scale,
                             True, norm_layer),
                 BiSeNetHead((4 + 2) * args.init_channels, out_planes, up_scale,
                             False, norm_layer)])

        self.semantic_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.6,
                                                    min_kept=int (args.train_bsize // 1 * 256 * 512 // 8),
                                                    use_weight=False)

        if self.with_spn:
            try:
                # from .spn.modules.gaterecurrent2dnoind import GateRecurrent2dnoind
                from .spn_t1.modules.gaterecurrent2dnoind import GateRecurrent2dnoind
            except:
                print('Cannot load spn model')
                sys.exit()
            self.spn_layer = GateRecurrent2dnoind(True,False)
            spnC = self.spn_init_channels
            self.refine_spn = [nn.Sequential(
                nn.Conv2d(3, spnC*2, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(spnC*2, spnC*2, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(spnC*2, spnC*2, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(spnC*2, spnC*3, 3, 1, 1, bias=False),
            )]
            self.refine_spn += [nn.Conv2d(1,spnC,3,1,1,bias=False)]
            self.refine_spn += [nn.Conv2d(spnC,1,3,1,1,bias=False)]
            self.refine_spn = nn.ModuleList(self.refine_spn)
        else:
            self.refine_spn = None

        self.feature_extraction = feature_extraction_conv(self.init_channels,
                                      self.nblocks)

        self.volume_postprocess = []

        for i in range(3):
            net3d = post_3dconvs(self.layers_3d, self.channels_3d*self.growth_rate[i])
            self.volume_postprocess.append(net3d)
        self.volume_postprocess = nn.ModuleList(self.volume_postprocess)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()



    def warp(self, x, disp):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W, device='cuda').view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device='cuda').view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        vgrid = torch.cat((xx, yy), 1).float()

        # vgrid = Variable(grid)
        vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        return output


    def _build_volume_2d(self, feat_l, feat_r, maxdisp, stride=1):
        assert maxdisp % stride == 0  # Assume maxdisp is multiple of stride
        cost = torch.zeros((feat_l.size()[0], maxdisp//stride, feat_l.size()[2], feat_l.size()[3]), device='cuda')
        for i in range(0, maxdisp, stride):
            cost[:, i//stride, :, :i] = feat_l[:, :, :, :i].abs().sum(1)
            if i > 0:
                cost[:, i//stride, :, i:] = torch.norm(feat_l[:, :, :, i:] - feat_r[:, :, :, :-i], 1, 1)
            else:
                cost[:, i//stride, :, i:] = torch.norm(feat_l[:, :, :, :] - feat_r[:, :, :, :], 1, 1)

        return cost.contiguous()

    def _build_volume_2d3(self, feat_l, feat_r, maxdisp, disp, stride=1):
        size = feat_l.size()
        batch_disp = disp[:,None,:,:,:].repeat(1, maxdisp*2-1, 1, 1, 1).view(-1,1,size[-2], size[-1])
        batch_shift = torch.arange(-maxdisp+1, maxdisp, device='cuda').repeat(size[0])[:,None,None,None] * stride
        batch_disp = batch_disp - batch_shift.float()
        batch_feat_l = feat_l[:,None,:,:,:].repeat(1,maxdisp*2-1, 1, 1, 1).view(-1,size[-3],size[-2], size[-1])
        batch_feat_r = feat_r[:,None,:,:,:].repeat(1,maxdisp*2-1, 1, 1, 1).view(-1,size[-3],size[-2], size[-1])
        cost = torch.norm(batch_feat_l - self.warp(batch_feat_r, batch_disp), 1, 1)
        cost = cost.view(size[0],-1, size[2],size[3])
        return cost.contiguous()

    def pred_seg(self, feat_l):
        # if self.is_training:
        #     loss0 = self.seg_head[0](feat_l[-1])
        #     loss1 = self.seg_head[1](feat_l[-1])
        #     loss2 = self.seg_head[2](feat_l[-1])
        #     return loss0 + loss1 + loss2
        # else:
        feat_upsample = F.upsample(feat_l[0], (feat_l[1].size(2), feat_l[1].size(3)),
                   mode = 'bilinear')

        feat_input = self.refines[0](torch.cat([feat_upsample, feat_l[1]], 1))
        pred_out = [self.seg_head[0](feat_input)]

        feat_upsample = F.upsample(feat_input, (feat_l[2].size(2), feat_l[2].size(3)),
                   mode = 'bilinear')
        feat_input = self.refines[1](torch.cat([feat_upsample, feat_l[2]], 1))
        pred_out.append(self.seg_head[1](feat_input))

        feat_upsample = F.upsample(feat_input, (feat_l[3].size(2), feat_l[3].size(3)),
                   mode = 'bilinear')
        feat_input = self.ffm(feat_upsample, feat_l[3])
        pred_out.append(self.seg_head[2](feat_input))
        return pred_out


    def forward(self, left, right, semantic):

        img_size = left.size()

        feats_l = self.feature_extraction(left)
        feats_r = self.feature_extraction(right)
        # for i in feats_l:
        #     print(i.shape)

        pred = []
        for scale in range(len(feats_l) - 1):
            if scale > 0:
                wflow = F.upsample(pred[scale-1], (feats_l[scale + 1].size(2), feats_l[scale + 1].size(3)),
                                   mode='bilinear') * feats_l[scale + 1].size(2) / img_size[2]
                cost = self._build_volume_2d3(feats_l[scale + 1], feats_r[scale + 1],
                                         self.maxdisplist[scale], wflow, stride=1)
            else:
                cost = self._build_volume_2d(feats_l[scale + 1], feats_r[scale + 1],
                                             self.maxdisplist[scale], stride=1)

            cost = torch.unsqueeze(cost, 1)
            cost = self.volume_postprocess[scale](cost)
            cost = cost.squeeze(1)
            if scale == 0:
                pred_low_res = disparityregression2(0, self.maxdisplist[0])(F.softmax(-cost, dim=1))
                pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(2)
                disp_up = F.upsample(pred_low_res, (img_size[2], img_size[3]), mode='bilinear')
                pred.append(disp_up)
            else:
                pred_low_res = disparityregression2(-self.maxdisplist[scale]+1, self.maxdisplist[scale], stride=1)(F.softmax(-cost, dim=1))
                pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(2)
                disp_up = F.upsample(pred_low_res, (img_size[2], img_size[3]), mode='bilinear')
                pred.append(disp_up+pred[scale-1])

        if self.refine_spn:
            spn_out = self.refine_spn[0](nn.functional.upsample(left, (img_size[2]//4, img_size[3]//4), mode='bilinear'))
            G1, G2, G3 = spn_out[:,:self.spn_init_channels,:,:], spn_out[:,self.spn_init_channels:self.spn_init_channels*2,:,:], spn_out[:,self.spn_init_channels*2:,:,:]
            sum_abs = G1.abs() + G2.abs() + G3.abs()
            G1 = torch.div(G1, sum_abs + 1e-8)
            G2 = torch.div(G2, sum_abs + 1e-8)
            G3 = torch.div(G3, sum_abs + 1e-8)
            pred_flow = nn.functional.upsample(pred[-1], (img_size[2]//4, img_size[3]//4), mode='bilinear')
            refine_flow = self.spn_layer(self.refine_spn[1](pred_flow), G1, G2, G3)
            refine_flow = self.refine_spn[2](refine_flow)
            pred.append(nn.functional.upsample(refine_flow, (img_size[2] , img_size[3]), mode='bilinear'))

        if self.is_training:
            pred_seg_out = self.pred_seg(feats_l)
            aux_loss = self.semantic_criterion(pred_seg_out[1], semantic)
            main_loss = self.semantic_criterion(pred_seg_out[-1], semantic)
            pred.append(0.25 * aux_loss + main_loss)
        else:
            pred_seg_out = self.pred_seg(feats_l)
            pred.append(pred_seg_out[-1].argmax(dim=1))
        return pred

class disparityregression2(nn.Module):
    def __init__(self, start, end, stride=1):
        super(disparityregression2, self).__init__()
        self.disp = torch.arange(start*stride, end*stride, stride, device='cuda', requires_grad=False).view(1, -1, 1, 1).float()

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1, keepdim=True)
        return out
