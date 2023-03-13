import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from mmseg.ops import resize
from mmcv.cnn import ConvModule
from .decode_head import BaseDecodeHead
from ..builder import HEADS
from ..utils import SequenceConv
import math

'''
    NOTE: CrossFrameAttention aims to exploiting similarity from other views to compensate central view.
          CrossFrameAttention is one part of implicit branch for Implicit Feature Integration.
'''
class CrossFrameAttention(nn.Module):

    def __init__(self,
                 matmul_norm=False):
        super(CrossFrameAttention, self).__init__()
        self.matmul_norm = matmul_norm
        self.radius = 0.1       # inconsistent search region reflected by disparity
        self.weight = 0.2       # additional weight for inconsistent region

    def forward(self, memory_keys, memory_values, query_query, disparity, sequence_index):
        sai_number, batch_size, key_channels, height, width = memory_keys.shape
        _, _, value_channels, _, _ = memory_values.shape
        assert query_query.shape[1] == key_channels
        memory_keys = memory_keys.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW
        memory_keys = memory_keys.view(batch_size, key_channels, sai_number * height * width)  # BxCxT*H*W

        query_query = query_query.view(batch_size, key_channels, height * width).permute(0, 2, 1).contiguous()  # BxH*WxCk
        
        # generate attention mask for region with large disparity
        # sequence_index: B * 1 * sai_number-1 * 2
        # disparity: B * 1 * H * W 
        weight = self.weight / sai_number / height / width
        total_atten_mask = []
        for i in range(sai_number):    
            atten_mask = torch.zeros([batch_size,1,height,width]).cuda()   # B * 1 * H * W
            distance = torch.sqrt((sequence_index[:,i,1]-5)**2 + (sequence_index[:,i,0]-5)**2) 
            total_disparity = distance.reshape(batch_size,1,1,1) * disparity      # B * 1 * H * W
            
            atten_mask[total_disparity > self.radius] = weight    
            atten_mask[total_disparity < -1 * self.radius] = weight     
            total_atten_mask.append(atten_mask)
            
        total_atten_mask = torch.stack(total_atten_mask,2)    # B * 1 * T * H * W
        total_atten_mask = total_atten_mask.view(batch_size,1,sai_number*height*width)  # B * 1 * THW
        
        key_attention = torch.bmm(query_query, memory_keys)  # BxH*WxT*H*W
        if self.matmul_norm:
            key_attention = (key_channels ** -.5) * key_attention
        key_attention = F.softmax(key_attention, dim=-1)   # BxH*WxT*H*W     
        key_attention = key_attention + total_atten_mask
        memory_values = memory_values.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW
        memory_values = memory_values.view(batch_size, value_channels, sai_number * height * width)
        memory_values = memory_values.permute(0, 2, 1).contiguous()  # BxT*H*WxC
        memory = torch.bmm(key_attention, memory_values)  # BxH*WxC
        memory = memory.permute(0, 2, 1).contiguous()  # BxCxH*W
        memory = memory.view(batch_size, value_channels, height, width)  # BxCxHxW
        return memory
        
'''
    NOTE: SelfFrameAttention is designed to gather context information within central view.
          SelfFrameAttention is another part of implicit branch for Implicit Feature Integration.
'''
class SelfFrameAttention(nn.Module):

    def __init__(self,
                 matmul_norm=False):
        super(SelfFrameAttention, self).__init__()
        self.matmul_norm = matmul_norm

    def forward(self, query_query, query_key, query_value):

        batch_size, key_channels, height, width = query_query.shape
        _, value_channels, _, _ = query_value.shape
        query_query = query_query.view(batch_size, key_channels, height * width).permute(0, 2, 1).contiguous()  # BxH*WxCk
        query_key = query_key.view(batch_size, key_channels, height * width)  # BxCkxH*W
        key_attention = torch.bmm(query_query, query_key)  # BxH*WxH*W
        if self.matmul_norm:
            key_attention = (key_channels ** -.5) * key_attention
        key_attention = F.softmax(key_attention, dim=-1)  # BxH*WxH*W
        
        query_value = query_value.view(batch_size, value_channels, height * width).permute(0, 2, 1).contiguous()  # BxH*WxCv
        memory = torch.bmm(key_attention, query_value)  # BxH*WxCv
        memory = memory.permute(0, 2, 1).contiguous()  # BxCvxH*W
        memory = memory.view(batch_size, value_channels, height, width)  # BxCxHxW
        return memory

'''
    NOTE: SELayer is channel attention operation for Explicit Feature Propagation.
'''
class SELayer(nn.Module):
    def __init__(self,out_ch,g=16):
        super(SELayer,self).__init__()
        self.attn = nn.Sequential(
                nn.Conv2d(out_ch,out_ch//g,1,1,0),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch//g,out_ch,1,1,0),
                nn.Sigmoid()
            )
        
    def forward(self,feature):
        x = F.adaptive_avg_pool2d(feature,(1,1))
        attn = self.attn(x)
        feature = feature * attn
        return feature
    
@HEADS.register_module()
class LF_IENET_HR(BaseDecodeHead):
    def __init__(self, sai_number, key_channels, value_channels, **kwargs):
        super(LF_IENET_HR, self).__init__(**kwargs)
        self.sai_number = sai_number
        self.warp_channel = 512
        
        self.reference_encoding = nn.Sequential(
            SequenceConv(self.in_channels, value_channels, 1, sai_number-1,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(value_channels, value_channels, 3, sai_number-1,
                         self.conv_cfg, self.norm_cfg, self.act_cfg,stride=2)
        )
        
        self.reference_key_conv = nn.Sequential(
            SequenceConv(value_channels, key_channels, 1, sai_number-1,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(key_channels, key_channels, 3, sai_number-1,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)
        )
        self.reference_value_conv = nn.Sequential(
            SequenceConv(value_channels, value_channels, 1, sai_number-1,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(value_channels, value_channels, 3, sai_number-1,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)
        )
        
        self.warp_feature_conv = nn.Sequential(
            SequenceConv(self.in_channels, self.warp_channel, 1, sai_number-1,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),
            SequenceConv(self.warp_channel, self.warp_channel, 3, sai_number-1,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)
        )
        
        self.center_encoding = nn.Sequential(
            ConvModule(
                self.in_channels,
                value_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                value_channels,
                value_channels,
                3,
                padding=1,
                stride=2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )
        
        self.center_query_conv = nn.Sequential(
            ConvModule(
                value_channels,
                key_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                key_channels,
                key_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )
        
        self.center_key_conv = nn.Sequential(
            ConvModule(
                value_channels,
                key_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                key_channels,
                key_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )
        
        
        self.center_value_conv = nn.Sequential(
            ConvModule(
                value_channels,
                value_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                value_channels,
                value_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )
        self.cross_attention = CrossFrameAttention(matmul_norm=False)
        self.self_attention = SelfFrameAttention(matmul_norm=False)
        
        self.remove_noise = nn.Sequential(
            ConvModule(
                1,
                64,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                64,
                64,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                64,
                64,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                64,
                64,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                64,
                1,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )
        
        self.channel_attention = SELayer(self.warp_channel*(sai_number-1))
        self.fuse_warp_conv = nn.Sequential(
            ConvModule(
                self.warp_channel*(sai_number-1),
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )
        
        self.bottleneck_1 = ConvModule(
            value_channels * 2,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        
        self.bottleneck_2 = ConvModule(
            self.channels * 2,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )


    def forward(self, inputs, sequence_imgs, disparity, sequence_index):
        # step1 : get center view feature (x) and reference view features (sequence_imgs)
        x = self._transform_inputs(inputs)
        sequence_imgs = [self._transform_inputs(inputs).unsqueeze(0) for inputs in sequence_imgs]  # T, BxCxHxW
        sequence_imgs = torch.cat(sequence_imgs, dim=0)  # TxBxCxHxW
        sai_number, batch_size, channels, height, width = sequence_imgs.shape

        assert sai_number == self.sai_number - 1
        
        # step2: calculate key/value matrix of reference view features
        reference_encoding_feature = self.reference_encoding(sequence_imgs)
        reference_key = self.reference_key_conv(reference_encoding_feature)
        reference_value = self.reference_value_conv(reference_encoding_feature)
        
        # step2: calculate query/key/value matrix of center view feature
        center_encoding_feature = self.center_encoding(x)
        center_query = self.center_query_conv(center_encoding_feature)  # BxCxHxW
        center_key = self.center_key_conv(center_encoding_feature)  # BxCxHxW
        center_value = self.center_value_conv(center_encoding_feature)  # BxCxHxW

        if len(sequence_index.size()) == 4:
            sequence_index = sequence_index.squeeze(1)  # batch * 1 * sai_number-1 * 2 -> batch * sai_number-1 * 2
        # step3: improve precision of estimated disparity
        disparity = self.remove_noise(disparity) + disparity
        
        
        disparity_4 = resize(
                disparity,
                size=x.shape[2:],
                mode='nearest',
                warning=False)          
        disparity_4 = disparity_4 / 4      
        
        disparity_8 = resize(
                disparity,
                size=center_key.shape[2:],
                mode='nearest',
                warning=False)         
        disparity_8 = disparity_8 / 8       
        
        # step4: calculate cross attention and self attention for Implicit Feature Integration
        cross_feature = self.cross_attention(reference_key, reference_value, center_query, disparity_8, sequence_index) + center_encoding_feature
        self_feature = self.self_attention(center_query, center_key, center_value) + center_encoding_feature
        output = torch.cat([self_feature, cross_feature], dim=1)
        
        # step5: warp each reference feature to central view based on estimated disparity 
        _, _, height, width = disparity_4.shape
        disparity_4 = disparity_4.permute(0, 2, 3, 1).contiguous()      #  batch * H * W * 1
        
        
        x = np.array([i for i in range(0, height)]).reshape(1, height, 1, 1).repeat(repeats=width, axis=2)
        y = np.array([i for i in range(0, width)]).reshape(1, 1, width, 1).repeat(repeats=height, axis=1)
        xy_position = torch.from_numpy(np.concatenate([x, y], axis=-1))     
        
        coords_x = torch.linspace(-1, 1, width).to(torch.float32)       
        coords_y = torch.linspace(-1, 1, height).to(torch.float32)       
        coords_x = coords_x.repeat(height, 1).reshape(height, width, 1)          
        coords_y = coords_y.repeat(width, 1).permute(1, 0).reshape(height, width, 1)
        coords = torch.cat([coords_x, coords_y], dim=2)             
        coords = coords.reshape(1, height, width, 2)         
        coords = coords[0, xy_position[:, :, :, 0].reshape(-1).to(torch.int64),xy_position[:, :, :, 1].reshape(-1).to(torch.int64), :]   
        coords = coords.reshape(-1, height, width, 2).cuda()                     
        coords = coords.repeat(batch_size,1,1,1)
        
        warp_features = []
        
        dst_u = 5   
        dst_v = 5   
        
        sequence_imgs = self.warp_feature_conv(sequence_imgs)
        
        for i in range(sai_number):                      
            current_feature = sequence_imgs[i]          
            offsetx = 2 * (dst_u - sequence_index[:,i,1]).reshape(batch_size,1,1,1) * disparity_4[:, :, :, :]    
            offsety = 2 * (dst_v - sequence_index[:,i,0]).reshape(batch_size,1,1,1) * disparity_4[:, :, :, :]    
            coords_x = (coords[:, :, :, 0:1] * width + offsetx) / width     
            coords_y = (coords[:, :, :, 1:2] * height + offsety) / height    
            coords_uv = torch.cat([coords_x, coords_y], dim=-1)    
            temp = F.grid_sample(current_feature, coords_uv[:, :, :, :],mode='bilinear',padding_mode='border')    
            warp_features.append(temp)
        
        # step6: calculate channel attention for Explicit Feature Propagation 
        warp_feature = torch.cat(warp_features,1)   # batch * (channel*sai_number-1) * H * W
        warp_feature_attention = self.channel_attention(warp_feature)
        warp_feature_final = self.fuse_warp_conv(warp_feature_attention)

        # step7: Concat the output of implicit branch and explicit branch to produce the final feature of central view for final predicted result
        output = self.bottleneck_1(output)      
        output = resize(
                        input=output,
                        size=warp_feature_final.shape[2:],
                        mode='bilinear',
                        align_corners=self.align_corners)
        
        output = self.bottleneck_2(torch.cat([warp_feature_final, output], dim=1))      
        output = self.cls_seg(output)
        return output,warp_features
