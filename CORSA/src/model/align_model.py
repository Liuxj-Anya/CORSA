import copy
import math
import sys
import torch
from torch import nn, reshape
from torch.nn import CrossEntropyLoss
from src.model.align_modeling_utils import BertSelfEncoder, BertCrossEncoder_AttnMap, BertPooler, BertLayerNorm
import torch.nn.functional as F
from transformers import  RobertaModel, AutoConfig

import logging
logger = logging.getLogger(__name__)


class Coarse2Fine(nn.Module):
    def __init__(self,roberta_name='roberta-base',img_feat_dim=2048,roi_num=100):
        super().__init__()
        self.img_feat_dim = img_feat_dim
        config = AutoConfig.from_pretrained(roberta_name)
        self.hidden_dim = config.hidden_size

        self.roberta = RobertaModel.from_pretrained(roberta_name)
        self.sent_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.feat_linear1 = nn.Linear(2048, self.hidden_dim)
        self.feat_linear2 = nn.Linear(2048, self.hidden_dim)
        self.feat_linear3 = nn.Linear(1024, self.hidden_dim)

        self.img_self_attn1 = BertSelfEncoder(config, layer_num=1) #图像transformer
        self.v2t1=BertCrossEncoder_AttnMap(config, layer_num=1) #交叉注意力（q:image,K,V:text）
        self.img_self_attn2 = BertSelfEncoder(config, layer_num=1)  # 图像transformer
        self.v2t2 = BertCrossEncoder_AttnMap(config, layer_num=1)  # 交叉注意力（q:image,K,V:text）
        self.img_self_attn3 = BertSelfEncoder(config, layer_num=1)  # 图像transformer
        self.v2t3 = BertCrossEncoder_AttnMap(config, layer_num=1)  # 交叉注意力（q:image,K,V:text）
        
        self.dropout1_1=nn.Dropout(0.1)
        self.dropout2_1 = nn.Dropout(0.1)

        self.dropout1_2 = nn.Dropout(0.1)
        self.dropout2_2 = nn.Dropout(0.2)

        self.dropout1_3 = nn.Dropout(0.1)
        self.dropout2_3 = nn.Dropout(0.2)

        #self.gather=nn.Linear(self.hidden_dim,1)
        self.gather1 = nn.Linear(2048, 1)
        self.gather2 = nn.Linear(2048, 1)
        self.gather3 = nn.Linear(2048, 1)

        self.pred1=nn.Linear(49,2) #相关性预测
        self.pred2 = nn.Linear(196, 2)  # 相关性预测
        self.pred3 = nn.Linear(784, 2)  # 相关性预测
        self.ce_loss=nn.CrossEntropyLoss()
        self.imgto2048_1=nn.Linear(768,2048)
        self.imgto2048_2 = nn.Linear(768, 2048)
        self.imgto2048_3 = nn.Linear(768, 2048)

        self.init_weight()
    

    def init_weight(self):
        ''' bert init
        '''
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)) and ('roberta' not in name ): #linear/embedding
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, BertLayerNorm) and ('roberta' not in name ):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None and ('roberta' not in name ):
                module.bias.data.zero_()


    def forward(self,input_ids, input_mask, img_feat,relation_label,pred_loss_ratio=1.):
        # input_ids,input_mask : [N, L]
        #             img_feat : [N, 100, 2048]

        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        batch_size, seq = input_ids.size()
        #_, roi_num, feat_dim = img_feat.size()  # =100


        # text feature
        roberta_output=self.roberta(input_ids,input_mask)
        sentence_output = roberta_output[0]
        #text_pooled_output = roberta_output[1]

        def v_v_t1(img_feat_,image_mask):
            # visual self Attention
            extended_image_mask = image_mask.unsqueeze(1).unsqueeze(2)
            extended_image_mask = extended_image_mask.to(dtype=next(self.parameters()).dtype)
            extended_image_mask = (1.0 - extended_image_mask) * -10000.0
            visual_output = self.img_self_attn1(img_feat_, extended_image_mask)          #image self atttention
            visual_output = visual_output[-1]  # [N*n, 100, 768]
       
        
            # visual query sentence :
            extended_sent_mask = input_mask.unsqueeze(1).unsqueeze(2)
            extended_sent_mask = extended_sent_mask.to(dtype=next(self.parameters()).dtype)
            extended_sent_mask = (1.0 - extended_sent_mask) * -10000.0
            sentence_aware_image,_=self.v2t1(visual_output,
                                            sentence_output,
                                            extended_sent_mask,
                                            output_all_encoded_layers=False)  # image query sentence
            sentence_aware_image=sentence_aware_image[-1]  #[N,100,768]
            sentence_aware_image=self.imgto2048_1(sentence_aware_image)

            return sentence_aware_image

        def v_v_t2(img_feat_, image_mask):
            # visual self Attention
            extended_image_mask = image_mask.unsqueeze(1).unsqueeze(2)
            extended_image_mask = extended_image_mask.to(dtype=next(self.parameters()).dtype)
            extended_image_mask = (1.0 - extended_image_mask) * -10000.0
            visual_output = self.img_self_attn2(img_feat_, extended_image_mask)  # image self atttention
            visual_output = visual_output[-1]  # [N*n, 100, 768]

            # visual query sentence :
            extended_sent_mask = input_mask.unsqueeze(1).unsqueeze(2)
            extended_sent_mask = extended_sent_mask.to(dtype=next(self.parameters()).dtype)
            extended_sent_mask = (1.0 - extended_sent_mask) * -10000.0
            sentence_aware_image, _ = self.v2t2(visual_output,
                                                sentence_output,
                                                extended_sent_mask,
                                                output_all_encoded_layers=False)  # image query sentence
            sentence_aware_image = sentence_aware_image[-1]  # [N,100,768]
            sentence_aware_image = self.imgto2048_2(sentence_aware_image)

            return sentence_aware_image

        def v_v_t3(img_feat_, image_mask):
            # visual self Attention
            extended_image_mask = image_mask.unsqueeze(1).unsqueeze(2)
            extended_image_mask = extended_image_mask.to(dtype=next(self.parameters()).dtype)
            extended_image_mask = (1.0 - extended_image_mask) * -10000.0
            visual_output = self.img_self_attn3(img_feat_, extended_image_mask)  # image self atttention
            visual_output = visual_output[-1]  # [N*n, 100, 768]

            # visual query sentence :
            extended_sent_mask = input_mask.unsqueeze(1).unsqueeze(2)
            extended_sent_mask = extended_sent_mask.to(dtype=next(self.parameters()).dtype)
            extended_sent_mask = (1.0 - extended_sent_mask) * -10000.0
            sentence_aware_image, _ = self.v2t3(visual_output,
                                                sentence_output,
                                                extended_sent_mask,
                                                output_all_encoded_layers=False)  # image query sentence
            sentence_aware_image = sentence_aware_image[-1]  # [N,100,768]
            sentence_aware_image = self.imgto2048_3(sentence_aware_image)

            return sentence_aware_image

        #feature1
        img_feat_0 = img_feat[0].view((-1,2048,49))
        img_feat_0 = img_feat_0.permute(0,2,1)
        img_feat_0 = self.feat_linear1(img_feat_0)  # [N*n, 100, 2048] ->[N*n, 100, 768]
        image_mask0 = torch.ones((batch_size, 49)).to(device)
        sentence_aware_image1=v_v_t1(img_feat_0,image_mask0)

        # feature2
        img_feat_1 = img_feat[1].view((-1, 2048, 196))
        img_feat_1 = img_feat_1.permute(0, 2, 1)
        img_feat_1 = self.feat_linear2(img_feat_1)  # [N*n, 100, 2048] ->[N*n, 100, 768]
        image_mask1 = torch.ones((batch_size, 196)).to(device)
        sentence_aware_image2 = v_v_t2(img_feat_1, image_mask1)

        # feature3
        img_feat_2 = img_feat[2].view((-1, 1024,784))
        img_feat_2 = img_feat_2.permute(0, 2, 1)
        img_feat_2 = self.feat_linear3(img_feat_2)  # [N*n, 100, 2048] ->[N*n, 100, 768]
        image_mask2 = torch.ones((batch_size, 784)).to(device)
        sentence_aware_image3 = v_v_t3(img_feat_2, image_mask2)

        # feature1
        gathered_sentence_aware_image1=self.gather1(self.dropout1_1(
                                                        sentence_aware_image1)).squeeze(2) #[N,100,768]->[N,100,1] ->[N,100]
        rel_pred1=self.pred1(self.dropout2_1(gathered_sentence_aware_image1)) #  [N,2]
        gate1=(torch.softmax(rel_pred1,dim=-1)[:,1].unsqueeze(1).expand(
                                    batch_size,49).unsqueeze(2).expand(batch_size,49,2048)) #更改
        gated_sentence_aware_image1 = gate1 * sentence_aware_image1
        pred_loss1=self.ce_loss(rel_pred1,relation_label.long())

        # feature2
        gathered_sentence_aware_image2 = self.gather2(self.dropout1_2(
                                    sentence_aware_image2)).squeeze(2)  # [N,100,768]->[N,100,1] ->[N,100]
        rel_pred2 = self.pred2(self.dropout2_2(gathered_sentence_aware_image2))  # [N,2]
        gate2 = (torch.softmax(rel_pred2, dim=-1)[:, 1].unsqueeze(1).expand(
            batch_size, 196).unsqueeze(2).expand(batch_size, 196, 2048))  # 更改
        gated_sentence_aware_image2 = gate2 * sentence_aware_image2
        pred_loss2 = self.ce_loss(rel_pred2, relation_label.long())

        # feature3
        gathered_sentence_aware_image3 = self.gather3(self.dropout1_3(
            sentence_aware_image3)).squeeze(2)  # [N,100,768]->[N,100,1] ->[N,100]
        rel_pred3 = self.pred3(self.dropout2_3(gathered_sentence_aware_image3))  # [N,2]
        gate3 = (torch.softmax(rel_pred3, dim=-1)[:, 1].unsqueeze(1).expand(
            batch_size, 784).unsqueeze(2).expand(batch_size, 784, 2048))  # 更改
        gated_sentence_aware_image3 = gate3 * sentence_aware_image3
        pred_loss3 = self.ce_loss(rel_pred3, relation_label.long())

        pred_loss=(pred_loss1+pred_loss2+pred_loss3)/6
        return pred_loss_ratio*pred_loss,gated_sentence_aware_image1,gated_sentence_aware_image2,gated_sentence_aware_image3
