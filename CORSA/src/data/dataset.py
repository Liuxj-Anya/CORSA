import torch
import numpy as np
import json
import csv
import os
import json
import torch.utils.data as data
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms
import pdb
from transformers import RobertaTokenizer, RobertaModel
import src.data.cfg as cfg
import math
# 设置打印选项
#np.set_printoptions(threshold=np.inf)

def one_hot(cls_num, i):  # cls_num 类别数，i当前哪一个类别
    b = np.zeros(cls_num)
    b[i] = 1.
    return b

class Twitter_Dataset(data.Dataset):
    def __init__(self,img_path,infos, split):
        self.path_img = img_path    #图片文件夹路径：Twitter_data//twitter2015_images
        self.infos = json.load(open(infos, 'r'))

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.tokenizer = tokenizer

        crop_size = 416  ######

        if split == 'train':
            self.data_set = json.load(
                open(self.infos['data_dir'] + '/train.json', 'r'))  #src/data/twitter2015/train.json 训练的标注文件
            self.transform = transforms.Compose([
                transforms.Resize((crop_size, crop_size)),
                #transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        elif split == 'dev':
            self.data_set = json.load(
                open(self.infos['data_dir'] + '/dev.json', 'r'))
            self.transform = transforms.Compose([
                transforms.Resize((crop_size, crop_size)),  # args.crop_size, by default it is set to be 224
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        elif split == 'test':
            self.data_set = json.load(
                open(self.infos['data_dir'] + '/test.json', 'r'))
            self.transform = transforms.Compose([
                transforms.Resize((crop_size, crop_size)),  # args.crop_size, by default it is set to be 224
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            raise RuntimeError("split type is not exist!!!")

        self.count_img_error=0

    def __len__(self):
        return len(self.data_set)

    #获取图像特征
    def get_img_feature(self,id):
        feat_dict = np.load(id)
        img_feat = feat_dict['x']  # [2048,100]
        img_feat = img_feat.transpose((1, 0))[:49]
        #print(img_feat.shape)
        img_feat = (img_feat / np.sqrt((img_feat ** 2).sum()))
        return img_feat

    def get_img(self,id):
        image_path = os.path.join(self.path_img, id)
        if not os.path.exists(image_path):
            print(image_path)
        try:
            image = Image.open(image_path).convert('RGB')
            img_width, img_height = image.size ##########
            image = self.transform(image)
        except:
            self.count_img_error += 1
            # print('image has problem!')
            image_path_fail = os.path.join(self.path_img, '17sssss.jpg')
            image = Image.open(image_path_fail).convert('RGB')
            img_width, img_height = image.size #######
            image = self.transform(image)
        return image,img_width,img_height #########

    def get_aesc_spans(self, dic):
        aesc_spans = []
        for x in dic:
            aesc_spans.append((x['from'], x['to'], x['polarity']))
        return aesc_spans

    def get_gt_aspect_senti(self, dic):
        gt = []
        for x in dic:
            gt.append((' '.join(x['term']), x['polarity']))
        return gt

    def rescale(selg,box,weight,height):
        # 计算宽度和高度的变化比例
        width_ratio = 416 / weight
        height_ratio = 416 / height
        # 调整框的位置和大小
        box[0] = int((box[0]) * width_ratio)  # 调整框的中心 x 坐标
        box[1] = int((box[1]) * height_ratio)  # 调整框的中心 y 坐标
        box[2] = int(box[2] * width_ratio)  # 调整框的宽度
        box[3] = int(box[3] * height_ratio)  # 调整框的高度
        return box

    def __getitem__(self, index):
        output = {}  #输出的一个样本的dataset
        data = self.data_set[index]
        img_id = data['image_id'] #image_id是图片的名字

        o_img,img_width,img_height = self.get_img(img_id)
        output['o_img'] = o_img

        #rcnn_id = '/data/liuxj/aspect_sentiment_detect/Twitter_data/image_rcnn/twitter2015/'+img_id+'.npz'
        #img_feature = self.get_img_feature(rcnn_id) #读取
        #output['img_feat'] = img_feature #读取的图片（还没有输入resnet）

        #add related image
        if data['score']>0.5:
            output['rel']=1
        else:
            output['rel']=0
        output['sentence'] = ' '.join(data['words']) #原句子
        max_seq_len=128
        input_ids = self.tokenizer(output['sentence'].lower())['input_ids']  # <s>text_a</s></s>text_b</s>
        input_mask = [1] * len(input_ids)
        padding_id = [1] * (max_seq_len - len(input_ids))  # <pad> :1
        padding_mask = [0] * (max_seq_len - len(input_ids))

        input_ids += padding_id
        input_mask += padding_mask

        output['sentence_input_ids']=input_ids
        output['sentence_input_mask']=input_mask

        #add object detect
        _boxes=[]
        #print(data['o_person'])
        #print(data['image_id'])
        if len(data['o_person'])!=0:
            for per in data['o_person']:
                _boxes.append(float(0))
                for bb in per:
                    _boxes.append(float(bb))

        if len(data['o_entity'])!=0:
            for ent in data['o_entity']:
                _boxes.append(float(1))
                for bb in ent:
                    _boxes.append(float(bb))

        if len(data['o_background'])!=0:
            for bg in data['o_background']:
                _boxes.append(float(2))
                for bb in bg:
                    _boxes.append(float(bb))

        bb_labels={}
        _boxes=np.array(_boxes)
        boxes = np.split(_boxes, len(_boxes) // 5)
        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            bb_labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3,
                                                   5 + cfg.CLASS_NUM))
            for box in boxes:
                cls, cx, cy, w, h = box  # （中心点和宽和高）制作txt文件或读取时进行转换
                oboxes=np.array([cx, cy, w, h])
                oboxes=self.rescale(oboxes,img_width,img_height)
                cx, cy, w, h = oboxes
                cx_offset, cx_index = math.modf(cx * feature_size / 416)
                cy_offset, cy_index = math.modf(cy * feature_size / 416)
                for i, anchor in enumerate(anchors):  # i代表的是第几个锚框，anchor代表的是锚框的大小
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]  # 锚框的面积
                    p_w, p_h = w / anchor[0], h / anchor[1]  # w是物体的真实宽，h是物体的真高，anchor[0]代表的真实宽，anchor[1]代表的是锚框的高
                    p_area = w * h  # 物体的真实面积
                    iou = min(p_area, anchor_area) / max(p_area,anchor_area)
                    bb_labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h),
                         *one_hot(cfg.CLASS_NUM, int(cls))])

        output['bbox_13']=bb_labels[7]
        output['bbox_26'] = bb_labels[14]
        output['bbox_52'] = bb_labels[28]
        # add
        output['noun']=data['noun'] #名词

        aesc_spans = self.get_aesc_spans(data['aspects']) #方面词的开始索引，结束索引和情感极性
        output['aesc_spans'] = aesc_spans
        output['image_id'] = img_id
        gt = self.get_gt_aspect_senti(data['aspects']) #真实的方面词和情感极性，针对trc预训练
        output['gt'] = gt

        #print(img_id)
        #print(output['sentence'])

        return output
