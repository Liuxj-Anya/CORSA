import warnings
import numpy as np
import torch
from itertools import chain
from transformers import RobertaTokenizer, RobertaModel

class Collator: #dataloader的回调函数
    """
    The collator for all types of dataset.
    Remember to add the corresponding collation code after adding a new type of task.
    """
    def __init__(self,
                 tokenizer,
                 has_label=True,
                 aesc_enabled=False,
                 text_only=False,
                 trc_enabled=False,
                 lm_max_len=30,
                 max_img_num=49,
                 max_span_len=20):
        """
        :param tokenizer: ConditionTokenizer
        :param mlm_enabled: bool, if use mlm for language modeling. False for autoregressive modeling
        :param mrm_enabled: bool, if use mrm
        :param rp_enabled: bool, if use relation prediction (VG)
        :param ap_enabled: bool, if use attribute prediction (VG)
        :param mlm_probability: float, probability to mask the tokens
        :param mrm_probability: float, probability to mask the regions
        """
        self._tokenizer = tokenizer
        self._has_label = has_label
        self._aesc_enabled = aesc_enabled
        self._trc_enabled=trc_enabled
        self._lm_max_len = lm_max_len
        self._max_img_num = max_img_num
        self._max_span_len = max_span_len
        self.text_only = text_only
        self.roberta_tokenizer = RobertaTokenizer
        if not has_label:
            raise ValueError(
                'mlm_enabled can not be true while has_label is false. MLM need labels.'
            )

    def _clip_text(self, text, length):
        tokenized = []
        for i, word in enumerate(text.split()):
            if i == 0:
                bpes = self._tokenizer._base_tokenizer.tokenize(word)
            else:
                bpes = self._tokenizer._base_tokenizer.tokenize(
                    word, add_prefix_space=True)
            bpes = self._tokenizer._base_tokenizer.convert_tokens_to_ids(bpes)
            tokenized.append(bpes)
        _tokenized = list(chain(*tokenized))
        return self._tokenizer.get_base_tokenizer().decode(_tokenized[:length])

    def __call__(self, batch): #对一个批量的dataset进行处理，返回dataloader
        batch = [entry for entry in batch if entry is not None] #包含一个批量数据的列表
        #image_features =np.array([x['img_feat'] for x in batch]) #一个批量的图片特征的列表，shape=(batch_size,)
         #[49,49,49,49,.....] len=batch_size,49为获取的图像块个数

        o_image = [x['o_img'] for x in batch]
        img_num = [49] * len(o_image)
        o_image = [x.numpy().tolist() for x in o_image]

        bbox_13= np.array([x['bbox_13'] for x in batch])
        bbox_26 = np.array([x['bbox_26'] for x in batch])
        bbox_52 = np.array([x['bbox_52'] for x in batch])

        #add related image
        rel=[x['rel'] for x in batch]
        s_i = [x['sentence_input_ids'] for x in batch]
        s_m = [x['sentence_input_mask'] for x in batch]
        s_i = list(map(list, zip(*s_i)))
        s_m = list(map(list, zip(*s_m)))

        s_i = torch.tensor(s_i, dtype=torch.long)
        s_m = torch.tensor(s_m, dtype=torch.long)
        rel=torch.tensor(rel, dtype=torch.long)

        #image_features = torch.tensor(image_features, dtype=torch.float)
        o_image = torch.tensor(o_image, dtype=torch.float)

        bbox_13 = torch.tensor(bbox_13, dtype=torch.float)
        bbox_26 = torch.tensor(bbox_26, dtype=torch.float)
        bbox_52 = torch.tensor(bbox_52, dtype=torch.float)

        target = [x['sentence'] for x in batch] #一个批量的原句子的列表，['word word word ....','........',........]
        sentence = list(target)

        encoded_conditions = self._tokenizer.encode_condition(
            img_num=img_num, sentence=sentence, text_only=self.text_only) #分词编码，返回ids

        input_ids = encoded_conditions['input_ids']
        output = {}
        #condition_img_mask = encoded_conditions['img_mask'] #img_mask和sentence_mask没有使用

        output['text_input_ids'] = encoded_conditions['text_input_ids']
        output['text_attention_mask'] = encoded_conditions['text_attention_mask']

        output['s_i']=torch.transpose(s_i, 0, 1)
        output['s_m']=torch.transpose(s_m, 0, 1)

        output['rel'] = rel
        output['input_ids'] = input_ids
        output['attention_mask'] = encoded_conditions['attention_mask']
        #output['image_features'] = image_features
        output['o_image'] = o_image

        #add obj_detect
        output['bbox_13'] = bbox_13
        output['bbox_26'] = bbox_26
        output['bbox_52'] = bbox_52


        #新加（情感嵌入，依赖矩阵，名词（候选方面词））
        output['sentiment_value']=encoded_conditions['sentiment_value']
        output['noun_mask']=encoded_conditions['noun_mask']
        output['dependency_matrix']=encoded_conditions['dependency_matrix']

        if self._has_label:
            if self._aesc_enabled:
                output['AESC'] = self._tokenizer.encode_aesc(
                    target, [x['aesc_spans'] for x in batch],  # target = [x['sentence'] for x in batch]
                    self._max_span_len)
                output['task'] = 'AESC'
            if self._trc_enabled: #针对trc预训练
                output['ifpairs']=[x['ifpairs'] for x in batch]

        output['image_id'] = [x['image_id'] for x in batch]
        if self._trc_enabled==False: #针对trc预训练
            output['gt'] = [x['gt'] for x in batch]
        return output