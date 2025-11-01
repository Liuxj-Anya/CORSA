import torch
import torch.nn as nn
import pdb

def eval(args, model,loader, metric, device):
    model.eval()

    for i, batch in enumerate(loader):
        # Forward pass
        if args.task == 'twitter_ae':
            aesc_infos = {
                key: value
                for key, value in batch['TWITTER_AE'].items()
            }
        elif args.task == 'twitter_sc':
            aesc_infos = {
                key: value
                for key, value in batch['TWITTER_SC'].items()
            }
        else:
            aesc_infos = {key: value for key, value in batch['AESC'].items()}

        """with torch.no_grad():
                imgs_f=[x.numpy().tolist() for x in batch['image_resnets']]
                imgs_f=torch.tensor(imgs_f).to(device)
                imgs_f, img_mean, img_att = img_encoder(imgs_f)
                img_att=img_att.view(-1, 2048, 49).permute(0, 2, 1)
                img_att=torch.tensor(img_att)"""

        predict = model.predict(
            input_ids=batch['input_ids'].to(device),
            image_features=batch['o_image'].to(device),
            sentiment_value=batch['sentiment_value'].to(device) if batch['sentiment_value'] is not None else None,
            noun_mask=batch['noun_mask'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            dependency_matrix=batch['dependency_matrix'].to(device),
            aesc_infos=aesc_infos,
            si_input_ids=batch['s_i'].to(device),
            sm_input_mask=batch['s_m'].to(device),
            relation_label=batch['rel'].to(device),
            o_target_13=batch['bbox_13'].to(device),
            o_target_26=batch['bbox_26'].to(device),
            o_target_52=batch['bbox_52'].to(device)
        )
        metric.evaluate(aesc_infos['spans'], predict,
                        aesc_infos['labels'].to(device))
        # break

    res = metric.get_metric()
    model.train()
    return res
