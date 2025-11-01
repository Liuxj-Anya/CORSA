from datetime import datetime
import numpy as np
from torch.cuda.amp import autocast
import src.model.utils as utils
import src.eval_utils as eval_utils
import torch
import os
from src.utils import save_training_data


def fine_tune(epochs,
              model,
              train_loader,
              dev_loader,
              test_loader,
              metric,
              optimizer,
              device,
              args,
              logger=None,
              callback=None,
              log_interval=1,
              tb_writer=None,
              tb_interval=1,
              scaler=None):

    total_step = len(train_loader)*epochs
    model.train()
    total_loss = 0
    epoch=0
    global_step=0
    start_time = datetime.now()
    best_dev_res = None
    best_dev_test_res = None
    best_test_res = None
    eval_step=5
    while epoch < epochs:
        logger.info('Epoch {}'.format(epoch + 1), pad=True)
        for i, batch in enumerate(train_loader):
            model.train()
            # Forward pass
            global_step+=1
            aesc_infos = {key: value for key, value in batch['AESC'].items()}

            with autocast(enabled=args.amp):
                loss = model.forward(
                    input_ids=batch['input_ids'].to(device),
                    image_features=batch['o_image'].to(device), #list(map(lambda x: x.to(device), imgs_f)),
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
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, args.epochs, epoch*len(train_loader) + i + 1, total_step, loss.item()))
            # Backward and optimize

            cur_step = i + 1 + epoch * total_step
            t_step = args.epochs * total_step
            liner_warm_rate = utils.liner_warmup(cur_step, t_step, args.warmup)
            utils.set_lr(optimizer, liner_warm_rate * args.lr)

            optimizer.zero_grad()

            #print(loss.type())
            loss.backward()
            utils.clip_gradient(optimizer, args.grad_clip)

            optimizer.step()

            # test
            if (global_step + 1) % eval_step == 0:
                #eval_step=args.eval_step
                logger.info('Step {}'.format(global_step + 1), pad=True)

                res_dev = eval_utils.eval(args, model,dev_loader, metric, device)
                res_test = eval_utils.eval(args, model,test_loader, metric, device)

                logger.info('DEV  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
                    res_dev['aesc_pre'], res_dev['aesc_rec'], res_dev['aesc_f']))
                logger.info('DEV  ae_p:{} ae_r:{} ae_f:{}'.format(
                    res_dev['ae_pre'], res_dev['ae_rec'], res_dev['ae_f']))
                logger.info('DEV  sc_acc:{} sc_r:{} sc_f:{}'.format(
                    res_dev['sc_acc'], res_dev['sc_rec'], res_dev['sc_f']))

                logger.info('TEST  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
                    res_test['aesc_pre'], res_test['aesc_rec'], res_test['aesc_f']))
                logger.info('TEST  ae_p:{} ae_r:{} ae_f:{}'.format(
                    res_test['ae_pre'], res_test['ae_rec'], res_test['ae_f']))
                logger.info('TEST  sc_acc:{} sc_r:{} sc_f:{}'.format(
                    res_test['sc_acc'], res_test['sc_rec'], res_test['sc_f']))

                save_flag = False
                if best_dev_res is None:
                    best_dev_res = res_dev
                    best_dev_test_res = res_test
                else:
                    if best_dev_res['aesc_f'] < res_dev['aesc_f']:
                        best_dev_res = res_dev
                        best_dev_test_res = res_test

                if best_test_res is None:
                    best_test_res = res_test
                    save_flag = True
                else:
                    if best_test_res['aesc_f'] < res_test['aesc_f']:
                        best_test_res = res_test
                        save_flag = True

                if args.is_check == 1 and save_flag:
                    current_checkpoint_path = os.path.join(args.checkpoint_path,
                                                           args.check_info)
                    model.seq2seq_model.save_pretrained(current_checkpoint_path)
                    #save_img_encoder(args,img_encoder)
                    #torch.save(img_encoder, os.path.join(args.checkpoint_path, 'resnet152.pt'))
                    torch.save(model,os.path.join(args.checkpoint_path,'AoM.pt'))
                    logger.info('save model to {} !!!!!!!!!!!'.format(current_checkpoint_path))

                if res_dev['aesc_f'] > 68:
                    eval_step = 5
                elif res_dev['aesc_f'] > 67:
                    eval_step = 10
                elif res_dev['aesc_f'] > 66:
                    eval_step = 20
                elif res_dev['aesc_f'] <= 66:
                    eval_step = args.eval_step
        epoch += 1

    logger.info("Training complete in: " + str(datetime.now() - start_time),pad=True)
    logger.info('---------------------------')
    logger.info('BEST DEV:-----')
    logger.info('BEST DEV  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
        best_dev_res['aesc_pre'], best_dev_res['aesc_rec'],
        best_dev_res['aesc_f']))
    logger.info('BEST DEV  ae_p:{} ae_r:{} ae_f:{}'.format(
        best_dev_res['ae_pre'], best_dev_res['ae_rec'],
        best_dev_res['ae_f']))
    logger.info('BEST DEV  sc_acc:{} sc_r:{} sc_f:{}'.format(
        best_dev_res['sc_acc'], best_dev_res['sc_rec'],
        best_dev_res['sc_f']))

    logger.info('BEST DEV TEST:-----')
    logger.info('BEST DEV--TEST  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
        best_dev_test_res['aesc_pre'], best_dev_test_res['aesc_rec'],
        best_dev_test_res['aesc_f']))
    logger.info('BEST DEV--TEST  ae_p:{} ae_r:{} ae_f:{}'.format(
        best_dev_test_res['ae_pre'], best_dev_test_res['ae_rec'],
        best_dev_test_res['ae_f']))
    logger.info('BEST DEV--TEST  sc_acc:{} sc_r:{} sc_f:{}'.format(
        best_dev_test_res['sc_acc'], best_dev_test_res['sc_rec'],
        best_dev_test_res['sc_f']))

    logger.info('BEST TEST:-----')
    logger.info('BEST TEST  aesc_p:{} aesc_r:{} aesc_f:{}'.format(
        best_test_res['aesc_pre'], best_test_res['aesc_rec'],
        best_test_res['aesc_f']))
    logger.info('BEST TEST  ae_p:{} ae_r:{} ae_f:{}'.format(
        best_test_res['ae_pre'], best_test_res['ae_rec'],
        best_test_res['ae_f']))
    logger.info('BEST TEST  sc_acc:{} sc_r:{} sc_f:{}'.format(
        best_test_res['sc_acc'], best_test_res['sc_rec'],
        best_test_res['sc_f']))



def save_finetune_model(model):
    torch.save(model.state_dict(),'/home/zhouru/ABSA3/save_model/best_model.pth')


def save_img_encoder(args,img_encoder):
    file_name=os.path.join(args.checkpoint_path,'resnet152.pth')
    torch.save(img_encoder.state_dict(),file_name)
    pass