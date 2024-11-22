# -*- coding: utf-8 -*-
# @Time    : 2022/11/5 16:54
# @Author  : codewen77
import json
import os
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from labels import  get_sentiment
from losses.acos_losses import calculate_entity_loss,  calculate_sentiment_loss, \
    FocalLoss
from metrics import ACOSScore
from question_template import get_English_Template
from tools import filter_unpaired


class ACOSTrainer:
    def __init__(self, logger, model, optimizer, scheduler, tokenizer, args):
        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.args = args
        # self.fgm = FGM(self.model)
        # self.pgd = PGD(self.model)
        self.focalLoss = FocalLoss(self.args.flp_gamma)

    def train(self, train_dataloader, epoch):
        with tqdm(total=len(train_dataloader), desc="train") as pbar:
            for batch_idx, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()

                loss_sum = self.get_train_loss(batch)
                loss_sum.backward()

                # # 使用FGM对抗训练
                # if self.args.use_FGM:
                #     # 在embedding层上添加对抗扰动
                #     self.fgm.attack()
                #     FGM_loss_sum = self.get_train_loss(batch)

                #     # 恢复embedding参数
                #     FGM_loss_sum.backward()
                #     self.fgm.restore()

                # # 使用PGD对抗训练
                # if self.args.use_PGD:
                #     self.pgd.backup_grad()
                #     for t in range(self.args.pgd_k):
                #         # 在embedding上添加对抗扰动, first attack时备份param.data
                #         self.pgd.attack(is_first_attack=(t == 0))
                #         if t != self.args.pgd_k - 1:
                #             self.model.zero_grad()
                #         else:
                #             self.pgd.restore_grad()

                #         PGD_loss_sum = self.get_train_loss(batch)
                #         # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                #         PGD_loss_sum.backward()
                #         # 恢复embedding参数
                #     self.pgd.restore()

                # 梯度下降 更新参数
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

                pbar.set_description(f'Epoch [{epoch}/{self.args.epoch_num}]')
                pbar.set_postfix({'loss': '{0:1.5f}'.format(loss_sum)})
                pbar.update(1)

    def eval(self, eval_dataloader):
        json_res = []
        acos_score = ACOSScore(self.logger)  # 用于评估aspect和sentiment的匹配情况
        self.model.eval()
        Forward_Q1, Q4 = get_English_Template()[:2]  # 仅保留aspect和sentiment模板
        f_asp_imp_start = 5  # aspect阈值

        for batch in tqdm(eval_dataloader):
            asp_predict, sentiment_predict = [], []  # 保存aspect和sentiment预测结果

            # Aspect预测部分
            passenge_index = batch.forward_asp_answer_start[0].gt(-1).float().nonzero()
            passenge = batch.forward_asp_query[0][passenge_index].squeeze(1)

            f_asp_start_scores, f_asp_end_scores = self.model(batch.forward_asp_query.cuda(),
                                                            batch.forward_asp_query_mask.cuda(),
                                                            batch.forward_asp_query_seg.cuda(), 0)
            f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)
            f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)
            f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=1)
            f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=1)

            f_asp_start_prob_temp = []
            f_asp_end_prob_temp = []
            f_asp_start_index_temp = []
            f_asp_end_index_temp = []
            for i in range(f_asp_start_ind.size(0)):
                if batch.forward_asp_answer_start[0, i] != -1:
                    if f_asp_start_ind[i].item() == 1:
                        f_asp_start_index_temp.append(i)
                        f_asp_start_prob_temp.append(f_asp_start_prob[i].item())
                    if f_asp_end_ind[i].item() == 1:
                        f_asp_end_index_temp.append(i)
                        f_asp_end_prob_temp.append(f_asp_end_prob[i].item())

            f_asp_start_index, f_asp_end_index, f_asp_prob = filter_unpaired(
                f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp, f_asp_imp_start
            )

            # Sentiment预测部分
            for i in range(len(f_asp_start_index)):
                sentiment_query = self.tokenizer.convert_tokens_to_ids(
                    [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[:6]]
                )
                for j in range(f_asp_start_index[i], f_asp_end_index[i] + 1):
                    sentiment_query.append(batch.forward_asp_query[0][j].item())
                sentiment_query.extend(self.tokenizer.convert_tokens_to_ids(
                    [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[6:]]
                ))

                sentiment_query_seg = [0] * len(sentiment_query)
                sentiment_query = torch.tensor(sentiment_query).long()
                sentiment_query = torch.cat([sentiment_query, passenge], -1).cuda().unsqueeze(0)
                sentiment_query_seg += [1] * passenge.size(0)
                sentiment_query_mask = torch.ones(sentiment_query.size(1)).float().cuda().unsqueeze(0)
                sentiment_query_seg = torch.tensor(sentiment_query_seg).long().cuda().unsqueeze(0)

                sentiment_scores = self.model(sentiment_query, sentiment_query_mask, sentiment_query_seg, 1)
                sentiment_scores = F.softmax(sentiment_scores[0], dim=1)
                sentiment_predicted = torch.argmax(sentiment_scores, dim=1).item()

                # 将结果转化为标注形式
                asp_f = [f_asp_start_index[i] - 6, f_asp_end_index[i] - 6]  # 去掉模板部分偏移量
                sentiment_predict.append(sentiment_predicted)
                asp_predict.append(asp_f)

            # 记录预测结果
            acos_score.update(batch.aspects[0], batch.sentiments[0], asp_predict, sentiment_predict)

            # 保存预测到json文件中
            json_res.append({
                'sentence': ' '.join(batch.sentence_token[0]),
                'pred_aspect': asp_predict,
                'pred_sentiment': sentiment_predict,
                'gold_aspect': batch.aspects[0],
                'gold_sentiment': batch.sentiments[0]
            })

        with open(os.path.join(self.args.output_dir, self.args.task, self.args.data_type, 'pred.json'), 'w', encoding='utf-8') as fP:
            json.dump(json_res, fP, ensure_ascii=False, indent=4)

        return acos_score.compute()


    def batch_eval(self, eval_dataloader):
        start_time = time.time()
        json_res = []
        acos_score = ACOSScore(self.logger)  # 用于评估aspect和sentiment的匹配情况
        self.model.eval()
        Forward_Q1, Q4 = get_English_Template()[:2]  # 保留aspect和sentiment模板
        f_asp_imp_start = 5  # aspect阈值

        for batch in tqdm(eval_dataloader):
            passenges = []
            for p in range(len(batch.forward_asp_answer_start)):
                passenge_index = batch.forward_asp_answer_start[p].gt(-1).float().nonzero()
                passenge = batch.forward_asp_query[p][passenge_index].squeeze(1)
                passenges.append(passenge)

            batch_size = len(passenges)
            turn1_query = batch.forward_asp_query
            turn1_mask = batch.forward_asp_query_mask
            turn1_seg = batch.forward_asp_query_seg

            # Forward Aspect Prediction
            f_asp_start_scores, f_asp_end_scores = self.model(turn1_query.cuda(),
                                                            turn1_mask.cuda(),
                                                            turn1_seg.cuda(), 0)

            f_asp_start_scores = F.softmax(f_asp_start_scores, dim=-1)
            f_asp_end_scores = F.softmax(f_asp_end_scores, dim=-1)
            f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=-1)
            f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=-1)

            f_asp_start_indexs, f_asp_end_indexs, f_asp_probs = [], [], []
            for b in range(f_asp_end_prob.size(0)):
                f_asp_start_prob_temp = []
                f_asp_end_prob_temp = []
                f_asp_start_index_temp = []
                f_asp_end_index_temp = []
                for i in range(f_asp_start_ind[b].size(0)):
                    # 填充部分不需要考虑
                    if batch.sentence_len[b] + f_asp_imp_start < i:
                        break
                    if batch.forward_asp_answer_start[b, i] != -1:
                        if f_asp_start_ind[b][i].item() == 1:
                            f_asp_start_index_temp.append(i)
                            f_asp_start_prob_temp.append(f_asp_start_prob[b][i].item())
                        if f_asp_end_ind[b][i].item() == 1:
                            f_asp_end_index_temp.append(i)
                            f_asp_end_prob_temp.append(f_asp_end_prob[b][i].item())

                f_asp_start_index, f_asp_end_index, f_asp_prob = filter_unpaired(
                    f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp,
                    f_asp_imp_start)
                f_asp_start_indexs.append(f_asp_start_index)
                f_asp_end_indexs.append(f_asp_end_index)
                f_asp_probs.append(f_asp_prob)

            # Sentiment Prediction
            ao_sentiment_querys, ao_sentiment_segs, ao_sentiment_masks = [], [], []
            batch_sentiment_idxs = []
            for b in range(len(f_asp_start_indexs)):
                for i in range(len(f_asp_start_indexs[b])):
                    sentiment_query = self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[:6]]
                    )
                    for j in range(f_asp_start_indexs[b][i], f_asp_end_indexs[b][i] + 1):
                        sentiment_query.append(batch.forward_asp_query[b][j].item())
                    sentiment_query.extend(self.tokenizer.convert_tokens_to_ids(
                        [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[6:]]
                    ))

                    sentiment_query_seg = [0] * len(sentiment_query)
                    sentiment_query = torch.tensor(sentiment_query).long()
                    sentiment_query = torch.cat([sentiment_query, passenges[b]], -1)
                    sentiment_query_seg += [1] * passenges[b].size(0)
                    sentiment_query_mask = torch.ones(sentiment_query.size(0)).float()
                    sentiment_query_seg = torch.tensor(sentiment_query_seg).long()

                    ao_sentiment_querys.append(sentiment_query)
                    ao_sentiment_segs.append(sentiment_query_seg)
                    ao_sentiment_masks.append(sentiment_query_mask)
                    batch_sentiment_idxs.append(b)

            if ao_sentiment_querys:
                # Padding Sentiment Queries
                ao_sentiment_querys = pad_sequence(ao_sentiment_querys, batch_first=True, padding_value=0).cuda()
                ao_sentiment_segs = pad_sequence(ao_sentiment_segs, batch_first=True, padding_value=1).cuda()
                ao_sentiment_masks = pad_sequence(ao_sentiment_masks, batch_first=True, padding_value=0).cuda()

                sentiment_scores = self.model(ao_sentiment_querys, ao_sentiment_masks, ao_sentiment_segs, 1)
                sentiment_scores = F.softmax(sentiment_scores, dim=-1)
                sentiment_predicted = torch.argmax(sentiment_scores, dim=-1)

                final_aspects, final_sentiments = [], []
                for idx in range(len(batch_sentiment_idxs)):
                    b_idx = batch_sentiment_idxs[idx]
                    aspect = [f_asp_start_indexs[b_idx][0] - 6, f_asp_end_indexs[b_idx][0] - 6]  # 修正偏移
                    sentiment = sentiment_predicted[idx].item()
                    if aspect not in final_aspects:
                        final_aspects.append(aspect)
                        final_sentiments.append(sentiment)

                # 记录结果
                json_res.append({
                    'sentence': ' '.join(batch.sentence_token[0]),
                    'pred_aspect': final_aspects,
                    'pred_sentiment': final_sentiments,
                    'gold_aspect': batch.aspects[0],
                    'gold_sentiment': batch.sentiments[0]
                })

                acos_score.update(batch.aspects, batch.sentiments, final_aspects, final_sentiments)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"*************************执行总耗时：{elapsed_time}秒*************************")
        return acos_score.compute()


    def inference(self, reviews):
        self.model.eval()
        id2Sentiment = get_sentiment(self.args.task.lower())[-1]
        Forward_Q1, Q4 = get_English_Template()[:2]  # 保留aspect和sentiment模板
        f_asp_imp_start = 5

        f_asp_query_list, f_asp_mask_list, f_asp_seg_list = [], [], []
        passenge_indexs, passenges = [], []
        f_max_len = 0

        for review in reviews:
            review = review.split(' ')
            f_temp_text = Forward_Q1 + ["null"] + review
            f_temp_text = list(map(self.tokenizer.tokenize, f_temp_text))
            f_temp_text = [item for indices in f_temp_text for item in indices]
            passenge_indexs.append([i + len(Forward_Q1) for i in range(len(f_temp_text) - len(Forward_Q1))])
            passenges.append(self.tokenizer.convert_tokens_to_ids(["null"] + review))
            _forward_asp_query = self.tokenizer.convert_tokens_to_ids(f_temp_text)
            if f_max_len < len(_forward_asp_query):
                f_max_len = len(_forward_asp_query)
            f_asp_query_list.append(_forward_asp_query)
            _forward_asp_mask = [1 for _ in range(len(_forward_asp_query))]
            f_asp_mask_list.append(_forward_asp_mask)
            _forward_asp_seg = [0] * len(self.tokenizer.convert_tokens_to_ids(Forward_Q1)) + [1] * (
                len(self.tokenizer.convert_tokens_to_ids(review)) + 1)
            f_asp_seg_list.append(_forward_asp_seg)

        for b in range(len(f_asp_query_list)):
            asp_predict, aspect_sentiments = [], []

            forward_asp_query = torch.tensor([f_asp_query_list[b]]).long()
            forward_asp_query_mask = torch.tensor([f_asp_mask_list[b]]).long()
            forward_asp_query_seg = torch.tensor([f_asp_seg_list[b]]).long()
            passenge = torch.tensor(passenges[b]).long()

            # Aspect Extraction
            f_asp_start_scores, f_asp_end_scores = self.model(forward_asp_query.cuda(),
                                                            forward_asp_query_mask.cuda(),
                                                            forward_asp_query_seg.cuda(), 0)
            f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)
            f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)
            f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=1)
            f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=1)
            f_asp_start_prob_temp = []
            f_asp_end_prob_temp = []
            f_asp_start_index_temp = []
            f_asp_end_index_temp = []
            for i in range(f_asp_start_ind.size(0)):
                if i in passenge_indexs[b]:
                    if f_asp_start_ind[i].item() == 1:
                        f_asp_start_index_temp.append(i)
                        f_asp_start_prob_temp.append(f_asp_start_prob[i].item())
                    if f_asp_end_ind[i].item() == 1:
                        f_asp_end_index_temp.append(i)
                        f_asp_end_prob_temp.append(f_asp_end_prob[i].item())

            f_asp_start_index, f_asp_end_index, f_asp_prob = filter_unpaired(
                f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp, f_asp_imp_start)

            # Sentiment Prediction
            sentiment_query_list, sentiment_mask_list, sentiment_seg_list = [], [], []
            for i in range(len(f_asp_start_index)):
                sentiment_query = self.tokenizer.convert_tokens_to_ids(
                    [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[:6]])
                for j in range(f_asp_start_index[i], f_asp_end_index[i] + 1):
                    sentiment_query.append(forward_asp_query[0][j].item())
                sentiment_query.extend(self.tokenizer.convert_tokens_to_ids(
                    [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in Q4[6:]]))

                sentiment_query_seg = [0] * len(sentiment_query)
                sentiment_query = torch.tensor(sentiment_query).long()
                sentiment_query = torch.cat([sentiment_query, passenge], -1).cuda().unsqueeze(0)
                sentiment_query_seg += [1] * passenge.size(0)
                sentiment_query_mask = torch.ones(sentiment_query.size(1)).float().cuda().unsqueeze(0)
                sentiment_query_seg = torch.tensor(sentiment_query_seg).long().cuda().unsqueeze(0)

                sentiment_scores = self.model(sentiment_query, sentiment_query_mask, sentiment_query_seg, 1)
                sentiment_scores = F.softmax(sentiment_scores, dim=1)
                sentiment_predicted = torch.argmax(sentiment_scores[0], dim=0).item()

                aspect = [f_asp_start_index[i] - 6, f_asp_end_index[i] - 6]  # 修正偏移
                asp_predict.append(aspect)
                aspect_sentiments.append(sentiment_predicted)

            # Format Results
            print_aspect_sentiment_pairs = []
            review_list = reviews[b].split(' ')
            tokenized_review = list(map(self.tokenizer.tokenize, review_list))
            subword_lengths = list(map(len, tokenized_review))
            token_start_idxs = np.cumsum([0] + subword_lengths[:-1])
            tokenized2word = {}
            for i in range(len(reviews[b].split(' '))):
                for j in range(token_start_idxs[i], token_start_idxs[i] + subword_lengths[i]):
                    tokenized2word[j] = i

            for i, aspect in enumerate(asp_predict):
                if aspect == [-1, -1]:
                    asp = 'NULL'
                else:
                    asp = ' '.join(review_list[tokenized2word[aspect[0]]:tokenized2word[aspect[-1]] + 1])
                sent = id2Sentiment[aspect_sentiments[i]]
                print_aspect_sentiment_pairs.append([asp, sent])

            print(f"`{reviews[b]}` 二元组抽取结果：`{print_aspect_sentiment_pairs}`")
        
    

    def get_train_loss(self, batch):
            """只计算 aspect 和 sentiment 的损失"""
            # 获取输入长度
            max_aspect_len = max(batch.aspect_query_len)
            max_sentiment_len = max(batch.sentiment_query_len)

            # 截取有效部分
            aspect_query = batch.aspect_query[:, :max_aspect_len]
            aspect_query_mask = batch.aspect_query_mask[:, :max_aspect_len]
            aspect_query_seg = batch.aspect_query_seg[:, :max_aspect_len]
            aspect_answer_start = batch.aspect_answer_start[:, :max_aspect_len]
            aspect_answer_end = batch.aspect_answer_end[:, :max_aspect_len]

            sentiment_query = batch.sentiment_query[:, :max_sentiment_len]
            sentiment_query_mask = batch.sentiment_query_mask[:, :max_sentiment_len]
            sentiment_query_seg = batch.sentiment_query_seg[:, :max_sentiment_len]
            sentiment_answer = batch.sentiment_answer[:, :max_sentiment_len]

            # 计算模型预测结果
            # q1: aspect extraction
            aspect_start_scores, aspect_end_scores = self.model(aspect_query.cuda(),
                                                                aspect_query_mask.cuda(),
                                                                aspect_query_seg.cuda(), 0)
            # q2: sentiment classification
            sentiment_scores = self.model(sentiment_query.cuda(),
                                        sentiment_query_mask.cuda(),
                                        sentiment_query_seg.cuda(), 1)

            # 计算 aspect 的损失
            aspect_loss = calculate_entity_loss(aspect_start_scores, aspect_end_scores,
                                                aspect_answer_start.cuda(), aspect_answer_end.cuda())

            # 计算 sentiment 的损失
            if self.args.use_FocalLoss:
                sentiment_loss = self.focalLoss(sentiment_scores, sentiment_answer.cuda())
            else:
                sentiment_loss = calculate_sentiment_loss(sentiment_scores, sentiment_answer.cuda())

            # 总损失为 aspect 损失和 sentiment 损失的加权和
            loss_sum = aspect_loss + 3 * sentiment_loss
            return loss_sum