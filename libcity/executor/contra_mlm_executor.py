import os
import time
import torch
import numpy as np
from libcity.executor.contrastive_executor import ContrastiveExecutor
from libcity.model import loss
from tqdm import tqdm


class ContrastiveMLMExecutor(ContrastiveExecutor):
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)
        self.criterion_mask = torch.nn.NLLLoss(ignore_index=0, reduction='none')
        self.mlm_ratio = self.config.get("mlm_ratio", 1.)
        self.contra_ratio = self.config.get("contra_ratio", 1.)

    def _cal_loss(self, pred, targets, targets_mask):
        batch_loss_list = self.criterion_mask(pred.transpose(1, 2), targets)
        batch_loss = torch.sum(batch_loss_list)
        num_active = targets_mask.sum()
        mean_loss = batch_loss / num_active  # mean loss (over samples) used for optimization
        return mean_loss, batch_loss, num_active

    def _cal_acc(self, pred, targets, targets_mask):
        mask_label = targets[targets_mask]  # (num_active, )
        lm_output = pred[targets_mask].argmax(dim=-1)  # (num_active, )
        correct_l = mask_label.eq(lm_output).sum().item()
        return correct_l

    def train(self, train_dataloader, eval_dataloader, test_dataloader=None):
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = -1
        train_time = []
        eval_time = []
        train_loss = []
        train_acc = []
        eval_loss = []
        eval_acc = []
        lr_list = []

        num_batches = len(train_dataloader)
        self._logger.info("Num_batches: train={}, eval={}".format(num_batches, len(eval_dataloader)))

        for epoch_idx in range(self.epochs):
            start_time = time.time()
            train_avg_loss, train_avg_acc = self._train_epoch(train_dataloader, epoch_idx)
            t1 = time.time()
            train_time.append(t1 - start_time)
            train_loss.append(train_avg_loss)
            train_acc.append(train_avg_acc)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            eval_avg_loss, eval_avg_acc = self._valid_epoch(eval_dataloader, epoch_idx, mode='Eval')
            end_time = time.time()
            eval_time.append(end_time - t2)
            eval_loss.append(eval_avg_loss)
            eval_acc.append(eval_avg_acc)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(eval_avg_loss)
                elif self.lr_scheduler_type.lower() == 'cosinelr':
                    self.lr_scheduler.step(epoch_idx + 1)
                else:
                    self.lr_scheduler.step()

            log_lr = self.optimizer.param_groups[0]['lr']
            lr_list.append(log_lr)
            if (epoch_idx % self.log_every) == 0:
                message = 'Epoch [{}/{}] ({})  train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'. \
                    format(epoch_idx, self.epochs, (epoch_idx + 1) * num_batches, train_avg_loss,
                           eval_avg_loss, log_lr, (end_time - start_time))
                self._logger.info(message)

            if eval_avg_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, eval_avg_loss, model_file_name))
                min_val_loss = eval_avg_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break

            if (epoch_idx + 1) % self.test_every == 0:
                self.evaluate(test_dataloader)

        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)

        self._draw_png([(train_loss, eval_loss, 'loss'), (train_acc, eval_acc, 'acc'), (lr_list, 'lr')])
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx):
        batches_seen = epoch_idx * len(train_dataloader)  # 总batch数

        self.model = self.model.train()

        epoch_loss = []  # total loss of epoch
        total_correct_l = 0  # total top@1 acc for masked elements in epoch
        total_active_elements_l = 0  # total masked elements in epoch

        for i, batch in tqdm(enumerate(train_dataloader), desc="Train epoch={}".format(
                epoch_idx), total=len(train_dataloader)):
            contra_view1, contra_view2, X, targets, target_masks, padding_masks, batch_temporal_mat = batch
            # contra_view1/contra_view2: (batch_size, padded_length, feat_dim)
            # X/targets/target_masks: (batch_size, padded_length, feat_dim)
            # padding_masks: (batch_size, padded_length)
            # batch_temporal_mat: (batch_size, padded_length, padded_length)
            contra_view1 = contra_view1.to(self.device)
            contra_view2 = contra_view2.to(self.device)
            X = X.to(self.device)
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore
            batch_temporal_mat = batch_temporal_mat.to(self.device)

            graph_dict = self.graph_dict

            z1, z2, predictions_l = self.model(contra_view1=contra_view1, contra_view2=contra_view2,
                                               argument_methods1=self.data_argument1,
                                               argument_methods2=self.data_argument2,
                                               masked_input=X, padding_masks=padding_masks,
                                               batch_temporal_mat=batch_temporal_mat,
                                               graph_dict=graph_dict)
            # (B, d_model), (B, d_model), (B, T, vocab_size), (B, T, 1441)
            targets_l, target_masks_l = targets[..., 0], target_masks[..., 0]
            mean_loss_l, batch_loss_l, num_active_l = self._cal_loss(predictions_l, targets_l, target_masks_l)
            mean_loss_con = self._contrastive_loss(z1, z2, self.contra_loss_type)

            mean_loss = self.mlm_ratio * mean_loss_l + self.contra_ratio * mean_loss_con

            if self.test_align_uniform or self.train_align_uniform:
                align_uniform_loss, align_loss, uniform_loss = self.align_uniform(z1, z2)
                if self.train_align_uniform:
                    mean_loss += align_uniform_loss

            if self.l2_reg is not None:
                total_loss = mean_loss + self.l2_reg * loss.l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            total_loss = total_loss / self.grad_accmu_steps
            batches_seen += 1

            # with torch.autograd.detect_anomaly():
            total_loss.backward()

            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            if batches_seen % self.grad_accmu_steps == 0:
                self.optimizer.step()
                if self.lr_scheduler_type == 'cosinelr' and self.lr_scheduler is not None:
                    self.lr_scheduler.step_update(num_updates=batches_seen // self.grad_accmu_steps)
                self.optimizer.zero_grad()

            with torch.no_grad():
                total_correct_l += self._cal_acc(predictions_l, targets_l, target_masks_l)
                total_active_elements_l += num_active_l.item()
                epoch_loss.append(mean_loss.item())  # add total loss of batch

            post_fix = {
                "mode": "Train",
                "epoch": epoch_idx,
                "iter": i,
                "lr": self.optimizer.param_groups[0]['lr'],
                "Loc acc(%)": total_correct_l / total_active_elements_l * 100,
                "MLM loss": mean_loss_l.item(),
                "Contrastive loss": mean_loss_con.item(),
            }
            if self.test_align_uniform or self.train_align_uniform:
                post_fix['align_loss'] = align_loss
                post_fix['uniform_loss'] = uniform_loss
            if i % self.log_batch == 0:
                self._logger.info(str(post_fix))

        epoch_loss = np.mean(epoch_loss)  # average loss per element for whole epoch
        total_correct_l = total_correct_l / total_active_elements_l * 100.0
        self._logger.info("Train: expid = {}, Epoch = {}, avg_loss = {}, total_loc_acc = {}%."
                          .format(self.exp_id, epoch_idx, epoch_loss, total_correct_l))
        self._writer.add_scalar('Train loss', epoch_loss, epoch_idx)
        self._writer.add_scalar('Train loc acc', total_correct_l, epoch_idx)
        return epoch_loss, total_correct_l

    def _valid_epoch(self, eval_dataloader, epoch_idx, mode='Eval'):
        self.model = self.model.eval()
        if mode == 'Test':
            self.evaluator.clear()

        epoch_loss = []  # total loss of epoch
        total_correct_l = 0  # total top@1 acc for masked elements in epoch
        total_active_elements_l = 0  # total masked elements in epoch

        with torch.no_grad():
            for i, batch in tqdm(enumerate(eval_dataloader), desc="{} epoch={}".format(
                    mode, epoch_idx), total=len(eval_dataloader)):
                contra_view1, contra_view2, X, targets, target_masks, padding_masks, batch_temporal_mat = batch
                # contra_view1/contra_view2: (batch_size, padded_length, feat_dim)
                # X/targets/target_masks: (batch_size, padded_length, feat_dim)
                # padding_masks: (batch_size, padded_length)
                # batch_temporal_mat: (batch_size, padded_length, padded_length)
                contra_view1 = contra_view1.to(self.device)
                contra_view2 = contra_view2.to(self.device)
                X = X.to(self.device)
                targets = targets.to(self.device)
                target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
                padding_masks = padding_masks.to(self.device)  # 0s: ignore
                batch_temporal_mat = batch_temporal_mat.to(self.device)

                z1, z2, predictions_l = self.model(contra_view1=contra_view1, contra_view2=contra_view2,
                                                   argument_methods1=self.data_argument1,
                                                   argument_methods2=self.data_argument2,
                                                   masked_input=X, padding_masks=padding_masks,
                                                   batch_temporal_mat=batch_temporal_mat,
                                                   graph_dict=self.graph_dict)
                # (B, d_model), (B, d_model), (B, T, vocab_size), (B, T, 1441)
                # 0维是路段的label
                targets_l, target_masks_l = targets[..., 0], target_masks[..., 0]
                mean_loss_l, batch_loss_l, num_active_l = self._cal_loss(predictions_l, targets_l, target_masks_l)
                # 对比loss
                mean_loss_con = self._contrastive_loss(z1, z2, self.contra_loss_type)

                mean_loss = self.mlm_ratio * mean_loss_l + self.contra_ratio * mean_loss_con

                if self.test_align_uniform or self.train_align_uniform:
                    align_uniform_loss, align_loss, uniform_loss = self.align_uniform(z1, z2)
                    if self.train_align_uniform:
                        mean_loss += align_uniform_loss

                if mode == 'Test':
                    evaluate_input = {
                        'loc_true': targets_l[target_masks_l].reshape(-1, 1).squeeze(-1).cpu().numpy(),  # (num_active, )
                        'loc_pred': predictions_l[target_masks_l].reshape(-1, predictions_l.shape[-1]).cpu().numpy()
                        # (num_active, n_class)
                    }
                    self.evaluator.collect(evaluate_input)

                total_correct_l += self._cal_acc(predictions_l, targets_l, target_masks_l)
                total_active_elements_l += num_active_l.item()
                epoch_loss.append(mean_loss.item())  # add total loss of batch

                post_fix = {
                    "mode": "Train",
                    "epoch": epoch_idx,
                    "iter": i,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "Loc acc(%)": total_correct_l / total_active_elements_l * 100,
                    "MLM loss": mean_loss_l.item(),
                    "Contrastive loss": mean_loss_con.item(),
                }
                if self.test_align_uniform or self.train_align_uniform:
                    post_fix['align_loss'] = align_loss
                    post_fix['uniform_loss'] = uniform_loss
                if i % self.log_batch == 0:
                    self._logger.info(str(post_fix))

            epoch_loss = np.mean(epoch_loss)  # average loss per element for whole epoch
            total_correct_l = total_correct_l / total_active_elements_l * 100.0
            self._logger.info("{}: expid = {}, Epoch = {}, avg_loss = {}, total_loc_acc = {}%."
                              .format(mode, self.exp_id, epoch_idx, epoch_loss, total_correct_l))
            self._writer.add_scalar('{} loss'.format(mode), epoch_loss, epoch_idx)
            self._writer.add_scalar('{} loc acc'.format(mode), total_correct_l, epoch_idx)

            if mode == 'Test':
                self.evaluator.save_result(self.evaluate_res_dir)
            return epoch_loss, total_correct_l
