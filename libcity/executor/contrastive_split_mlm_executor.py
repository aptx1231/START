import torch
import numpy as np
from libcity.executor.contra_mlm_executor import ContrastiveMLMExecutor
from libcity.model import loss
from tqdm import tqdm


class ContrastiveSplitMLMExecutor(ContrastiveMLMExecutor):
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)

    def _train_epoch(self, train_dataloader, epoch_idx):
        batches_seen = epoch_idx * len(train_dataloader)

        self.model = self.model.train()

        epoch_loss = []  # total loss of epoch
        total_correct_l = 0  # total top@1 acc for masked elements in epoch
        total_active_elements_l = 0  # total masked elements in epoch

        for i, batch in tqdm(enumerate(train_dataloader), desc="Train epoch={}".format(
                epoch_idx), total=len(train_dataloader)):
            contra_view1, contra_view2, padding_masks1, padding_masks2, batch_temporal_mat1, batch_temporal_mat2, \
            X, targets, target_masks, padding_masks, batch_temporal_mat = batch
            # contra_view1/contra_view2: (batch_size, padded_length, feat_dim)
            # X/targets/target_masks: (batch_size, padded_length, feat_dim)
            # padding_masks/1/2: (batch_size, padded_length)
            # batch_temporal_mat/1/2: (batch_size, padded_length, padded_length)
            contra_view1 = contra_view1.to(self.device)
            contra_view2 = contra_view2.to(self.device)
            padding_masks1 = padding_masks1.to(self.device)  # 0s: ignore
            padding_masks2 = padding_masks2.to(self.device)  # 0s: ignore
            batch_temporal_mat1 = batch_temporal_mat1.to(self.device)
            batch_temporal_mat2 = batch_temporal_mat2.to(self.device)
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
                                               padding_masks1=padding_masks1,
                                               batch_temporal_mat1=batch_temporal_mat1,
                                               padding_masks2=padding_masks2,
                                               batch_temporal_mat2=batch_temporal_mat2,
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
                contra_view1, contra_view2, padding_masks1, padding_masks2, batch_temporal_mat1, batch_temporal_mat2, \
                X, targets, target_masks, padding_masks, batch_temporal_mat = batch
                # contra_view1/contra_view2: (batch_size, padded_length, feat_dim)
                # X/targets/target_masks: (batch_size, padded_length, feat_dim)
                # padding_masks/1/2: (batch_size, padded_length)
                # batch_temporal_mat/1/2: (batch_size, padded_length, padded_length)
                contra_view1 = contra_view1.to(self.device)
                contra_view2 = contra_view2.to(self.device)
                padding_masks1 = padding_masks1.to(self.device)  # 0s: ignore
                padding_masks2 = padding_masks2.to(self.device)  # 0s: ignore
                batch_temporal_mat1 = batch_temporal_mat1.to(self.device)
                batch_temporal_mat2 = batch_temporal_mat2.to(self.device)
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
                                                   padding_masks1=padding_masks1,
                                                   batch_temporal_mat1=batch_temporal_mat1,
                                                   padding_masks2=padding_masks2,
                                                   batch_temporal_mat2=batch_temporal_mat2,
                                                   graph_dict=self.graph_dict)
                # (B, d_model), (B, d_model), (B, T, vocab_size), (B, T, 1441)
                targets_l, target_masks_l = targets[..., 0], target_masks[..., 0]
                mean_loss_l, batch_loss_l, num_active_l = self._cal_loss(predictions_l, targets_l, target_masks_l)
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
