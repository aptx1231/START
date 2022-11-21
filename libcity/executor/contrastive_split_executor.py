import torch
import numpy as np
from libcity.executor.contrastive_executor import ContrastiveExecutor
from libcity.model import loss
from tqdm import tqdm


class ContrastiveSplitExecutor(ContrastiveExecutor):
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)

    def _train_epoch(self, train_dataloader, epoch_idx):
        batches_seen = epoch_idx * len(train_dataloader)  # 总batch数

        self.model = self.model.train()

        epoch_loss = []

        for i, batch in tqdm(enumerate(train_dataloader), desc="Train epoch={}".format(
                epoch_idx), total=len(train_dataloader)):
            X1, X2, padding_masks1, padding_masks2, batch_temporal_mat1, batch_temporal_mat2 = batch
            # X1/X2: (batch_size, padded_length, feat_dim)
            # padding_masks1/2: (batch_size, padded_length)
            # batch_temporal_mat1/2: (batch_size, padded_length, padded_length)
            X1 = X1.to(self.device)
            X2 = X2.to(self.device)
            padding_masks1 = padding_masks1.to(self.device)  # 0s: ignore
            padding_masks2 = padding_masks2.to(self.device)  # 0s: ignore
            batch_temporal_mat1 = batch_temporal_mat1.to(self.device)
            batch_temporal_mat2 = batch_temporal_mat2.to(self.device)

            graph_dict = self.graph_dict

            z1 = self.model(x=X1, padding_masks=padding_masks1, batch_temporal_mat=batch_temporal_mat1,
                            argument_methods=self.data_argument1, graph_dict=graph_dict)  # (batch_size, d_model)
            z2 = self.model(x=X2, padding_masks=padding_masks2, batch_temporal_mat=batch_temporal_mat2,
                            argument_methods=self.data_argument2, graph_dict=graph_dict)  # (batch_size, d_model)

            mean_loss = self._contrastive_loss(z1, z2, self.contra_loss_type)

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
                epoch_loss.append(mean_loss.item())

            post_fix = {
                "mode": "Train",
                "epoch": epoch_idx,
                "iter": i,
                "lr": self.optimizer.param_groups[0]['lr'],
                "loss": mean_loss.item(),
            }
            if self.test_align_uniform or self.train_align_uniform:
                post_fix['align_loss'] = align_loss
                post_fix['uniform_loss'] = uniform_loss
            if i % self.log_batch == 0:
                self._logger.info(str(post_fix))

        epoch_loss = np.mean(epoch_loss)  # average loss per element for whole epoch
        self._logger.info("Train: expid = {}, Epoch = {}, avg_loss = {}.".format(
            self.exp_id, epoch_idx, epoch_loss))
        self._writer.add_scalar('Train loss', epoch_loss, epoch_idx)
        return epoch_loss

    def _valid_epoch(self, eval_dataloader, epoch_idx, mode='Eval'):
        self.model = self.model.eval()

        epoch_loss = []

        with torch.no_grad():
            for i, batch in tqdm(enumerate(eval_dataloader), desc="{} epoch={}".format(
                    mode, epoch_idx), total=len(eval_dataloader)):
                X1, X2, padding_masks1, padding_masks2, batch_temporal_mat1, batch_temporal_mat2 = batch
                # X1/X2: (batch_size, padded_length, feat_dim)
                # padding_masks1/2: (batch_size, padded_length)
                # batch_temporal_mat1/2: (batch_size, padded_length, padded_length)
                X1 = X1.to(self.device)
                X2 = X2.to(self.device)
                padding_masks1 = padding_masks1.to(self.device)  # 0s: ignore
                padding_masks2 = padding_masks2.to(self.device)  # 0s: ignore
                batch_temporal_mat1 = batch_temporal_mat1.to(self.device)
                batch_temporal_mat2 = batch_temporal_mat2.to(self.device)

                z1 = self.model(x=X1, padding_masks=padding_masks1, batch_temporal_mat=batch_temporal_mat1,
                                argument_methods=self.data_argument1,
                                graph_dict=self.graph_dict)  # (batch_size, d_model)
                z2 = self.model(x=X2, padding_masks=padding_masks2, batch_temporal_mat=batch_temporal_mat2,
                                argument_methods=self.data_argument2,
                                graph_dict=self.graph_dict)  # (batch_size, d_model)

                mean_loss = self._contrastive_loss(z1, z2, self.contra_loss_type)

                if self.test_align_uniform or self.train_align_uniform:
                    align_uniform_loss, align_loss, uniform_loss = self.align_uniform(z1, z2)
                    if self.train_align_uniform:
                        mean_loss += align_uniform_loss

                epoch_loss.append(mean_loss.item())

                post_fix = {
                    "mode": mode,
                    "epoch": epoch_idx,
                    "iter": i,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "loss": mean_loss.item(),
                }
                if self.test_align_uniform or self.train_align_uniform:
                    post_fix['align_loss'] = align_loss
                    post_fix['uniform_loss'] = uniform_loss
                if i % self.log_batch == 0:
                    self._logger.info(str(post_fix))

            epoch_loss = np.mean(epoch_loss)  # average loss per element for whole epoch
            self._logger.info("{}: expid = {}, Epoch = {}, avg_loss = {}.".format(
                mode, self.exp_id, epoch_idx, epoch_loss))
            self._writer.add_scalar('{} loss'.format(mode), epoch_loss, epoch_idx)

            return epoch_loss
