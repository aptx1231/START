import os
import time
import torch
import numpy as np
from libcity.executor.bert_executor import BertBaseExecutor
from libcity.model import loss
import torch.nn.functional as F
from tqdm import tqdm


class ContrastiveExecutor(BertBaseExecutor):
    def __init__(self, config, model, data_feature):
        super().__init__(config, model, data_feature)
        self.batch_size = self.config.get("batch_size", 64)
        self.n_views = self.config.get("n_views", 2)
        self.similarity = self.config.get("similarity", 'inner')  # or cosine
        self.temperature = self.config.get("temperature", 0.05)
        self.contra_loss_type = self.config.get("contra_loss_type", 'simclr').lower()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.data_argument1 = self.config.get("data_argument1", ['shuffle_position'])
        self.data_argument2 = self.config.get("data_argument2", ['shuffle_position'])

        self.align_w = self.config.get("align_w", 1.)
        self.unif_w = self.config.get("unif_w", 1.)
        self.align_alpha = self.config.get("align_alpha", 2)
        self.unif_t = self.config.get("unif_t", 2)
        self.train_align_uniform = self.config.get("train_align_uniform", False)
        self.test_align_uniform = self.config.get("test_align_uniform", True)
        self.norm_align_uniform = self.config.get("norm_align_uniform", False)

    def align_loss(self, x, y, alpha=2):
        if self.norm_align_uniform:
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    def uniform_loss(self, x, t=2):
        if self.norm_align_uniform:
            x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def align_uniform(self, x, y):
        align_loss_val = self.align_loss(x, y, alpha=self.align_alpha)
        unif_loss_val = (self.uniform_loss(x, t=self.unif_t) + self.uniform_loss(y, t=self.unif_t)) / 2
        sum_loss = align_loss_val * self.align_w + unif_loss_val * self.unif_w
        return sum_loss, align_loss_val.item(), unif_loss_val.item()

    def _contrastive_loss(self, z1, z2, loss_type):
        if loss_type == 'simsce':
            return self._contrastive_loss_simsce(z1, z2)
        elif loss_type == 'simclr':
            return self._contrastive_loss_simclr(z1, z2)
        elif loss_type == 'consert':
            return self._contrastive_loss_consert(z1, z2)
        else:
            raise ValueError('Error contrastive loss type {}!'.format(loss_type))

    def _contrastive_loss_simsce(self, z1, z2):
        assert z1.shape == z2.shape
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        if self.similarity == 'inner':
            similarity_matrix = torch.matmul(z1, z2.T)
        elif self.similarity == 'cosine':
            similarity_matrix = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1)
        else:
            similarity_matrix = torch.matmul(z1, z2.T)
        similarity_matrix /= self.temperature

        labels = torch.arange(similarity_matrix.shape[0]).long().to(self.device)
        loss_res = self.criterion(similarity_matrix, labels)
        return loss_res

    def _contrastive_loss_simclr(self, z1, z2):
        """

        Args:
            z1(torch.tensor): (batch_size, d_model)
            z2(torch.tensor): (batch_size, d_model)

        Returns:

        """
        assert z1.shape == z2.shape
        batch_size, d_model = z1.shape
        features = torch.cat([z1, z2], dim=0)  # (batch_size * 2, d_model)

        labels = torch.cat([torch.arange(batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)
        if self.similarity == 'inner':
            similarity_matrix = torch.matmul(features, features.T)
        elif self.similarity == 'cosine':
            similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        else:
            similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # [batch_size * 2, 1]

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # [batch_size * 2, 2N-2]

        logits = torch.cat([positives, negatives], dim=1)  # (batch_size * 2, batch_size * 2 - 1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)  # (batch_size * 2, 1)
        logits = logits / self.temperature

        loss_res = self.criterion(logits, labels)
        return loss_res

    def _contrastive_loss_consert(self, z1, z2):
        """

        Args:
            z1(torch.tensor): (batch_size, d_model)
            z2(torch.tensor): (batch_size, d_model)

        Returns:

        """
        assert z1.shape == z2.shape
        batch_size, d_model = z1.shape

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        hidden1_large = z1
        hidden2_large = z2

        labels = torch.arange(0, batch_size).to(device=self.device)
        masks = F.one_hot(torch.arange(0, batch_size), num_classes=batch_size).to(
            device=self.device, dtype=torch.float)

        if self.similarity == 'inner':
            logits_aa = torch.matmul(z1, hidden1_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
            logits_bb = torch.matmul(z2, hidden2_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
            logits_ab = torch.matmul(z1, hidden2_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
            logits_ba = torch.matmul(z2, hidden1_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
        elif self.similarity == 'cosine':
            logits_aa = F.cosine_similarity(z1.unsqueeze(1), hidden1_large.unsqueeze(0), dim=-1) / self.temperature  # shape (bsz, bsz)
            logits_bb = F.cosine_similarity(z2.unsqueeze(1), hidden2_large.unsqueeze(0), dim=-1) / self.temperature  # shape (bsz, bsz)
            logits_ab = F.cosine_similarity(z1.unsqueeze(1), hidden2_large.unsqueeze(0), dim=-1) / self.temperature  # shape (bsz, bsz)
            logits_ba = F.cosine_similarity(z2.unsqueeze(1), hidden1_large.unsqueeze(0), dim=-1) / self.temperature  # shape (bsz, bsz)
        else:
            logits_aa = torch.matmul(z1, hidden1_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
            logits_bb = torch.matmul(z2, hidden2_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
            logits_ab = torch.matmul(z1, hidden2_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
            logits_ba = torch.matmul(z2, hidden1_large.transpose(0, 1)) / self.temperature  # shape (bsz, bsz)
        logits_aa = logits_aa - masks * 1e9
        logits_bb = logits_bb - masks * 1e9
        loss_a = self.criterion(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = self.criterion(torch.cat([logits_ba, logits_bb], dim=1), labels)
        return loss_a + loss_b

    def train(self, train_dataloader, eval_dataloader, test_dataloader=None):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = -1
        train_time = []
        eval_time = []
        train_loss = []
        eval_loss = []
        lr_list = []

        num_batches = len(train_dataloader)
        self._logger.info("Num_batches: train={}, eval={}".format(num_batches, len(eval_dataloader)))

        for epoch_idx in range(self.epochs):
            start_time = time.time()
            train_avg_loss = self._train_epoch(train_dataloader, epoch_idx)
            t1 = time.time()
            train_time.append(t1 - start_time)
            train_loss.append(train_avg_loss)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            eval_avg_loss = self._valid_epoch(eval_dataloader, epoch_idx, mode='Eval')
            end_time = time.time()
            eval_time.append(end_time - t2)
            eval_loss.append(eval_avg_loss)

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

        self._draw_png([(train_loss, eval_loss, 'loss'), (lr_list, 'lr')])
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx):
        batches_seen = epoch_idx * len(train_dataloader)  # 总batch数

        self.model = self.model.train()

        epoch_loss = []

        for i, batch in tqdm(enumerate(train_dataloader), desc="Train epoch={}".format(
                epoch_idx), total=len(train_dataloader)):
            X1, X2, padding_masks, batch_temporal_mat = batch
            # X1/X2: (batch_size, padded_length, feat_dim)
            # padding_masks: (batch_size, padded_length)
            # batch_temporal_mat: (batch_size, padded_length, padded_length)
            X1 = X1.to(self.device)
            X2 = X2.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore
            batch_temporal_mat = batch_temporal_mat.to(self.device)

            graph_dict = self.graph_dict

            z1 = self.model(x=X1, padding_masks=padding_masks, batch_temporal_mat=batch_temporal_mat,
                            argument_methods=self.data_argument1, graph_dict=graph_dict)  # (batch_size, d_model)
            z2 = self.model(x=X2, padding_masks=padding_masks, batch_temporal_mat=batch_temporal_mat,
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
                X1, X2, padding_masks, batch_temporal_mat = batch
                # X1/X2: (batch_size, padded_length, feat_dim)
                # padding_masks: (batch_size, padded_length)
                # batch_temporal_mat: (batch_size, padded_length, padded_length)
                X1 = X1.to(self.device)
                X2 = X2.to(self.device)
                padding_masks = padding_masks.to(self.device)  # 0s: ignore
                batch_temporal_mat = batch_temporal_mat.to(self.device)

                z1 = self.model(x=X1, padding_masks=padding_masks, batch_temporal_mat=batch_temporal_mat,
                                argument_methods=self.data_argument1,
                                graph_dict=self.graph_dict)  # (batch_size, d_model)
                z2 = self.model(x=X2, padding_masks=padding_masks, batch_temporal_mat=batch_temporal_mat,
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
