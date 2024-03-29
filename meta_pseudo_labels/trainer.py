"""
Some code adapted from https://github.com/kekmodel/MPL-pytorch/
"""

import torch as t
from torch.nn.functional import cross_entropy
from helpers import format_loss, evaluate, write_and_print, BaseTrainer, augment
from shared_components import Model
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from torch.utils.data import DataLoader
from globals import FINE_TUNE_LR, N_EPOCHS_FINETUNE, N_EPOCHS, MPL_LR, N_CHANNELS_CONV_LG


class MPLTrainer(BaseTrainer):

    def __init__(self, n_epochs_finetune=N_EPOCHS_FINETUNE, lr=MPL_LR, fine_tune_lr=FINE_TUNE_LR):
        self.n_epochs_finetune = n_epochs_finetune
        self.lr = lr
        self.fine_tune_lr = fine_tune_lr

    def finetune(self, model, data_loader, test_loader, log_file, criterion):
        optimizer = t.optim.Adam(model.parameters(), lr=self.fine_tune_lr)
        write_and_print(f"Fine tuning model for {self.n_epochs_finetune} epochs", log_file)
        model.train()
        period = len(data_loader) // 10
        for epoch in range(self.n_epochs_finetune):
            write_and_print(f"Epoch {epoch}", log_file)
            avg_loss = 0
            for step, (data, labels) in enumerate(data_loader):
                data = data.cuda()
                labels = labels.cuda()
                out = model(data)
                loss = criterion(out, labels)
                avg_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if step % period == 0 and step > 0:
                    write_and_print(f"Loss: {format_loss(avg_loss / period)}", log_file)
                    avg_loss = 0
            accuracy = evaluate(model, test_loader, max_batches=100)
            write_and_print(f"Accuracy after training for {epoch + 1} epochs: {accuracy}", log_file)

    def mpl_train(self, labelled_loader, unlabelled_loader, test_loader, teacher_model, student_model, criterion, t_optimizer, s_optimizer, log_file,
                  base_name) -> float:
        
        n_steps = len(labelled_loader) * N_EPOCHS

        write_and_print(f"Meta Pseudo Labels training for {n_steps} steps", log_file)

        labelled_iter = iter(labelled_loader)
        unlabelled_iter = iter(unlabelled_loader)
        period_evaluate = n_steps // 5
        period_log_loss = n_steps // 10

        avg_s_loss = 0
        avg_t_loss_l = 0
        avg_t_loss_mpl = 0

        for step in range(n_steps):
            teacher_model.train()
            student_model.train()
            try:
                batch_l, labels_l = next(labelled_iter)
            except StopIteration:
                labelled_iter = iter(labelled_loader)
                batch_l, labels_l = next(labelled_iter)
            try:
                batch_u, _ = next(unlabelled_iter)
            except StopIteration:
                unlabelled_iter = iter(unlabelled_loader)
                batch_u, _ = next(unlabelled_iter)

            batch_u = batch_u.cuda()  # Batch of unlabelled data
            batch_l = batch_l.cuda()  # Batch of labelled data
            labels_l = labels_l.cuda()  # Labels for labelled data
            batch_u_a = augment(batch_u).cuda()

            batch_size = batch_l.shape[0]
            t_logits = teacher_model(t.cat((batch_l, batch_u, batch_u_a)))
            t_logits_l, t_logits_u, t_logits_u_a = t_logits[:batch_size], t_logits[batch_size:batch_size*2], t_logits[batch_size*2:]
            t_loss_l = criterion(t_logits_l, labels_l)
            soft_pseudo_label = t.softmax(t_logits_u.detach(), dim=-1)
            max_probs, hard_pseudo_label = t.max(soft_pseudo_label, dim=-1)
            s_logits = student_model(t.cat((batch_l, batch_u)))
            s_logits_l, s_logits_u = s_logits[:batch_size], s_logits[batch_size:]
            s_loss_l_old = cross_entropy(s_logits_l.detach(), labels_l)  # Student's performance on labelled data
            s_loss = criterion(s_logits_u, hard_pseudo_label)  # Student's performance on augmented unlabelled data wrt teacher's hard pseudo label based on unaugmented unlabelled data
            s_loss.backward()
            clip_grad_norm(student_model.parameters(), 10)
            s_optimizer.step()
            with t.no_grad():
                s_logits_l = student_model(batch_l)
            s_loss_l_new = cross_entropy(s_logits_l.detach(), labels_l)  # Student's performance on labelled data after weights updated from pseudo label based update
            diff = s_loss_l_new - s_loss_l_old  # We want new loss to be lower so diff should be negative
            _, hard_pseudo_label = t.max(t_logits_u.detach(), dim=-1)  # Pseudo labels from teach based on augmented unlabelled batch
            t_loss_mpl = diff * cross_entropy(t_logits_u, hard_pseudo_label)  # Changing p(pseudolabel sampled)
            t_loss = t_loss_l + t_loss_mpl
            t_loss.backward()
            clip_grad_norm(teacher_model.parameters(), 10)
            t_optimizer.step()
            t_optimizer.zero_grad()
            s_optimizer.zero_grad()

            avg_s_loss += s_loss.item()
            avg_t_loss_l += t_loss_l.item()
            avg_t_loss_mpl += t_loss_mpl.item()

            if step % period_log_loss == 0 and step != 0:
                write_and_print(
                    f"{step + 1} MPL steps || Student loss on pseudo-labels: {format_loss(avg_s_loss / period_log_loss)}, Teacher loss on labelled data: {format_loss(avg_t_loss_l / period_log_loss)}, Teacher MPL loss: {format_loss(avg_t_loss_mpl / period_log_loss)}",
                    log_file)
                avg_s_loss = 0
                avg_t_loss_l = 0

            if step % period_evaluate == 0 and step != 0:
                student_model.eval()
                accuracy = evaluate(student_model, test_loader)
                student_model.train()
                write_and_print(f"Accuracy after {step + 1} MPL steps: {accuracy}", log_file)

        student_model.head.visualise_embeddings(f"{base_name}_student_embeddings")
        teacher_model.head.visualise_embeddings(f"{base_name}_teacher_embeddings")

        self.finetune(model=student_model, data_loader=labelled_loader, test_loader=test_loader, log_file=log_file, criterion=criterion)

        student_model.eval()
        accuracy = evaluate(student_model, test_loader, use_circ=True)
        student_model.head.visualise_embeddings(f"{base_name}_student_embeddings_finetuned")

        return accuracy

    def train(self, labelled_dataloader: DataLoader, unlabelled_dataloader: DataLoader, test_dataloader: DataLoader, filename: str) -> float:
        teacher_model = Model(channels=N_CHANNELS_CONV_LG, augment_input=True).cuda()
        student_model = Model(augment_input=True).cuda()
        criterion = t.nn.CrossEntropyLoss()
        t_optimiser = t.optim.Adam(teacher_model.parameters(), self.lr)
        s_optimiser = t.optim.Adam(student_model.parameters(), self.lr)
        base_name = filename.split(".")[0]
        with open(filename, "w") as log_file:
            acc = self.mpl_train(labelled_loader=labelled_dataloader,
                                 unlabelled_loader=unlabelled_dataloader,
                                 test_loader=test_dataloader,
                                 teacher_model=teacher_model,
                                 student_model=student_model,
                                 criterion=criterion,
                                 t_optimizer=t_optimiser,
                                 s_optimizer=s_optimiser,
                                 log_file=log_file,
                                 base_name=base_name)
        return acc



