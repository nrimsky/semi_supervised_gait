from torch.autograd import Function
import torch as t
from helpers import ensure_dir, format_loss, BaseTrainer, write_and_print, augment, evaluate
from shared_components import Body, ClassificationHead, ProjectionHead
import torch.optim as optim
from torch.utils.data import DataLoader
from globals import NORMAL_LR, GRL_ALPHA, N_EPOCHS


class ReverseLayer(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Model(t.nn.Module):

    def __init__(self, augment_input=True):
        super().__init__()
        self.body = Body()
        self.class_classification_head = ClassificationHead()
        self.domain_classification_head = ProjectionHead(output_dim=2)
        self.augment = augment_input

    def forward(self, input_data, alpha=0):
        if self.augment and self.training:
            with t.no_grad():
                input_data = augment(input_data)
        feature = self.body(input_data)
        reverse_feature = ReverseLayer.apply(feature, alpha)
        class_output = self.class_classification_head(feature)
        domain_output = self.domain_classification_head(reverse_feature)
        return class_output, domain_output


class GRLTrainer(BaseTrainer):

    def __init__(self, lr=NORMAL_LR):
        self.lr = lr

    def train(self, labelled_dataloader: DataLoader, unlabelled_dataloader: DataLoader, test_dataloader: DataLoader, filename: str) -> float:
        n_steps = len(labelled_dataloader) * N_EPOCHS
        base_name = filename.split(".")[0]
        ensure_dir(base_name)

        # load model

        model = Model()
        model.train()

        # setup optimizer

        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        criterion = t.nn.CrossEntropyLoss()

        model = model.cuda()

        target_iter = iter(unlabelled_dataloader)
        source_iter = iter(labelled_dataloader)

        # training

        avg_err_s_label = 0
        avg_err_s_domain = 0
        avg_err_t_domain = 0


        with open(filename, "w") as txtfile:
            for step in range(n_steps):
                try:
                    batch_s, labels_s = next(source_iter)
                except StopIteration:
                    source_iter = iter(labelled_dataloader)
                    batch_s, labels_s = next(source_iter)
                try:
                    batch_t, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(unlabelled_dataloader)
                    batch_t, _ = next(target_iter)

                alpha = GRL_ALPHA

                batch_size = len(labels_s)
                s_data = batch_s.cuda().float()
                s_label = labels_s.cuda().long()
                domain_label = t.zeros(batch_size, device='cuda', dtype=t.long)

                class_output, domain_output = model(input_data=s_data, alpha=alpha)
                err_s_label = criterion(class_output, s_label)
                err_s_domain = criterion(domain_output, domain_label)

                t_data = batch_t.cuda().float()
                domain_label = t.ones(batch_size, device='cuda', dtype=t.long)
                _, domain_output = model(input_data=t_data, alpha=alpha)
                err_t_domain = criterion(domain_output, domain_label)
                err = err_t_domain + err_s_domain + err_s_label
                err.backward()
                optimizer.step()
                optimizer.zero_grad()

                avg_err_s_label += err_s_label.cpu().item()
                avg_err_s_domain += err_s_domain.cpu().item()
                avg_err_t_domain += err_t_domain.cpu().item()

                if step % (n_steps // 30) == 0 and step != 0:
                    write_and_print(f'step: {step} : '
                                    f'err_s_label: {format_loss(avg_err_s_label / (n_steps // 30))}, '
                                    f'err_s_domain: {format_loss(avg_err_s_domain / (n_steps // 30))}, '
                                    f'err_t_domain: {format_loss(avg_err_t_domain / (n_steps // 30))}', txtfile)
                    avg_err_s_label = 0
                    avg_err_s_domain = 0
                    avg_err_t_domain = 0
                if step % (n_steps // 10) == 0 and step != 0:
                    model.eval()
                    acc = evaluate(model=model, data_loader=test_dataloader)
                    write_and_print(f"Accuracy: {acc}", txtfile)
                    model.train()
            model.eval()
            acc = evaluate(model=model, data_loader=test_dataloader, use_circ=True)
            write_and_print(f"Accuracy: {acc}", txtfile)
            model.class_classification_head.visualise_embeddings(f"{base_name}_embeddings")
            print('done')

        return acc


