import torch as t
from helpers import write_and_print
import torch.nn.functional as F
from shared_components import Body, ProjectionHead
import globals


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with t.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ConvNetSimCLR(t.nn.Module):

    def __init__(self):
        super().__init__()
        self.body = Body()
        self.head = ProjectionHead(output_dim=globals.PROJECTION_DIM, hidden_dim=globals.SIMCLR_HIDDEN_DIM)

    def forward(self, x):
        return self.head(self.body(x))


class SimCLR(object):
    """
    Adapted from https://github.com/sthalles/SimCLR/
    """

    def __init__(self, model, optimizer, device, batch_size, n_views, temperature, epochs, filename):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.n_views = n_views
        self.temperature = temperature
        self.filename = filename
        self.criterion = t.nn.CrossEntropyLoss().to(self.device)
        self.epochs = epochs

    def info_nce_loss(self, features):
        labels = t.cat([t.arange(self.batch_size) for _ in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        features = F.normalize(features, dim=1)
        similarity_matrix = t.matmul(features, features.T)
        # discard the main diagonal from both: labels and similarities matrix
        mask = t.eye(labels.shape[0], dtype=t.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = t.cat([positives, negatives], dim=1)
        labels = t.zeros(logits.shape[0], dtype=t.long).to(self.device)
        logits = logits / self.temperature
        return logits, labels

    def train(self, train_loader, view_generator) -> ConvNetSimCLR:

        with open(self.filename, "w") as logfile:

            n_iter = 0
            write_and_print(f"Start SimCLR training for {self.epochs} epochs.", logfile)

            for epoch_counter in range(self.epochs):
                write_and_print(f"Epoch: {epoch_counter}", logfile)
                for _batch, _ in train_loader:
                    batch = _batch.to(self.device, dtype=t.float)
                    batch = view_generator(batch)
                    batch = t.cat(batch, dim=0)
                    features = self.model(batch)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if n_iter % 100 == 0:
                        top1, top5 = accuracy(logits, labels, topk=(1, 5))
                        write_and_print(
                            f"n_iter: {n_iter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}\tTop5 accuracy: {top5[0]}",
                            logfile)
                    n_iter += 1
            return self.model
