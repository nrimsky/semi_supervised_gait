from shared_components import Model
from helpers import evaluate, BaseTrainer, write_and_print
import torch as t
from torch.utils.data import DataLoader
from globals import NORMAL_LR, N_EPOCHS_BASELINE


class StandardTrainer(BaseTrainer):

    def __init__(self, lr=NORMAL_LR, n_epochs=N_EPOCHS_BASELINE):
        self.lr = lr
        self.n_epochs = n_epochs

    def train(self, labelled_dataloader: DataLoader, unlabelled_dataloader: DataLoader, test_dataloader: DataLoader, filename: str) -> float:
        with open(filename, "w") as txtfile:
            write_and_print(f"LR: {self.lr}", txtfile)
            model = Model(use_dropout=False, augment_input=True)
            t.cuda.empty_cache()
            model.train()
            model.cuda()
            optimiser = t.optim.Adam(model.parameters(), lr=self.lr)
            criterion = t.nn.CrossEntropyLoss()
            l = len(labelled_dataloader)
            period = l // 10
            for epoch in range(self.n_epochs):
                avg_loss = 0
                write_and_print(f"Epoch {epoch + 1}", txtfile)
                for i, (_batch, _label) in enumerate(labelled_dataloader):
                    optimiser.zero_grad()
                    label = _label.long().cuda()
                    batch = _batch.cuda()
                    out = model(batch)
                    loss = criterion(out, label)
                    avg_loss += loss.item()
                    loss.backward()
                    optimiser.step()
                    if i % period == 0 and i != 0:
                        write_and_print(f"Loss = {avg_loss / period}", txtfile)
                        avg_loss = 0
                model.eval()
                with t.no_grad():
                    acc = evaluate(model, test_dataloader)
                model.train()
                write_and_print(f"Test accuracy = {acc}", txtfile)

            model.eval()
            acc = evaluate(model, test_dataloader, use_circ=True)
            model.head.visualise_embeddings(f"{filename.split('.')[0]}_embeddings")
            return acc
