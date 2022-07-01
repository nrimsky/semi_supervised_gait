import torch as t
from helpers import evaluate, format_loss, BaseTrainer, write_and_print, RotationTransform, NoiseTransform, ChannelDeletionTransform
from shared_components import ClassificationHead
from simclr.SimCLR import ConvNetSimCLR, SimCLR
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import transforms
from globals import N_EPOCHS, SIMCLR_LR, FINE_TUNE_LR, BATCH_SIZE


class ContrastiveLearningViewGenerator(object):

    def __init__(self, rotation_std=40, noise_scale=0.2, erasing_scale=(0.1, 0.3), n_channels_delete=1, n_views=2):
        self.base_transform = transforms.Compose([RotationTransform(std=rotation_std),
                                                  ChannelDeletionTransform(channel_dim=-3, n_channels=n_channels_delete),
                                                  transforms.RandomErasing(p=1, scale=erasing_scale),
                                                  NoiseTransform(scale=noise_scale)])
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]


class SimCLRTrainer(BaseTrainer):

    def __init__(self, n_epochs_unsupervised=N_EPOCHS, n_epochs_supervised=N_EPOCHS, lr_unsupervised=SIMCLR_LR, lr_supervised=FINE_TUNE_LR):
        self.n_epochs_unsupervised = n_epochs_unsupervised
        self.n_epochs_supervised = n_epochs_supervised
        self.lr_unsupervised = lr_unsupervised
        self.lr_supervised = lr_supervised


    @staticmethod
    def make_supervised_model(unsupervised_model: ConvNetSimCLR):
        model = ConvNetSimCLR()
        model.eval()
        model.load_state_dict(unsupervised_model.state_dict())
        model.head = ClassificationHead().cuda()
        model.cuda()
        model.train()
        print("Model set up")
        return model

    def unsupervised_pretrain(self, dataloader, logfilename, n_views=2, batch_size=BATCH_SIZE, temperature=0.07) -> ConvNetSimCLR:
        contrastive_view_gen = ContrastiveLearningViewGenerator(n_views=n_views)
        model = ConvNetSimCLR()
        optimizer = t.optim.Adam(model.parameters(), self.lr_unsupervised)
        simclr = SimCLR(model=model,
                        optimizer=optimizer,
                        device="cuda",
                        batch_size=batch_size,
                        n_views=n_views,
                        temperature=temperature,
                        epochs=self.n_epochs_unsupervised,
                        filename=logfilename)
        model = simclr.train(dataloader, view_generator=contrastive_view_gen)
        return model

    def finetune(self, pretrained_model, train_loader, test_loader, filename, freeze_body=False) -> Tuple[ConvNetSimCLR, float]:
        if freeze_body:
            optimizer = t.optim.Adam(pretrained_model.head.parameters(), lr=self.lr_supervised)  # Only update head!
        else:
            optimizer = t.optim.Adam(pretrained_model.parameters(), lr=self.lr_supervised)
        criterion = t.nn.CrossEntropyLoss()
        acc = 0
        with open(filename, "w") as txtfile:
            for epoch_counter in range(self.n_epochs_supervised):
                write_and_print(f"Epoch: {epoch_counter}", txtfile)
                avg_loss = 0
                for idx, (batch, labels) in enumerate(train_loader):
                    batch = batch.to("cuda", dtype=t.float)
                    labels = labels.to("cuda", dtype=t.long)
                    logits = pretrained_model(batch)
                    loss = criterion(logits, labels)
                    avg_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if idx % 300 == 0:
                        write_and_print(f"Loss: {format_loss(avg_loss / 300)}", txtfile)
                        avg_loss = 0
                pretrained_model.eval()
                acc = evaluate(pretrained_model, test_loader)
                pretrained_model.train()
                write_and_print(f"Accuracy: {acc}", txtfile)
            pretrained_model.eval()
            acc = evaluate(pretrained_model, test_loader, use_circ=True)
        return pretrained_model, acc

    def train(self, labelled_dataloader: DataLoader, unlabelled_dataloader: DataLoader, test_dataloader: DataLoader, filename: str) -> float:
        base_name = filename.split(".")[0]
        pretrained_model = self.unsupervised_pretrain(dataloader=unlabelled_dataloader, logfilename="pretrain_"+filename)
        model = self.make_supervised_model(pretrained_model)
        model, accuracy = self.finetune(pretrained_model=model, train_loader=labelled_dataloader, test_loader=test_dataloader, filename="fine_tune_" + filename)
        model.eval()
        model.head.visualise_embeddings(f"{base_name}_embeddings")
        return accuracy



