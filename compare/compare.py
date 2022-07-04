from supervised.trainer import StandardTrainer
from simclr.trainer import SimCLRTrainer
from pseudolabelling.trainer import PseudolabellingTrainer
from meta_pseudo_labels.trainer import MPLTrainer
from grl.trainer import GRLTrainer
from random import shuffle
from datetime import datetime
from helpers import ensure_dir, get_loaders, write_and_print
import os
from globals import FRACTION_UNLABELLED, FRACTION_TEST


class ApproachComparer:

    def __init__(self, nn_trainers: dict, results_dir="results"):
        self.trainers = nn_trainers
        self.results_dir = results_dir

    def run(self, unsupervised_subjects, supervised_subjects, labelled_loader, unlabelled_loader, test_loader, identifier=""):
        ensure_dir(self.results_dir, empty=False)
        with open(os.path.join(self.results_dir, f"results_{datetime.now().strftime('%d%m_%H%M')}_{identifier}.txt"), "w") as results_file:

            write_and_print(f"n_batches \t labelled: {len(labelled_loader)} \t unlabelled: {len(unlabelled_loader)} \t test: {len(test_loader)}", results_file)

            results_file.write(f"{supervised_subjects=}\n")
            results_file.write(f"{unsupervised_subjects=}\n")
            for approach_name, trainer in self.trainers.items():
                acc, circular_error = trainer.train(labelled_dataloader=labelled_loader,
                                                    unlabelled_dataloader=unlabelled_loader,
                                                    test_dataloader=test_loader,
                                                    filename=f"{approach_name.replace(' ', '_')}.txt")
                print(approach_name, acc, circular_error)
                results_file.write(f"{approach_name} : {acc},{circular_error}\n")
                if approach_name == "Supervised only" and acc < 0.3:
                    break


def main(n_trials):

    for i in range(n_trials):
        subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 28, 29, 32, 33, 901]
        subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 32, 33, 901]
        shuffle(subjects)
        trainers = {
            "Supervised only": StandardTrainer(),
            "SimCLR": SimCLRTrainer(),
            "Pseudo-labelling": PseudolabellingTrainer(),
            "Meta Pseudo Labels": MPLTrainer(),
            "GRL": GRLTrainer(),
        }
        split = int(len(subjects) * (FRACTION_UNLABELLED + FRACTION_TEST))
        unlabelled_subjects = subjects[0:split]
        labelled_subjects = subjects[split:]
        unlabelled_loader, labelled_loader, test_loader = get_loaders(labelled_subjects=labelled_subjects, unlabelled_subjects=unlabelled_subjects)
        ac = ApproachComparer(nn_trainers=trainers)
        ac.run(unsupervised_subjects=unlabelled_subjects,
               supervised_subjects=labelled_subjects,
               labelled_loader=labelled_loader,
               unlabelled_loader=unlabelled_loader,
               test_loader=test_loader,
               identifier="")


if __name__ == '__main__':
    main(10)
