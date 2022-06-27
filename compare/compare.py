from supervised.trainer import StandardTrainer
from simclr.trainer import SimCLRTrainer
from pseudolabelling.trainer import PseudolabellingTrainer
from meta_pseudo_labels.trainer import MPLTrainer
from grl.trainer import GRLTrainer
from random import shuffle
from datetime import datetime
from helpers import ensure_dir
import os


class ApproachComparer:

    def __init__(self, nn_trainers: dict, results_dir="results"):
        self.trainers = nn_trainers
        self.results_dir = results_dir

    def run(self, unsupervised_subjects, supervised_subjects, proportion_unlabelled, identifier=""):
        ensure_dir(self.results_dir, empty=False)
        with open(os.path.join(self.results_dir, f"results_{datetime.now().strftime('%d%m_%H%M')}_{identifier}.txt"), "w") as results_file:
            results_file.write(f"{supervised_subjects=}\n")
            results_file.write(f"{unsupervised_subjects=}\n")
            for approach_name, trainer in self.trainers.items():
                acc, circular_error = trainer.train(unsupervised_subjects=unsupervised_subjects,
                                                    supervised_subjects=supervised_subjects,
                                                    proportion_unlabelled=proportion_unlabelled,
                                                    filename=f"{approach_name.replace(' ', '_')}.txt")
                print(approach_name, acc, circular_error)
                results_file.write(f"{approach_name} : {acc},{circular_error}\n")
                if approach_name == "Supervised only" and acc < 0.3:
                    break


if __name__ == '__main__':

    for i in range(5):
        subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21]
        shuffle(subjects)
        trainers = {
            "Supervised only": StandardTrainer(),
            "SimCLR": SimCLRTrainer(),
            "Pseudo-labelling": PseudolabellingTrainer(),
            "Meta Pseudo Labels": MPLTrainer(),
            "GRL": GRLTrainer(),
        }
        split = int(len(subjects) * 0.8)
        ac = ApproachComparer(nn_trainers=trainers)
        ac.run(unsupervised_subjects=subjects[0:split], supervised_subjects=subjects[split:], proportion_unlabelled=0.8, identifier="")
