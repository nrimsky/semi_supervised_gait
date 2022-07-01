import glob
import os
from collections import defaultdict

SO = 'Supervised only'


def get_results_files():
    return glob.glob(os.path.join("results", "*.txt"))


def read_results(filenames):
    errs = defaultdict(list)
    accs = defaultdict(list)
    for n in filenames:
        with open(n, "r") as txtfile:
            lines = txtfile.readlines()
            s_o_a = 0
            s_o_e = 0
            for l in lines:
                if " : " in l:
                    approach, scores = l.split(" : ")
                    acc, err = scores.split(",")
                    acc, err = float(acc), float(err)
                    if approach == SO:
                        s_o_e = err
                        s_o_a = acc
                    else:
                        err = (s_o_e - err) / s_o_e
                        acc = (acc - s_o_a) / s_o_a
                    accs[approach].append(acc)
                    errs[approach].append(err)
    mean_errs = {a: sum(e) / len(e) for a, e in errs.items()}
    mean_accs = {a: sum(ac) / len(ac) for a, ac in accs.items()}
    print(mean_errs)
    print(mean_accs)


if __name__ == '__main__':
    fns = get_results_files()
    read_results(fns)
