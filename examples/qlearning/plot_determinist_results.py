import os
import sys
from os.path import dirname, join

import numpy as np
import matplotlib.pyplot as plt

from pandas import read_csv


def plot_results(reader):
    plt.xlabel("Step")
    plt.ylabel("Weight")

    w = np.array([])

    for i in range(reader.shape[1] - 1):
        w_i = reader['w_' + str(i)].tolist()

        if not all(x == w_i[0] for x in w_i):
            plt.plot(w_i, label="w_" + str(i))
            w = np.append(w, w_i)

    w.flatten()

    # plt.text(5, 10, 'α = 0.95\nε = 0.2\nγ = 0.95', style='italic', bbox={'alpha': 0., 'pad': 10})

    plt.xlim([0, reader.shape[0] - 1])
    plt.ylim([min(w) - 0.5, max(w) + 0.5])

    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)


def main(argv):
    mean = 100

    ql_path = argv[1]
    rdm_path = argv[2]

    # Creating results directory if it doesn't exist
    os.makedirs(dirname(join("results_" + ql_path, "file")), exist_ok=True)


    # addA weights evolution over steps
    addA_reader = read_csv("./" + ql_path + "/addA_weights.csv")

    plt.figure("addA_weights", figsize=(16, 9))
    # plt.title("addA weights")

    plot_results(addA_reader)
    plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=14, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/addA_weights.png")


    # rmA weights evolution over steps
    rmA_reader = read_csv("./" + ql_path + "/rmA_weights.csv")

    plt.figure("rmA_weights", figsize=(16, 9))
    # plt.title("rmA weights")

    plot_results(rmA_reader)
    plt.legend(bbox_to_anchor=(0.5, 1.), loc='upper center', ncol=14, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/rmA_weights.png")


    # chB weights evolution over steps
    chB_reader = read_csv("./" + ql_path + "/chB_weights.csv")

    plt.figure("chB_weights", figsize=(16, 9))
    # plt.title("chB weights")

    plot_results(chB_reader)
    plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=14, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/chB_weights.png")

    ql_steps_reader = read_csv("./" + ql_path + "/steps.csv")
    rdm_steps_reader = read_csv("./" + rdm_path + "/steps.csv")


    # Mean of weights evolution over steps for each action
    plt.figure("Moyennes des poids par steps", figsize=(16, 9))
    # plt.title("Moyennes des poids par steps")

    plt.xlabel("Steps", fontsize=20)
    plt.ylabel("Moyenne des poids", fontsize=20)

    addA_w_mean = [np.array([addA_reader.iloc[i][1:].tolist()]).mean() for i in range(addA_reader.shape[0] - 1)]
    rmA_w_mean = [np.array([rmA_reader.iloc[i][1:].tolist()]).mean() for i in range(rmA_reader.shape[0] - 1)]
    chB_w_mean = [np.array([chB_reader.iloc[i][1:].tolist()]).mean() for i in range(chB_reader.shape[0] - 1)]

    plt.plot(range(addA_reader.shape[0] - 1), addA_w_mean, label="addA", color='dodgerblue')
    plt.plot(range(rmA_reader.shape[0] - 1), rmA_w_mean, label="rmA", color='firebrick')
    plt.plot(range(chB_reader.shape[0] - 1), chB_w_mean, label="chB", color='forestgreen')

    plt.xlim([0, addA_reader.shape[0] - 1])
    plt.ylim([min([min(addA_w_mean), min(rmA_w_mean), min(chB_w_mean)]) - 0.5,
              max([max(addA_w_mean), max(rmA_w_mean), max(chB_w_mean)]) + 0.5])

    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/mean_weights.png")


    # Execution time evolution over steps
    plt.figure("Temps d'exécution par steps", figsize=(16, 9))
    # plt.title("Temps d'exécution par steps")

    plt.xlabel("Step")
    plt.ylabel("Time (min)")

    ql_timestamps = np.array(ql_steps_reader['timestamps'].tolist()) / 60
    rd_timestamps = np.array(rdm_steps_reader['timestamps'].tolist()) / 60

    plt.plot(ql_timestamps, label="Q-Learning", color='dodgerblue')
    plt.plot(rd_timestamps, label="Random", color='firebrick')

    plt.annotate('%0.2f' % ql_timestamps[-1], xy=(1, ql_timestamps[-1]), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.annotate('%0.2f' % rd_timestamps[-1], xy=(1, rd_timestamps[-1]), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')

    plt.xlim([0, len(ql_timestamps)])
    plt.ylim([0, max(ql_timestamps) + 5])

    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

    plt.legend()
    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/timestamps.png")


    # Number of tabu fails over steps
    plt.figure("Nombre de tabu fails par steps", figsize=(16, 9))
    # plt.title("Nombre de tabu fails par steps")

    plt.xlabel("Step")
    plt.ylabel("Échecs")

    ql_tabu = np.array(ql_steps_reader['n_discarded_tabu'].tolist())
    rdm_tabu = np.array(rdm_steps_reader['n_discarded_tabu'].tolist())

    plt.plot(ql_tabu, label="Q-Learning", color='dodgerblue')
    plt.plot(rdm_tabu, label="Random", color='firebrick')

    plt.fill_between(np.array(range(len(ql_tabu))), ql_tabu, 0, color='dodgerblue', alpha=.25)
    plt.fill_between(np.array(range(len(rdm_tabu))), rdm_tabu, 0, color='firebrick', alpha=.25)

    plt.xlim([0, len(ql_tabu)])
    plt.ylim([0, max(ql_tabu + rdm_tabu) + 5])

    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

    plt.legend()
    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/tabu_fails.png")


    # Mean of number of tabu fails over steps
    plt.figure("Moyennes du nombre de molécules ne passant pas le filtre tabu par steps "
               "(division par " + str(int(np.floor(len(ql_tabu) / mean))) + ")", figsize=(16, 9))
    # plt.title("Moyennes du nombre de molécules ne passant pas le filtre tabu par steps "
    #           "(division par " + str(int(np.floor(len(ql_tabu) / mean))) + ")")

    plt.xlabel("Steps", fontsize=20)
    plt.ylabel("Nombre d'échecs", fontsize=20)

    ql_mean = [ql_tabu[i : i + int(np.floor(len(ql_tabu) / mean))].mean()
               for i in range(0, len(ql_tabu), int(np.floor(len(ql_tabu) / mean)))]

    rdm_mean = [rdm_tabu[i : i + int(np.floor(len(rdm_tabu) / mean))].mean()
                for i in range(0, len(rdm_tabu), int(np.floor(len(rdm_tabu) / mean)))]

    plt.plot(np.array(range(0, len(ql_tabu), int(np.floor(len(ql_tabu) / mean)))), ql_mean, color='dodgerblue', label="Q-Learning")
    plt.plot(np.array(range(0, len(rdm_tabu), int(np.floor(len(rdm_tabu) / mean)))), rdm_mean, color='firebrick', label="Random")

    plt.fill_between(np.array(range(0, len(ql_tabu), int(np.floor(len(ql_tabu) / mean)))), ql_mean, 0, color='dodgerblue', alpha=.25)
    plt.fill_between(np.array(range(0, len(rdm_tabu), int(np.floor(len(rdm_tabu) / mean)))), rdm_mean, 0, color='firebrick', alpha=.25)

    ql_total_mean = np.array(ql_tabu).mean()
    rdm_total_mean = np.array(rdm_tabu).mean()

    plt.plot(np.array(range(0, len(ql_tabu), int(np.floor(len(ql_tabu) / mean)))), [ql_total_mean] * len(ql_mean), color='dodgerblue', linestyle='--', label="Moyenne Q-Learning")
    plt.plot(np.array(range(0, len(rdm_tabu), int(np.floor(len(rdm_tabu) / mean)))), [rdm_total_mean] * len(rdm_mean), color='firebrick', linestyle='--', label="Moyenne Random")

    plt.annotate(int(np.ceil(ql_total_mean)), xy=(1, ql_total_mean), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points', fontsize=20)
    plt.annotate(int(np.ceil(rdm_total_mean)), xy=(1, rdm_total_mean), xytext=(8, 0),
                    xycoords=('axes fraction', 'data'), textcoords='offset points', fontsize=20)

    plt.xlim([0, len(ql_tabu)])
    plt.ylim([0, max(ql_mean + rdm_mean) + 5])

    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/tabu_fails_means.png")


    # Number of molecules discarded by the silly walks filter over steps
    ql_silly_walks_reader = read_csv("./" + ql_path + "/steps.csv")
    rdm_silly_walks_reader = read_csv("./" + rdm_path + "/steps.csv")

    ql_silly_walks = np.array(ql_silly_walks_reader['n_discarded_sillywalks'])
    rdm_silly_walks = np.array(rdm_silly_walks_reader['n_discarded_sillywalks'])

    plt.figure("Nombre de molécules ne passant pas le filtre des silly walks par steps", figsize=(16, 9))
    # plt.title("Nombre de molécules ne passant pas le filtre des silly walks par steps")

    plt.xlabel("Steps")
    plt.ylabel("Nombre d'échecs")

    plt.plot(np.array(range(len(ql_silly_walks))), ql_silly_walks,
            color='dodgerblue', label="Q-Learning")

    plt.plot(np.array(range(len(rdm_silly_walks))), rdm_silly_walks,
            color='firebrick', label="Random")

    plt.fill_between(np.array(range(len(ql_silly_walks))), ql_silly_walks,
                     0, color='dodgerblue', alpha=.25)

    plt.fill_between(np.array(range(len(rdm_silly_walks))), rdm_silly_walks,
                     0, color='firebrick', alpha=.25)

    plt.xlim([0, len(ql_silly_walks)])
    plt.ylim([0, max(ql_silly_walks + rdm_silly_walks) + 5])

    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

    plt.legend()
    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/silly_walks_fails.png")

    # Mean of number of molecules discarded by the silly walks filter over steps
    plt.figure("Moyennes du nombre de molécules ne passant pas le filtre des silly walks par steps "
               "(division par " + str(int(np.floor(len(ql_silly_walks) / mean))) + ")", figsize=(16, 9))
    # plt.title("Moyennes du nombre de molécules ne passant pas le filtre des silly walks par steps "
    #           "(division par " + str(int(np.floor(len(ql_silly_walks) / mean))) + ")")

    plt.xlabel("Steps", fontsize=20)
    plt.ylabel("Nombre d'échecs", fontsize=20)

    ql_mean = [ql_silly_walks[i : i + int(np.floor(len(ql_silly_walks) / mean))].mean()
               for i in range(0, len(ql_silly_walks), int(np.floor(len(ql_silly_walks) / mean)))]

    rdm_mean = [rdm_silly_walks[i : i + int(np.floor(len(rdm_silly_walks) / mean))].mean()
                for i in range(0, len(rdm_silly_walks), int(np.floor(len(rdm_silly_walks) / mean)))]

    plt.plot(np.array(range(0, len(ql_silly_walks), int(np.floor(len(ql_silly_walks) / mean)))), ql_mean, color='dodgerblue', label="Q-Learning")
    plt.plot(np.array(range(0, len(rdm_silly_walks), int(np.floor(len(rdm_silly_walks) / mean)))), rdm_mean, color='firebrick', label="Random")

    plt.fill_between(np.array(range(0, len(ql_silly_walks), int(np.floor(len(ql_silly_walks) / mean)))), ql_mean, 0, color='dodgerblue', alpha=.25)
    plt.fill_between(np.array(range(0, len(rdm_silly_walks), int(np.floor(len(rdm_silly_walks) / mean)))), rdm_mean, 0, color='firebrick', alpha=.25)

    ql_total_mean = np.array(ql_silly_walks).mean()
    rdm_total_mean = np.array(rdm_silly_walks).mean()

    plt.plot(np.array(range(0, len(ql_silly_walks), int(np.floor(len(ql_silly_walks) / mean)))), [ql_total_mean] * len(ql_mean), color='dodgerblue', linestyle='--', label="Moyenne Q-Learning")
    plt.plot(np.array(range(0, len(rdm_silly_walks), int(np.floor(len(rdm_silly_walks) / mean)))), [rdm_total_mean] * len(rdm_mean), color='firebrick', linestyle='--', label="Moyenne Random")

    plt.annotate(int(np.ceil(ql_total_mean)), xy=(1, ql_total_mean), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points', fontsize=20)
    plt.annotate(int(np.ceil(rdm_total_mean)), xy=(1, rdm_total_mean), xytext=(8, 0),
                    xycoords=('axes fraction', 'data'), textcoords='offset points', fontsize=20)

    plt.xlim([0, len(ql_silly_walks)])
    plt.ylim([0, max(ql_mean + rdm_mean) + 5])

    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/silly_walks_fail_means.png")

    plt.show()


if __name__ == "__main__":
    main(sys.argv)
