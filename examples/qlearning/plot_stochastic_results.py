import os
import sys
from os.path import dirname, join

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from pandas import read_csv


def plot_results(reader):
    # n = 0
    # m = 0

    plt.xlabel("Step")
    plt.ylabel("Weight")

    w = np.array([])

    # for i in range(reader.shape[1] - 1):
    #     w_i = reader['w_' + str(i)].tolist()
    #
    #     if not all(x == w_i[0] for x in w_i):
    #         n += 1

    # colors = plt.cm.rainbow(np.linspace(0, 1, n))
    # plt.rc('axes', prop_cycle=(cycler('linestyle', [':', '-.', '-', '--'])))

    for i in range(reader.shape[1] - 1):
        w_i = reader['w_' + str(i)].tolist()

        if not all(x == w_i[0] for x in w_i):
            plt.plot(w_i, label="w_" + str(i)) # color=colors[m])
            w = np.append(w, w_i)
            # m += 1

    w.flatten()

    plt.xlim([0, reader.shape[0] - 1])
    plt.ylim([min(w), max(w) + 0.01])

    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

    plt.legend()


def main(argv):
    ql_path = argv[1]
    rdm_path = argv[2]

    # Creating results directory if it doesn't exist
    os.makedirs(dirname(join("results_" + ql_path, "file")), exist_ok=True)

    mean = 100


    # addA success evolution over steps
    addA_success_reader = read_csv("./data/" + ql_path + "/addA_success.csv")

    plt.figure("addA success", figsize=(16, 9))
    # plt.title("addA success")

    plot_results(addA_success_reader)
    plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=14, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/addA_success.png")


    # rmA success evolution over steps
    rmA_success_reader = read_csv("./data/" + ql_path + "/rmA_success.csv")

    plt.figure("rmA success", figsize=(16, 9))
    # plt.title("rmA success")

    plot_results(rmA_success_reader)
    plt.legend(bbox_to_anchor=(0.5, 1.), loc='upper center', ncol=14, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/rmA_success.png")


    # chB usage evolution over steps
    chB_success_reader = read_csv("./data/" + ql_path + "/chB_success.csv")

    plt.figure("chB success", figsize=(16, 9))
    # plt.title("chB success")

    plot_results(chB_success_reader)
    plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=14, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/chB_success.png")


    # addA usage evolution over steps
    addA_usage_reader = read_csv("./data/" + ql_path + "/addA_usage.csv")

    plt.figure("addA usage", figsize=(16, 9))
    # plt.title("addA usage")

    plot_results(addA_usage_reader)
    plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=14, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/addA_usage.png")


    # rmA usage evolution over steps
    rmA_usage_reader = read_csv("./data/" + ql_path + "/rmA_usage.csv")

    plt.figure("rmA usage", figsize=(16, 9))
    # plt.title("rmA usage")

    plot_results(rmA_usage_reader)
    plt.legend(bbox_to_anchor=(0.5, 1.), loc='upper center', ncol=14, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/rmA_usage.png")


    # chB usage evolution over steps
    chB_usage_reader = read_csv("./data/" + ql_path + "/chB_usage.csv")

    plt.figure("chB usage", figsize=(16, 9))
    # plt.title("chB usage")

    plot_results(chB_usage_reader)
    plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=14, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/chB_usage.png")


    # addA success rates evolution over steps
    addA_reader = read_csv("./data/" + ql_path + "/addA_success_rates.csv")

    plt.figure("addA success rates", figsize=(16, 9))
    # plt.title("addA success_rates")

    plot_results(addA_reader)
    plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=14, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/addA_success_rates.png")


    # rmA success rates evolution over steps
    rmA_reader = read_csv("./data/" + ql_path + "/rmA_success_rates.csv")

    plt.figure("rmA success rates", figsize=(16, 9))
    # plt.title("rmA success_rates")

    plot_results(rmA_reader)
    plt.legend(bbox_to_anchor=(0.5, 1.), loc='upper center', ncol=14, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/rmA_success_rates.png")


    # chB success rates evolution over steps
    chB_reader = read_csv("./data/" + ql_path + "/chB_success_rates.csv")

    plt.figure("chB success rates", figsize=(16, 9))
    # plt.title("chB success rates")

    plot_results(chB_reader)
    plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=14, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/chB_success_rates.png")


    # addA success evolution over usage
    plt.figure("addA succès par utilisations", figsize=(16, 9))
    # plt.title("addA succès par utilisations")

    plt.xlabel("Utilisations")
    plt.ylabel("Succès")

    w = np.array([])

    for i in range(addA_usage_reader.shape[1] - 1):
        usage_i = addA_usage_reader['w_' + str(i)].tolist()
        success_i = addA_success_reader['w_' + str(i)].tolist()

        if not all(x == usage_i[0] for x in usage_i):
            plt.plot(np.array(range(len(usage_i))), success_i, label="w_" + str(i))
            w = np.append(w, success_i[-1])

    w.flatten()

    plt.xlim([0, addA_usage_reader.shape[0] - 1])
    plt.ylim([min(w), max(w) + 0.01])

    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

    plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=14, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/addA_success_per_usage.png")


    # rmA success evolution over usage
    plt.figure("rmA succès par utilisations", figsize=(16, 9))
    # plt.title("rmA succès par utilisations")

    plt.xlabel("Utilisations")
    plt.ylabel("Succès")

    w = np.array([])

    for i in range(rmA_usage_reader.shape[1] - 1):
        usage_i = rmA_usage_reader['w_' + str(i)].tolist()
        success_i = rmA_success_reader['w_' + str(i)].tolist()

        if not all(x == usage_i[0] for x in usage_i):
            plt.plot(np.array(range(len(usage_i))), success_i, label="w_" + str(i))
            w = np.append(w, success_i[-1])

    w.flatten()

    plt.xlim([0, rmA_usage_reader.shape[0] - 1])
    plt.ylim([min(w), max(w) + 0.01])

    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

    plt.legend()
    plt.legend(bbox_to_anchor=(0.5, 1.), loc='upper center', ncol=14, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/rmA_success_per_usage.png")


    # chB success evolution over usage
    plt.figure("chB succès par utilisations", figsize=(16, 9))
    # plt.title("chB succès par utilisations")

    plt.xlabel("Utilisations")
    plt.ylabel("Succès")

    w = np.array([])

    for i in range(chB_usage_reader.shape[1] - 1):
        usage_i = chB_usage_reader['w_' + str(i)].tolist()
        success_i = chB_success_reader['w_' + str(i)].tolist()

        if not all(x == usage_i[0] for x in usage_i):
            plt.plot(np.array(range(len(usage_i))), success_i, label="w_" + str(i))
            w = np.append(w, success_i[-1])

    w.flatten()

    plt.xlim([0, chB_usage_reader.shape[0] - 1])
    plt.ylim([min(w), max(w) + 0.01])

    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

    plt.legend()
    plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=14, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/chB_success_per_usage.png")


    # Mean of weights evolution over steps for each action
    plt.figure("Moyennes des success rates par steps", figsize=(16, 9))
    # plt.title("Moyennes des success rates par steps")

    plt.xlabel("Steps")
    plt.ylabel("Moyenne des poids")

    addA_w_mean = [np.array([addA_reader.iloc[i][1:].tolist()]).mean() for i in range(addA_reader.shape[0] - 1)]
    rmA_w_mean = [np.array([rmA_reader.iloc[i][1:].tolist()]).mean() for i in range(rmA_reader.shape[0] - 1)]
    chB_w_mean = [np.array([chB_reader.iloc[i][1:].tolist()]).mean() for i in range(chB_reader.shape[0] - 1)]

    plt.plot(range(addA_reader.shape[0] - 1), [np.array([addA_reader.iloc[i][1:].tolist()]).mean() for i in range(addA_reader.shape[0] - 1)],
             label="addA", color='dodgerblue')
    plt.plot(range(rmA_reader.shape[0] - 1), [np.array([rmA_reader.iloc[i][1:].tolist()]).mean() for i in range(rmA_reader.shape[0] - 1)],
             label="rmA", color='firebrick')
    plt.plot(range(chB_reader.shape[0] - 1), [np.array([chB_reader.iloc[i][1:].tolist()]).mean() for i in range(chB_reader.shape[0] - 1)],
             label="chB", color='forestgreen')

    plt.xlim([0, addA_reader.shape[0] - 1])
    plt.ylim([min([min(addA_w_mean), min(rmA_w_mean), min(chB_w_mean)]),
              max([max(addA_w_mean), max(rmA_w_mean), max(chB_w_mean)]) + 0.01])

    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

    plt.legend()
    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/mean_success_rates.png")


    ql_steps_reader = read_csv("./data/" + ql_path + "/steps.csv")
    rdm_steps_reader = read_csv("../random/" + rdm_path + "/steps.csv")


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

    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

    plt.xlim([0, len(ql_timestamps)])
    plt.ylim([0, max(ql_timestamps) + 5])

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

    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

    plt.xlim([0, len(ql_tabu)])
    plt.ylim([0, max(ql_tabu + rdm_tabu) + 5])

    plt.legend()
    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/tabu_fails.png")


    # Mean of number of tabu fails over steps

    plt.figure("Moyennes du nombre de molécules ne passant pas le filtre tabu par steps "
               "(division par " + str(int(np.floor(len(ql_tabu) / mean))) + ")", figsize=(16, 9))
    # plt.title("Moyennes du nombre de molécules ne passant pas le filtre tabu par steps "
    #           "(division par " + str(int(np.floor(len(ql_tabu) / mean))) + ")")

    plt.xlabel("Steps")
    plt.ylabel("Nombre d'échecs")

    ql_mean = [ql_tabu[i : i + int(np.floor(len(ql_tabu) / mean))].mean()
               for i in range(0, len(ql_tabu), int(np.floor(len(ql_tabu) / mean)))]

    rdm_mean = [rdm_tabu[i : i + int(np.floor(len(rdm_tabu) / mean))].mean()
                for i in range(0, len(rdm_tabu), int(np.floor(len(rdm_tabu) / mean)))]

    plt.plot(np.array(range(0, len(ql_tabu), int(np.floor(len(ql_tabu) / mean)))), ql_mean, color='dodgerblue', label="Q-Learning")
    plt.plot(np.array(range(0, len(rdm_tabu), int(np.floor(len(rdm_tabu) / mean)))), rdm_mean, color='firebrick', label="Random")

    plt.fill_between(np.array(range(0, len(ql_tabu), int(np.floor(len(ql_tabu) / mean)))), ql_mean, 0, color='dodgerblue', alpha=.25)
    plt.fill_between(np.array(range(0, len(rdm_tabu), int(np.floor(len(rdm_tabu) / mean)))), rdm_mean, 0, color='firebrick', alpha=.25)

    ql_total_mean = np.array(ql_tabu[1:]).mean()
    rdm_total_mean = np.array(rdm_tabu[1:]).mean()

    plt.plot(np.array(range(0, len(ql_tabu), int(np.floor(len(ql_tabu) / mean)))), [ql_total_mean] * len(ql_mean), color='dodgerblue', linestyle='--', label="Moyenne Q-Learning")
    plt.plot(np.array(range(0, len(rdm_tabu), int(np.floor(len(rdm_tabu) / mean)))), [rdm_total_mean] * len(rdm_mean), color='firebrick', linestyle='--', label="Moyenne Random")

    plt.annotate(int(np.ceil(ql_total_mean)), xy=(1, ql_total_mean), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.annotate(int(np.ceil(rdm_total_mean)), xy=(1, rdm_total_mean), xytext=(8, 0),
                    xycoords=('axes fraction', 'data'), textcoords='offset points')

    plt.xlim([0, len(ql_tabu)])
    plt.ylim([0, max(ql_mean + rdm_mean) + 5])
    plt.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.75,
            alpha=0.2)

    plt.legend()
    plt.tight_layout()
    plt.savefig("./results_" + ql_path + "/tabu_fails_means.png")


    # Number of molecules discarded by the silly walks filter over steps
    ql_silly_walks_reader = read_csv("./data/" + ql_path + "/steps.csv")
    rdm_silly_walks_reader = read_csv("../random/" + rdm_path + "/steps.csv")

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


    plt.figure("Moyennes du nombre de molécules ne passant pas le filtre des silly walks par steps "
               "(division par " + str(int(np.floor(len(ql_silly_walks) / mean))) + ")", figsize=(16, 9))
    # plt.title("Moyennes du nombre de molécules ne passant pas le filtre des silly walks par steps "
    #           "(division par " + str(int(np.floor(len(ql_silly_walks) / mean))) + ")")

    plt.xlabel("Steps", fontsize=16)
    plt.ylabel("Nombre d'échecs", fontsize=16)

    ql_mean = [ql_silly_walks[i : i + int(np.floor(len(ql_silly_walks) / mean))].mean()
               for i in range(0, len(ql_silly_walks), int(np.floor(len(ql_silly_walks) / mean)))]

    rdm_mean = [rdm_silly_walks[i : i + int(np.floor(len(rdm_silly_walks) / mean))].mean()
                for i in range(0, len(rdm_silly_walks), int(np.floor(len(rdm_silly_walks) / mean)))]

    plt.plot(np.array(range(0, len(ql_silly_walks), int(np.floor(len(ql_silly_walks) / mean)))), ql_mean, color='dodgerblue', label="Q-Learning")
    plt.plot(np.array(range(0, len(rdm_silly_walks), int(np.floor(len(rdm_silly_walks) / mean)))), rdm_mean, color='firebrick', label="Random")

    plt.fill_between(np.array(range(0, len(ql_silly_walks), int(np.floor(len(ql_silly_walks) / mean)))), ql_mean, 0, color='dodgerblue', alpha=.25)
    plt.fill_between(np.array(range(0, len(rdm_silly_walks), int(np.floor(len(rdm_silly_walks) / mean)))), rdm_mean, 0, color='firebrick', alpha=.25)

    ql_total_mean = np.array(ql_silly_walks[1:]).mean()
    rdm_total_mean = np.array(rdm_silly_walks[1:]).mean()

    plt.plot(np.array(range(0, len(ql_silly_walks), int(np.floor(len(ql_silly_walks) / mean)))), [ql_total_mean] * len(ql_mean), color='dodgerblue', linestyle='--', label="Moyenne Q-Learning")
    plt.plot(np.array(range(0, len(rdm_silly_walks), int(np.floor(len(rdm_silly_walks) / mean)))), [rdm_total_mean] * len(rdm_mean), color='firebrick', linestyle='--', label="Moyenne Random")

    plt.annotate(int(np.ceil(ql_total_mean)), xy=(1, ql_total_mean), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points', fontsize=16)
    plt.annotate(int(np.ceil(rdm_total_mean)), xy=(1, rdm_total_mean), xytext=(8, 0),
                    xycoords=('axes fraction', 'data'), textcoords='offset points', fontsize=16)

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
