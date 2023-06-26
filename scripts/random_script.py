from evomol.plot_exploration import exploration_graph
from evomol import run_model, RandomActionTypeSelectionStrategy
from evomol.mutation import KRandomGraphOpsImprovingMutationStrategy

from pathlib import Path


def main():
    max_depth = 1
    max_steps = 500
    atoms = "C,N,O,F"

    model_path = "../examples/random_" + str(max_steps) + "_" + str(max_depth) + "_" + atoms

    run_model({
        "obj_function": "sillywalks_proportion",
        "optimization_parameters": {
            "problem_type": "min",
            "max_steps": max_steps,
            "mutable_init_pop": False,
            "mutation_max_depth": max_depth,
            "neighbour_generation_strategy": RandomActionTypeSelectionStrategy,
            "improving_mutation_strategy": KRandomGraphOpsImprovingMutationStrategy
        },
        "io_parameters": {
            "model_path": model_path,
            "save_n_steps": 1,
            "record_history": True,
            "record_all_generated_individuals": True,
            "smiles_list_init_path": "initial_population.smi",
            "silly_molecules_reference_db_path": "data/ECFP/complete_ChEMBL_ecfp4_dict.json"
        },
        "action_space_parameters": {
            "atoms": atoms,
            "substitution": False,
            "cut_insert": False,
            "move_group": False,
            "remove_group": False,
            "sillywalks_threshold": 0
        },
    })

    exploration_graph(model_path=str(Path.cwd()) + "/" +  model_path, layout="dot", draw_actions=True, plot_images=True, draw_scores=True,
                      root_node="c1ccncc1CCC1CCC1", legend_scores_keys_strat=["total"], mol_size_inches=0.1 ,
                      mol_size_px=(200, 200), figsize=(30, 30 / 1.5), legend_offset=(-0.007, -0.05), legends_font_size=13)


if __name__ == "__main__":
    main()