import json
from abc import ABC, abstractmethod

import numpy as np
import random

from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles

from evomol.molgraphops.molgraph import MolGraphBuilder
from evomol.molgraphops.actionspace import ActionSpace
from evomol.notification import Observer


def _compute_root_node_id():
    return ""


def _compute_new_edge_name(action_coords):
    """
    Computing the name of the edge created by applying the given action to the given state of the molecular graph
    The edge name is the concatenation of the action type and the action id
    @param action_coords:
    @return:
    """
    return str(action_coords[0]) + "-" + str(action_coords[1])


def _compute_new_node_id(parent_node_id, action_coords):
    """
    Computing the identifier of a node from the action coordinates and the identifier of its parent.
    The node id is the concatenation of the id of its parent and the name of its edge with its parent (action)
    :param parent_node_id:
    :param action_coords
    :return:
    """

    if parent_node_id == _compute_root_node_id():
        separator = ""
    else:
        separator = "_"

    return parent_node_id + separator + _compute_new_edge_name(action_coords)


def random_neighbour(molgraph_builder, depth, return_mol_graph=False, uniform_action_type=True):
    """
    Computing a random neighbour of depth level
    Returning a tuple (SMILES, id) or (mol. graph, id)
    Raising an exception if no neighbour of given depth was found
    @param depth:
    @param molgraph_builder:
    @param return_mol_graph: whether to return a QuMolGraph object or a SMILES
    @param uniform_action_type: If true, the action type is drawn with a uniform law before the action is drawn. If
    false, the action is drawn directly with uniform law among all possible actions
    """

    # Initialization of molecular graph ID
    id = _compute_root_node_id()

    # Copying QuMolGraphBuilder
    molgraph_builder = molgraph_builder.copy()

    for i in range(depth):

        # Valid action list initialization
        valid_action_coords_list = []

        if uniform_action_type:

            # Drawing the action space
            action_space_k = np.random.choice(molgraph_builder.get_action_spaces_keys())

            # Computing the mask of valid actions
            action_space_mask = molgraph_builder.get_valid_mask_from_key(action_space_k)

            # Extracting the set of valid actions
            valid_actions = np.nonzero(action_space_mask)

            # Creating the set of valid action coords
            for valid_act in valid_actions[0]:
                valid_action_coords_list.append((action_space_k, int(valid_act)))

        else:
            # Computing valid actions
            valid_action_dict = molgraph_builder.get_action_spaces_masks()

            # Iterating over the actions of the different action spaces
            for key, validity in valid_action_dict.items():

                # Recording the id of the valid actions for the current action space
                curr_key_valid_actions = np.nonzero(validity)

                # Iterating over the valid actions for the current action space
                for curr_key_valid_act in curr_key_valid_actions[0]:
                    # Adding the current valid action to the list
                    valid_action_coords_list.append((key, int(curr_key_valid_act)))

        if valid_action_coords_list:

            # Drawing action to apply
            action_coords = valid_action_coords_list[np.random.choice(np.arange(len(valid_action_coords_list)))]

            # Updating molecule ID
            id = _compute_new_node_id(id, action_coords)

            # Applying action
            molgraph_builder.execute_action_coords(action_coords)

    if return_mol_graph:
        return molgraph_builder.qu_mol_graph, id
    else:
        return molgraph_builder.qu_mol_graph.to_aromatic_smiles(), id


class NeighbourGenerationStrategy(ABC):
    """
    Strategy that defines how neighbour solutions are generated.
    Either a neighbour is selected randomly with uniform low from the set of all possible valid neighbours
    (preselect_action_type=False), either the type of perturbation/mutation is selected first and then the action is
    selected randomly with uniform law among valid neighbours from selected perturbation type
    (preselect_action_type=True). In the latter case, the implementations of this class define how the action type is
    selected.
    """

    def __init__(self, preselect_action_type=True):
        """
        :param preselect_action_type: whether to first select the action type and then select the actual valid
        perturbation of selected type (True), or whether to select the actual perturbation among all possible ones of
        all types (False).
        """
        self.preselect_action_type = preselect_action_type

    @abstractmethod
    def select_action_type(self, action_types_list, evaluation_strategy):
        """
        Selection of the action type.
        :param action_types_list: list of available action types
        :param evaluation_strategy: instance of evomol.evaluation.EvaluationStrategyComposite
        :return: a single selected action type
        """
        pass

    def generate_neighbour(self, molgraph_builder, depth, evaluation_strategy, return_mol_graph=False):
        """
        :param molgraph_builder: evomol.molgraphops.molgraph.MolGraphBuilder instance previously set up to apply
        perturbations on the desired molecular graph.
        :param depth in number of perturbations of the output neighbour.
        :param evaluation_strategy: evomol.evaluation.EvaluationStrategyComposite instance that is used to evaluate the
        solutions in the EvoMol optimization procedure
        :param return_mol_graph: whether to return the molecular graph (evomol.molgraphops.molgraph.MolGraph) or a
        SMILES.
        :return: (evomol.molgraphops.molgraph.MolGraph, string id of the perturbation) or
        (SMILES, string id of the perturbation)
        """

        # Initialization of molecular graph ID
        id = _compute_root_node_id()

        # Copying QuMolGraphBuilder
        molgraph_builder = molgraph_builder.copy()

        for i in range(depth):

            # Valid action list initialization
            valid_action_coords_list = []

            # The perturbation type is selected before selecting the actual perturbation
            if self.preselect_action_type:

                # Selecting the action type
                action_space_k = self.select_action_type(molgraph_builder.get_action_spaces_keys(), evaluation_strategy)

                # Computing the mask of valid actions
                action_space_mask = molgraph_builder.get_valid_mask_from_key(action_space_k)

                # Extracting the set of valid actions
                valid_actions = np.nonzero(action_space_mask)

                # Creating the set of valid action coords
                for valid_act in valid_actions[0]:
                    valid_action_coords_list.append((action_space_k, int(valid_act)))

            # The perturbation is selected randomly from the set of all possible perturbations
            else:

                # Computing valid actions
                valid_action_dict = molgraph_builder.get_action_spaces_masks()

                # Iterating over the actions of the different action spaces
                for key, validity in valid_action_dict.items():

                    # Recording the id of the valid actions for the current action space
                    curr_key_valid_actions = np.nonzero(validity)

                    # Iterating over the valid actions for the current action space
                    for curr_key_valid_act in curr_key_valid_actions[0]:
                        # Adding the current valid action to the list
                        valid_action_coords_list.append((key, int(curr_key_valid_act)))

            if valid_action_coords_list:

                # Drawing action to apply
                action_coords = valid_action_coords_list[np.random.choice(np.arange(len(valid_action_coords_list)))]

                # Updating molecule ID
                id = _compute_new_node_id(id, action_coords)

                # Applying action
                molgraph_builder.execute_action_coords(action_coords)

        if return_mol_graph:
            return molgraph_builder.qu_mol_graph, id
        else:
            return molgraph_builder.qu_mol_graph.to_aromatic_smiles(), id


class RandomActionTypeSelectionStrategy(NeighbourGenerationStrategy):
    """
    Selection of the action type randomly with uniform law.
    """

    def select_action_type(self, action_types_list, evaluation_strategy):
        return np.random.choice(action_types_list)


class AlwaysFirstActionSelectionStrategy(NeighbourGenerationStrategy):
    """
    Always selecting the first action type
    """

    def select_action_type(self, action_types_list, evaluation_strategy):
        return action_types_list[0]


class QLearningActionSelectionStrategy(NeighbourGenerationStrategy, Observer, ABC):
    """
    Selection of the action type according to a Q-learning strategy.
    """

    def __init__(self, depth, number_of_accepted_atoms, valid_ecfp_file_path=None,
                 init_weights_file_path=None, preselect_action_type=True):
        """
        :param depth: number of consecutive executed actions before evaluation
        :param number_of_accepted_atoms: number of accepted atoms in the molecule
        :param valid_ecfp_file_path: path to the file containing the valid ECFPs
        :param init_weights_file_path: initial weights for the Q-learning strategy
        :param preselect_action_type: whether to preselect the action type
        before selecting the actual action
        """

        super().__init__(preselect_action_type)

        # Initializing the depth of the search
        self.depth = depth
        self.depth_counter = 0

        # Initializing the array of valid ECFP-0
        self.valid_ecfps = self.extract_valid_ecfps_from_file(valid_ecfp_file_path)

        # Number of valid contexts base on the number of the valid molecules ECFP-0
        self.number_of_contexts = self.valid_ecfps.shape[0]

        # Initializing the weights for each action type
        init_weights = self.initialize_weights(init_weights_file_path, number_of_accepted_atoms)

        self.w_addA = init_weights[0]
        self.w_rmA = init_weights[1]
        self.w_chB = init_weights[2]

        # Feature vector of the current state needed for the update of the weights
        self.current_features = []

    def extract_valid_ecfps_from_file(self, file_path):
        """
        Extracting the valid ECFP-0 from the specified json file
        :param file_path: path to the file containing the valid ECFPs
        :return: list of valid ECFP-0
        """

        if file_path is None:
            raise ValueError('The path to the file containing the valid ECFPs is not specified.')

        with open(file_path, 'r') as f:
            return np.array(json.load(f))

    def get_valid_ecfp_vectors(self, ecfps):
        """
        Iterating over the valid ECFP-0 of the database to check if they are in
        the current molecule and recording the result in a binary vector
        :param ecfps: list of ECFP-0 of the current molecule
        :return: binary vector of valid ECFP-0
        """

        # Can be optimized by using a dictionary for self.valid_ecfps
        return [valid_ecfp in ecfps for valid_ecfp in self.valid_ecfps]

    def binary_vector_to_index(self, vector, context_id=0):
        """
        Only adapted for ECFP-0 (only one valid ECFP-0 per context)
        :param vector: binary vector of valid ECFP-0
        :param context_id: ID of the context
        :return: index of the valid ECFP-0 in the binary vector
        """

        # Index where there is no valid ECFP-0 for the current context
        index = self.number_of_contexts * context_id

        # Iterating over the binary vector to find the valid ECFP-0 for the current context
        # and computing the index for the feature vector
        for i in range(len(vector)):
            if vector[i]:
                # Found a valid ECFP-0 for the current context
                index += i + 1

        return index

    def extract_features(self, molgraph_builder, ecfps, valid_action, action_space):
        """
        :param molgraph_builder: molecule graph builder
        :param ecfps: list of ECFP-0 of the molecule
        :param valid_action: valid action to apply
        :param action_space: action space
        :return: feature vector of the current state for a given action
        """

        qry_ecfp = set()
        context = action_space.get_context(valid_action[1], molgraph_builder.parameters, qu_mol_graph=molgraph_builder.qu_mol_graph)

        # Getting the ECFP-0 where the action will be applied and translating it into
        # a float vector of valid ECFP-0
        if valid_action[0] == 'AddA':
            # Getting the ECFPs of the atom where the action will be applied
            for i in range(len(ecfps[context[0]])):
                qry_ecfp.add(ecfps[context[0]][i])

            # Getting the valid ECFPs for the current context
            valid_ecfps = self.get_valid_ecfp_vectors(qry_ecfp)

            # Computing the feature vector
            features = np.zeros((self.number_of_contexts + 1) * len(molgraph_builder.parameters.accepted_atoms))
            features[self.binary_vector_to_index(valid_ecfps, context[1])] = 1.
        elif valid_action[0] == 'RmA':
            for i in range(len(ecfps[context[0]])):
                qry_ecfp.add(ecfps[context[0]][i])

            valid_ecfps = self.get_valid_ecfp_vectors(qry_ecfp)

            features = np.zeros(self.number_of_contexts + 1)
            features[self.binary_vector_to_index(valid_ecfps)] = 1.
        elif valid_action[0] == 'ChB':
            # Atom on the left of the bond
            for i in range(len(ecfps[context[0]])):
                qry_ecfp.add(ecfps[context[0]][i])
            valid_ecfps = self.get_valid_ecfp_vectors(qry_ecfp)

            features = np.zeros((self.number_of_contexts + 1) * 4)
            features[self.binary_vector_to_index(valid_ecfps, context[3])] = 1.

            # Atom on the right of the bond
            qry_ecfp_ = set()

            for i in range(len(ecfps[context[1]])):
                qry_ecfp_.add(ecfps[context[1]][i])

            valid_ecfps = self.get_valid_ecfp_vectors(qry_ecfp_)

            features[self.binary_vector_to_index(valid_ecfps, context[3])] = 1.
        else:
            raise ValueError('Invalid action')

        return features

    def generate_neighbour(self, molgraph_builder, depth, evaluation_strategy, return_mol_graph=False):
        """
        :param molgraph_builder: evomol.molgraphops.molgraph.MolGraphBuilder instance previously set up to apply
        perturbations on the desired molecular graph.
        :param depth in number of perturbations of the output neighbour.
        :param evaluation_strategy: evomol.evaluation.EvaluationStrategyComposite instance that is used to evaluate the
        solutions in the EvoMol optimization procedure
        :param return_mol_graph: whether to return the molecular graph (evomol.molgraphops.molgraph.MolGraph) or a
        SMILES.
        :return: list of (evomol.molgraphops.molgraph.MolGraph, string id of the perturbation, list of chosen actions) or
        (list of SMILES, string id of the perturbation, list of chosen actions)
        """

        # Initialization of molecular graph ID
        id = _compute_root_node_id()

        # Initializing the list of molgraph_builders and chosen_actions to be updated
        molgraph_builders = []
        chosen_actions = []

        # Initializing the depth counter and the current_features list
        self.depth_counter = 0
        self.current_features = []

        # Iterating over the number of actions to be executed
        for i in range(depth):
            # Copying QuMolGraphBuilder
            molgraph_builder = molgraph_builder.copy()

            # Getting the ECFP of the current molecule
            ecfps_trace = {}
            AllChem.GetMorganFingerprint(MolFromSmiles(molgraph_builder.qu_mol_graph.to_smiles()), 0, bitInfo=ecfps_trace)

            # Putting atom ids as keys and the corresponding ECFP and radius as values
            ecfps = dict()

            for k in ecfps_trace.keys():
                for (atom_id, rad) in ecfps_trace[k]:
                    if not atom_id in ecfps.keys():
                        ecfps[atom_id] = list()

                    ecfps[atom_id].append(k)

            # Selecting the action to be executed
            chosen_action = self.select_action_type([], evaluation_strategy, ecfps, molgraph_builder)

            # Updating molecule ID
            id = _compute_new_node_id(id, chosen_action)

            # Applying action
            molgraph_builder.execute_action_coords(chosen_action)

            # Updating the list of molgraph_builders and chosen_actions
            molgraph_builders.append(molgraph_builder.copy())
            chosen_actions.append(chosen_action)

        if return_mol_graph:
            return molgraph_builder.qu_mol_graph, id, molgraph_builders, chosen_actions
        else:
            return molgraph_builder.qu_mol_graph.to_aromatic_smiles(), id, molgraph_builders, chosen_actions

    @abstractmethod
    def initialize_weights(self, file_path, number_of_accepted_atoms):
        """
        Initializing the weights for each action type
        :param file_path: path to the file containing the initial weights
        :param number_of_accepted_atoms: number of accepted atoms in the molecule
        return: list of initial weights
        """

        pass

    @abstractmethod
    def select_action_type(self, action_types_list, evaluation_strategy, ecfps=None, molgraph_builder=MolGraphBuilder([], [])):
        """
        Selecting the action type according to the Q-learning strategy
        :param action_types_list: list of action types authorized
        :param evaluation_strategy: evaluation strategy
        :param ecfps: list of ECFPs of the current molecule
        :param molgraph_builder: molecule graph builder
        :return: valid action to execute
        """

        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Updating the weights of the Q-learning strategy
        :param args: arguments of the update method
        :param kwargs: keyword arguments of the update method
        """

        pass


class DeterministQLearningActionSelectionStrategy(QLearningActionSelectionStrategy):
    """
    Selection of the action type according to a determinist Q-learning strategy.
    """

    def __init__(self, depth, number_of_accepted_atoms, alpha, epsilon, gamma,
                 valid_ecfp_file_path=None, init_weights_file_path=None, preselect_action_type=True):
        """
        :param depth: number of consecutive executed actions before evaluation
        :param number_of_accepted_atoms: number of accepted atoms in the molecule
        :param alpha: learning rate
        :param epsilon: exploration rate
        :param gamma: discount factor
        :param valid_ecfp_file_path: path to the file containing the valid ECFPs
        :param init_weights_file_path: initial weights for the Q-learning strategy
        :param preselect_action_type: whether to preselect the action type
        before selecting the actual action
        """

        super().__init__(depth, number_of_accepted_atoms, valid_ecfp_file_path,init_weights_file_path,
                         preselect_action_type)

        # Initializing the hyperparameters of the Q-learning strategy
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

    def initialize_weights(self, file_path, number_of_accepted_atoms):
        """
        Initializing the weights for each action type
        :param file_path: path to the file containing the initial weights
        :param number_of_accepted_atoms: number of accepted atoms in the molecule
        """

        try:
            if file_path is None:
                # Returning the weights for each action type
                return [np.array([np.random.normal() for _ in range((self.number_of_contexts + 1) * number_of_accepted_atoms)]),
                        np.array([np.random.normal() for _ in range(self.number_of_contexts + 1)]),
                        np.array([np.random.normal() for _ in range((self.number_of_contexts + 1) * 4)])]
            else:
                # Reading the last line of the file to get the initial weights
                with open(file_path, "r") as f:
                    init_weights = json.load(f)

                # Returning the weights for each action type
                return [np.array(init_weights['w_addA']),
                        np.array(init_weights['w_rmA']),
                        np.array(init_weights['w_chB'])]

        except FileNotFoundError:
            print('The file containing the initial weights does not exist.')

    def select_action_type(self, action_types_list, evaluation_strategy, ecfps=None, molgraph_builder=MolGraphBuilder([], [])):
        """
        :param action_types_list: list of action types authorized
        :param evaluation_strategy: evaluation strategy
        :param ecfps: list of ECFP-0 of the current molecule
        :param molgraph_builder: graph builder of the current molecule
        :return: action selected
        """

        if ecfps is None:
            ecfps = []

        # Getting the action spaces of the current molecule
        actions = molgraph_builder.get_action_spaces_keys()

        # Initializing the feature vectors for each action type
        features_addA = np.array([])
        features_rmA = np.array([])
        features_chB = np.array([])

        # Initializing the valid actions for each action type
        valid_action_index_list = []

        for action in actions:
            # Initializing the list of valid actions for the current action type
            valid_action_coords_list = []

            # Getting the valid actions for the current action type
            action_mask = molgraph_builder.get_valid_mask_from_key(action)
            action_space = molgraph_builder.action_spaces_d[action]
            valid_actions = np.nonzero(action_mask)

            # Getting the valid actions coordinates
            for valid_act in valid_actions[0]:
                valid_action_coords_list.append((action, int(valid_act)))

            # Adding the valid actions coordinates to the list of valid actions
            valid_action_index_list.append(valid_action_coords_list)

            # Computing the feature vectors for each action type
            if valid_action_coords_list:
                for valid_action in valid_action_coords_list:
                    valid_action_features = self.extract_features(molgraph_builder, ecfps, valid_action, action_space)

                    if valid_action[0] == 'AddA':
                        features_addA = np.append(features_addA, valid_action_features)
                    elif valid_action[0] == 'RmA':
                        features_rmA = np.append(features_rmA, valid_action_features)
                    elif valid_action[0] == 'ChB':
                        features_chB = np.append(features_chB, valid_action_features)
                    else:
                        raise ValueError('Invalid action')

        # Reshaping the feature vectors
        features_addA = features_addA.reshape(-1, (self.number_of_contexts + 1) * len(molgraph_builder.parameters.accepted_atoms))
        features_rmA = features_rmA.reshape(-1, self.number_of_contexts + 1)
        features_chB = features_chB.reshape(-1, (self.number_of_contexts + 1) * 4)

        # Computing the Q-values for each action type
        q_states_addA = (features_addA @ self.w_addA).flatten()
        q_states_rmA = (features_rmA @ self.w_rmA).flatten()
        q_states_chB = (features_chB @ self.w_chB).flatten()

        # Choosing a random feature vector in the list of valid actions
        if random.random() < self.epsilon:
            # Initializing the list of non-empty action lists
            non_empty_action_lists = []

            # Appending the non-empty action lists
            if q_states_addA.any():
                non_empty_action_lists.append(0)
            if q_states_rmA.any():
                non_empty_action_lists.append(1)
            if q_states_chB.any():
                non_empty_action_lists.append(2)

            # Choosing a random action type
            action_index = random.choice(non_empty_action_lists)

            # Choosing a random action in the chosen action type and
            # saving the corresponding feature vector
            if action_index == 0:
                coord_index = random.randrange(0, features_addA.shape[0])
                self.current_features.append(features_addA[coord_index])
            elif action_index == 1:
                coord_index = random.randrange(0, features_rmA.shape[0])
                self.current_features.append(features_rmA[coord_index])
            elif action_index == 2:
                coord_index = random.randrange(0, features_chB.shape[0])
                self.current_features.append(features_chB[coord_index])
            else:
                raise ValueError('Invalid action')

            # Incrementing the depth counter
            self.depth_counter += 1

            return valid_action_index_list[action_index][coord_index]

        # Choosing the best feature vector in the list of valid actions
        else:
            # Initializing the list of non-empty action lists
            non_empty_action_list_indexes = dict()

            # Appending the non-empty action lists
            if q_states_addA.any():
                non_empty_action_list_indexes['0'] = np.argmax(q_states_addA)
            if q_states_rmA.any():
                non_empty_action_list_indexes['1'] = np.argmax(q_states_rmA)
            if q_states_chB.any():
                non_empty_action_list_indexes['2'] = np.argmax(q_states_chB)

            if not non_empty_action_list_indexes:
                raise ValueError('No valid actions')

            # Choosing the best action type key
            q_max_keys = max(non_empty_action_list_indexes, key=lambda x: non_empty_action_list_indexes[x])

            # Choosing the best action in the chosen action type and
            # saving the corresponding feature vector
            if q_max_keys == '0':
                self.current_features.append(features_addA[non_empty_action_list_indexes[q_max_keys]])
                valid_action_index = non_empty_action_list_indexes[q_max_keys]
            elif q_max_keys == '1':
                self.current_features.append(features_rmA[non_empty_action_list_indexes[q_max_keys]])
                valid_action_index = non_empty_action_list_indexes[q_max_keys]
            elif q_max_keys == '2':
                self.current_features.append(features_chB[non_empty_action_list_indexes[q_max_keys]])
                valid_action_index = non_empty_action_list_indexes[q_max_keys]
            else:
                raise ValueError('Invalid action')

            # Incrementing the depth counter
            self.depth_counter += 1

            return valid_action_index_list[int(q_max_keys)][valid_action_index]

    def update(self, *args, **kwargs):
        """
        Updates the weights according to the action(s) chosen previously
        :molgraph_builder: molecule graph builder object for the current context
        :executed_action: action executed on the molecule
        :reward: reward obtained from the evaluation of the executed action on the current molecule
        :inverted_reward: boolean indicating whether the reward should be inverted or not
        :boolean_reward: boolean indicating whether the reward should be boolean or not
        """

        # Checking the needed arguments
        try:
            molgraph_builder = args[1]
            executed_action = args[2]
            reward = args[3]
        except IndexError:
            raise ValueError('Invalid arguments')

        # Checking the optional arguments
        if 'inverted_reward' in kwargs:
            if 'boolean_reward' in kwargs:
                if reward == 0.:
                    reward = 1.
                else:
                    reward = 0.
            else:
                reward = -reward

        # Initializing the feature vector and the matrix of valid actions
        features = np.array([])
        valid_action_coords_list = []

        # Getting the valid actions for the current context
        action_mask = molgraph_builder.get_valid_mask_from_key(executed_action[0])
        action_space = molgraph_builder.action_spaces_d[executed_action[0]]
        valid_actions = np.nonzero(action_mask)

        # Getting the ECFP of the current molecule
        ecfps_trace = {}
        AllChem.GetMorganFingerprint(MolFromSmiles(molgraph_builder.qu_mol_graph.to_smiles()), 0, bitInfo=ecfps_trace)

        # Putting atom ids as keys and the corresponding ECFP and radius as values
        ecfps = dict()

        for k in ecfps_trace.keys():
            for (atom_id, rad) in ecfps_trace[k]:
                if not atom_id in ecfps.keys():
                    ecfps[atom_id] = list()

                ecfps[atom_id].append(k)

        # Getting the features of the current context
        current_features = self.current_features[self.depth - self.depth_counter]

        # Getting the valid actions coordinates
        for valid_act in valid_actions[0]:
            valid_action_coords_list.append((executed_action[0], int(valid_act)))

        # Extracting the features of the valid actions
        if valid_action_coords_list:
            for executed_action in valid_action_coords_list:
                valid_action_features = self.extract_features(molgraph_builder, ecfps, executed_action, action_space)
                features = np.append(features, [valid_action_features])

        # Updating the weights according to the executed action
        if executed_action[0] == 'AddA':
            features = features.reshape(-1, (self.number_of_contexts + 1) * len(molgraph_builder.parameters.accepted_atoms))
            q_states = features @ self.w_addA

            target = reward + self.gamma * np.amax(q_states)
            q_state = current_features @ self.w_addA.reshape(-1, 1)

            self.w_addA = np.array([self.w_addA[i] - 2 * self.alpha * current_features[i] * (q_state - target) for i in range(len(self.w_addA))])

            # Decrementing the depth counter
            self.depth_counter -= 1
        elif executed_action[0] == 'RmA':
            features = features.reshape(-1, self.number_of_contexts + 1)
            q_states = features @ self.w_rmA

            target = reward + self.gamma * np.amax(q_states)
            q_state = current_features @ self.w_rmA.reshape(-1, 1)

            self.w_rmA = np.array([self.w_rmA[i] - 2 * self.alpha * current_features[i] * (q_state - target) for i in range(len(self.w_rmA))])
            self.depth_counter -= 1
        elif executed_action[0] == 'ChB':
            features = features.reshape(-1, (self.number_of_contexts + 1) * 4)
            q_states = features @ self.w_chB

            target = reward + self.gamma * np.amax(q_states)
            q_state = current_features @ self.w_chB.reshape(-1, 1)

            self.w_chB = np.array([self.w_chB[i] - 2 * self.alpha * current_features[i] * (q_state - target) for i in range(len(self.w_chB))])
            self.depth_counter -= 1
        else:
            raise ValueError('Invalid action')


class SuccessRate:
    """
    Class for storing the context success rate
    """

    def __init__(self, success_usage_tuple):
        self.success = success_usage_tuple[0]
        self.usage = success_usage_tuple[1]

    def get_success_rate(self, epsilon=None):
        return self.success / self.usage if self.usage > 0 else epsilon if epsilon else 0.


class StochasticQLearningActionSelectionStrategy(QLearningActionSelectionStrategy):
    """
    Stochastic Q-Learning action selection strategy based on the success rate of the contexts
    """

    def __init__(self, depth, number_of_accepted_atoms, epsilon,
                 valid_ecfp_file_path=None, init_weights_file_path=None, preselect_action_type=True):
        """
        :param depth: number of consecutive executed actions before evaluation
        :param number_of_accepted_atoms: number of accepted atoms in the molecule
        :param epsilon: initial context probability
        :param valid_ecfp_file_path: path to the file containing the valid ECFPs
        :param init_weights_file_path: initial weights for the Q-learning strategy
        :param preselect_action_type: whether to preselect the action type
        before selecting the actual action
        """

        super().__init__(depth, number_of_accepted_atoms, valid_ecfp_file_path=valid_ecfp_file_path,
                 init_weights_file_path=init_weights_file_path, preselect_action_type=preselect_action_type)

        self.epsilon = epsilon

    def initialize_weights(self, file_path, number_of_accepted_atoms):
        """
        Initializing the weights for each action type
        :param file_path: path to the file containing the initial weights
        :param number_of_accepted_atoms: number of accepted atoms in the molecule
        return: list of initial weights
        """

        try:
            if file_path is None:
                # Returning the weights for each action type
                return [
                    np.array([SuccessRate([0, 0]) for _ in range((self.number_of_contexts + 1) * number_of_accepted_atoms)]),
                    np.array([SuccessRate([0, 0]) for _ in range(self.number_of_contexts + 1)]),
                    np.array([SuccessRate([0, 0]) for _ in range((self.number_of_contexts + 1) * 4)])]
            else:
                # Reading the last line of the file to get the initial weights
                with open(file_path, "r") as f:
                    init_weights = json.load(f)

                # Returning the weights for each action type
                return [np.array([SuccessRate(success_usage_tuple) for success_usage_tuple in init_weights['w_addA']]),
                        np.array([SuccessRate(success_usage_tuple) for success_usage_tuple in init_weights['w_rmA']]),
                        np.array([SuccessRate(success_usage_tuple) for success_usage_tuple in init_weights['w_chB']])]

        except FileNotFoundError:
            print('The file containing the initial weights does not exist.')



    def select_action_type(self, action_types_list, evaluation_strategy, ecfps=None, molgraph_builder=MolGraphBuilder([], [])):
        """
        :param action_types_list: list of action types authorized
        :param evaluation_strategy: evaluation strategy
        :param ecfps: list of ECFP-0 of the current molecule
        :param molgraph_builder: graph builder of the current molecule
        :return: action selected
        """

        if ecfps is None:
            ecfps = []

        # Getting the action spaces of the current molecule
        actions = molgraph_builder.get_action_spaces_keys()

        # Initializing the feature vectors for each action type
        features_addA = np.array([])
        features_rmA = np.array([])
        features_chB = np.array([])

        # Initializing the probabilities of each action type
        probabilities_addA = np.array([])
        probabilities_rmA = np.array([])
        probabilities_chB = np.array([])

        # Computing the success rate of each context
        w_addA_success_rate = np.array([self.w_addA[i].get_success_rate(self.epsilon) for i in range(len(self.w_addA))])
        w_rmA_success_rate = np.array([self.w_rmA[i].get_success_rate(self.epsilon) for i in range(len(self.w_rmA))])
        w_chB_success_rate = np.array([self.w_chB[i].get_success_rate(self.epsilon) for i in range(len(self.w_chB))])

        # Initializing the valid actions for each action type
        valid_action_index_list = []

        for action in actions:
            # Initializing the list of valid actions for the current action type
            valid_action_coords_list = []

            # Getting the valid actions for the current action type
            action_mask = molgraph_builder.get_valid_mask_from_key(action)
            action_space = molgraph_builder.action_spaces_d[action]
            valid_actions = np.nonzero(action_mask)

            # Getting the valid actions coordinates
            for valid_act in valid_actions[0]:
                valid_action_coords_list.append((action, int(valid_act)))

            # Adding the valid actions coordinates to the list of valid actions
            valid_action_index_list.append(valid_action_coords_list)

            # Computing the feature vectors for each action type
            if valid_action_coords_list:
                for valid_action in valid_action_coords_list:
                    valid_action_features = self.extract_features(molgraph_builder, ecfps, valid_action, action_space)

                    if valid_action[0] == 'AddA':
                        features_addA = np.append(features_addA, valid_action_features)

                        action_contexts = np.argwhere(valid_action_features == 1.)[0]
                        probabilities_addA = np.append(probabilities_addA, [np.sum(w_addA_success_rate[action_context]) for action_context in action_contexts][0])
                    elif valid_action[0] == 'RmA':
                        features_rmA = np.append(features_rmA, valid_action_features)

                        action_contexts = np.argwhere(valid_action_features == 1.)[0]
                        probabilities_rmA = np.append(probabilities_rmA, [np.sum(w_rmA_success_rate[action_context]) for action_context in action_contexts][0])
                    elif valid_action[0] == 'ChB':
                        features_chB = np.append(features_chB, valid_action_features)

                        action_contexts = np.argwhere(valid_action_features == 1.)[0]
                        probabilities_chB = np.append(probabilities_chB, [np.sum(w_chB_success_rate[action_context]) for action_context in action_contexts][0])
                    else:
                        raise ValueError('Invalid action')

        # Computing the divider for the probability distribution by summing the valid actions success rates
        probability_divider = np.sum(np.concatenate((probabilities_addA, probabilities_rmA, probabilities_chB)))

        # Computing the probability distribution for each action type and choose one accordingly
        if probability_divider == 0.0:
            chosen_action_index = np.random.choice(len(probabilities_addA) + len(probabilities_rmA) + len(probabilities_chB))
        else:
            chosen_action_index = np.random.choice(len(probabilities_addA) + len(probabilities_rmA) + len(probabilities_chB),
                                                    p=[probabilities_addA[i] / probability_divider for i in range(len(probabilities_addA))] +
                                                    [probabilities_rmA[i] / probability_divider for i in range(len(probabilities_rmA))] +
                                                    [probabilities_chB[i] / probability_divider for i in range(len(probabilities_chB))])

        # Getting the chosen action
        if chosen_action_index < len(probabilities_addA):
            chosen_action = valid_action_index_list[0][chosen_action_index]
            features_addA = features_addA.reshape(-1, (self.number_of_contexts + 1) * len(molgraph_builder.parameters.accepted_atoms))
            self.current_features.append(features_addA[chosen_action_index])
        elif chosen_action_index < len(probabilities_addA) + len(probabilities_rmA):
            chosen_action = valid_action_index_list[1][chosen_action_index - len(probabilities_addA)]
            features_rmA = features_rmA.reshape(-1, self.number_of_contexts + 1)
            self.current_features.append(features_rmA[chosen_action_index - len(probabilities_addA)])
        else:
            chosen_action = valid_action_index_list[2][chosen_action_index - len(probabilities_addA) - len(probabilities_rmA)]
            features_chB = features_chB.reshape(-1, (self.number_of_contexts + 1) * 4)
            self.current_features.append(features_chB[chosen_action_index - len(probabilities_addA) - len(probabilities_rmA)])

        # Incrementing the depth counter
        self.depth_counter += 1

        return chosen_action

    def update(self, *args, **kwargs):
        """
        Updates the weights according to the action(s) chosen previously
        :molgraph_builder: molecule graph builder object for the current context
        :executed_action: action executed on the molecule
        :reward: reward obtained from the evaluation of the executed action on the current molecule
        :inverted_reward: boolean indicating whether the reward should be inverted or not
        :boolean_reward: boolean indicating whether the reward should be boolean or not
        """

        # Checking the needed arguments
        try:
            action_type = args[2][0]
            successful = args[3]
        except IndexError:
            raise ValueError('Invalid arguments')

        if 'inverted_reward' in kwargs:
           successful = not successful

        contexts_indexes = np.argwhere(self.current_features[self.depth - self.depth_counter] == 1.)[0]

        if successful:
            # Updating the amount of success for the chosen context
            if action_type == 'AddA':
                for context_index in contexts_indexes:
                    self.w_addA[context_index].success += 1
            elif action_type == 'RmA':
                for context_index in contexts_indexes:
                    self.w_rmA[context_index].success += 1
            else:
                for context_index in contexts_indexes:
                    self.w_chB[context_index].success += 1

        # Updating the amount of usage for the chosen context
        if action_type == 'AddA':
            for context_index in contexts_indexes:
                self.w_addA[context_index].usage += 1
        elif action_type == 'RmA':
            for context_index in contexts_indexes:
                self.w_rmA[context_index].usage += 1
        else:
            for context_index in contexts_indexes:
                self.w_chB[context_index].usage += 1

        # Decrementing the depth counter
        self.depth_counter -= 1

        oui = 0
