import numpy as np

from dynamic_programming.grid_world_env import GridWorldEnv
from dynamic_programming.mdp import MDP
from dynamic_programming.stochastic_grid_word_env import StochasticGridWorldEnv

# Exercice 2: Résolution du MDP
# -----------------------------
# Ecrire une fonction qui calcule la valeur de chaque état du MDP, en
# utilisant la programmation dynamique.
# L'algorithme de programmation dynamique est le suivant:
#   - Initialiser la valeur de chaque état à 0
#   - Tant que la valeur de chaque état n'a pas convergé:
#       - Pour chaque état:
#           - Estimer la fonction de valeur de chaque état
#           - Choisir l'action qui maximise la valeur
#           - Mettre à jour la valeur de l'état
#
# Indice: la fonction doit être itérative.


def mdp_value_iteration(mdp: MDP, max_iter: int = 1000, gamma=1.0) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration":
    https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration
    """
    values = np.zeros(mdp.observation_space.n)
    # BEGIN SOLUTION
    epsilon = 1e-5 
    change = float('inf')
    iteration_count = 0  

    while iteration_count < max_iter and change > epsilon:
        previous_values = values.copy()
        change = 0 

        for current_state in range(mdp.observation_space.n):
            state_action_values = []  
            for selected_action in range(mdp.action_space.n):
                next_state, reward, _ = mdp.P[current_state][selected_action]
                state_action_values.append(reward + gamma * previous_values[next_state])

            values[current_state] = max(state_action_values)  
            change = max(change, abs(previous_values[current_state] - values[current_state])) 

        iteration_count += 1  
    # END SOLUTION
    return values


def grid_world_value_iteration(
    env: GridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-5,
) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration".
    theta est le seuil de convergence (différence maximale entre deux itérations).
    """
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    iteration = 0
    max_diff = float('inf')

    while iteration < max_iter and max_diff > theta:
        previous_values = values.copy()
        max_diff = 0

        for x in range(4):
            for y in range(4):
                potential_values = []

                if env.grid[x, y] in {"P", "N", "W"}:
                    potential_values.append(0)
                    continue

                for move in range(env.action_space.n):
                    env.current_position = (x, y)
                    next_position, reward, terminated, _ = env.step(move)

                    if next_position == (x, y):
                        potential_values.append(0)
                        continue

                    potential_values.append(reward + gamma * previous_values[next_position])

                values[x, y] = np.max(potential_values)
                max_diff = max(max_diff, abs(previous_values[x, y] - values[x, y]))

        iteration += 1
    # END SOLUTION
    return values


def value_iteration_per_state(env, values, gamma, prev_val, delta):
    row, col = env.current_position
    values[row, col] = float("-inf")
    for action in range(env.action_space.n):
        next_states = env.get_next_states(action=action)
        current_sum = 0
        for next_state, reward, probability, _, _ in next_states:
            next_row, next_col = next_state
            current_sum += (
                probability
                * env.moving_prob[row, col, action]
                * (reward + gamma * prev_val[next_row, next_col])
            )
        values[row, col] = max(values[row, col], current_sum)
    delta = max(delta, np.abs(values[row, col] - prev_val[row, col]))
    return delta


def stochastic_grid_world_value_iteration(
    env: StochasticGridWorldEnv,
    max_iter: int = 1000,
    gamma: float = 1.0,
    theta: float = 1e-5,
) -> np.ndarray:
    value_matrix = np.zeros((4, 4))
    # BEGIN SOLUTION
    env = GridWorldEnv()
    iteration = 0
    max_delta = float('inf')

    while iteration < max_iter and max_delta > theta:
        previous_values = value_matrix.copy()
        max_delta = 0

        for row in range(4):
            for col in range(4):
                if env.grid[row, col] in {"P", "N", "W"}:
                    continue

                action_values = []

                for action in range(env.action_space.n):
                    action_value = 0

                    for offset in [-1, 1]:
                        env.current_position = (row, col)
                        next_state, reward, _, _ = env.step((action + offset) % 4)
                        if next_state == (1, 1):
                            next_state = (row, col)
                        action_value += 0.05 * (reward + gamma * previous_values[next_state])

                    env.current_position = (row, col)
                    next_state, reward, _, _ = env.step(action)
                    if next_state == (1, 1):
                        next_state = (row, col)
                    action_value += 0.9 * (reward + gamma * previous_values[next_state])
                    action_values.append(action_value)

                value_matrix[row, col] = np.max(action_values)

        max_delta = np.sum(np.abs(previous_values - value_matrix))
        iteration += 1
    # END SOLUTION
    return value_matrix
