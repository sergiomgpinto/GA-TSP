from numba import njit
import Reporter
import numpy as np


@njit()
def initialization(n: int,
                   distance_matrix: np.ndarray,
                   population_size: int,
                   population: np.ndarray,
                   k: int) -> np.ndarray:
    for i in range(population_size):
        population[i] = initialize_individual(n, distance_matrix, k)
    return population


@njit()
def initialize_individual(n: int,
                          distance_matrix: np.ndarray,
                          k: int) -> np.ndarray:
    individual = np.full(n, -1, dtype=np.int32)
    individual[0] = np.random.randint(0, n)
    cities = np.random.permutation(n)
    current_idx = 0

    while current_idx < n - 1:
        individual_set = set(individual)
        cities_left = np.array([city for city in cities if city not in individual_set], dtype=np.int32)

        winner_city = np.random.choice(cities_left)
        winner_city_distance = distance_matrix[individual[current_idx], winner_city]
        for j in range(k):
            contender_city = np.random.choice(cities_left)
            contender_city_distance = distance_matrix[individual[current_idx], contender_city]
            if contender_city_distance < winner_city_distance and contender_city not in individual_set:
                winner_city = contender_city
                winner_city_distance = contender_city_distance
        individual[current_idx + 1] = winner_city
        current_idx += 1
    return individual


@njit()
def fitness(n: int,
            individual: np.ndarray,
            distance_matrix: np.ndarray) -> np.float32:
    cost = np.float32(0)

    for i in range(n):
        cost += distance_matrix[individual[i], individual[(i + 1) % n]]
    return cost


@njit()
def evaluate(population_size: int,
             population: np.ndarray,
             individuals_fitness: np.ndarray,
             distance_matrix: np.ndarray,
             n: int) -> np.ndarray:
    for i in range(population_size):
        individuals_fitness[i] = fitness(n, population[i], distance_matrix)
    return individuals_fitness


@njit()
def selection(nb_parents: int,
              population: np.ndarray,
              individuals_fitness: np.ndarray,
              k: int,
              n: int,
              population_size: int) -> np.ndarray:
    i = 0
    new_parents = np.zeros((nb_parents, n), dtype=np.int32)

    while i < nb_parents:
        winner_index = np.random.randint(0, population_size)
        winner = population[winner_index]
        best_score = individuals_fitness[winner_index]

        for j in range(k):
            contender_index = np.random.randint(0, population_size)
            contender = population[contender_index]
            score = individuals_fitness[contender_index]
            if score < best_score:
                best_score = score
                winner = contender
        new_parents[i] = winner
        i += 1

    return new_parents


@njit(fastmath=True)
def mutation_operators(children: np.ndarray,
                       alpha: float,
                       n: int,
                       nb_children: int,
                       diver: np.float64) -> np.ndarray:
    i = 0
    randoms = np.random.rand(nb_children)

    while i < nb_children:
        if randoms[i] < alpha:
            swap_indexes = np.random.randint(0, n, 2)
            if swap_indexes[0] == swap_indexes[1]:
                continue
            start = np.min(swap_indexes)
            end = np.max(swap_indexes)
            if diver > 0.01:
                children[i][start:end] = children[i][start:end][::-1]
            elif 0.005 < diver < 0.01:
                children[i][start], children[i][end] = children[i][end], children[i][start]
            else:
                children[i][swap_indexes[0]], children[i][(swap_indexes[0] + 1) % n] = \
                    children[i][(swap_indexes[0] + 1) % n], children[i][swap_indexes[0]]
        i += 1
    return children


@njit()
def general_local_search_operator(population: np.ndarray,
                                  distance_matrix: np.ndarray,
                                  n: int,
                                  int_refinement: float,
                                  population_size) -> np.ndarray:
    for i in range(population_size):
        population[i] = local_search_operator(population[i], distance_matrix, n, int_refinement)
    return population


@njit()
def local_search_operator(individual: np.ndarray,
                          distance_matrix: np.ndarray,
                          n: int,
                          int_refinement: float) -> np.ndarray:
    best_swap_indexes = np.zeros(2, dtype=np.int32)
    best_cost_reduction = np.float32(0)
    window_size = int(n * int_refinement)
    indexes = np.random.randint(0, n, 2)
    ii = np.min(indexes)
    jj = np.max(indexes)

    while window_size < (jj - ii) or (jj - ii) == 0:
        indexes = np.random.randint(0, n, 2)
        ii = np.min(indexes)
        jj = np.max(indexes)

    for i in range(ii, jj):
        for j in range(i + 1, jj):

            if j - i == 1:
                pre_i_to_i = distance_matrix[individual[i - 1], individual[i]]
                i_to_j = distance_matrix[individual[i], individual[j]]
                j_to_post_j = distance_matrix[individual[j], individual[(j + 1) % n]]

                initial_cost = pre_i_to_i + i_to_j + j_to_post_j

                pre_i_to_j = distance_matrix[individual[i - 1], individual[j]]
                j_to_i = distance_matrix[individual[j], individual[i]]
                i_to_post_j = distance_matrix[individual[i], individual[(j + 1) % n]]

                swap_cost = pre_i_to_j + j_to_i + i_to_post_j

                current_cost_reduction = initial_cost - swap_cost
            else:
                pre_i_to_i = distance_matrix[individual[i - 1], individual[i]]
                post_i_to_i = distance_matrix[individual[i], individual[(i + 1) % n]]
                pre_j_to_j = distance_matrix[individual[j - 1], individual[j]]
                post_j_to_j = distance_matrix[individual[j], individual[(j + 1) % n]]

                initial_cost = pre_i_to_i + post_i_to_i + pre_j_to_j + post_j_to_j

                pre_i_to_j = distance_matrix[individual[i - 1], individual[j]]
                j_to_post_i = distance_matrix[individual[j], individual[(i + 1) % n]]
                pre_j_to_i = distance_matrix[individual[j - 1], individual[i]]
                i_to_post_j = distance_matrix[individual[i], individual[(j + 1) % n]]

                swap_cost = pre_i_to_j + j_to_post_i + pre_j_to_i + i_to_post_j

                current_cost_reduction = initial_cost - swap_cost
            if current_cost_reduction > best_cost_reduction:
                best_swap_indexes[0] = i
                best_swap_indexes[1] = j
                best_cost_reduction = current_cost_reduction
    if best_swap_indexes[0] != 0 and best_swap_indexes[1] != 0:
        individual[best_swap_indexes[0]], individual[best_swap_indexes[1]] = \
            individual[best_swap_indexes[1]], individual[best_swap_indexes[0]]

    return individual


@njit(fastmath=True)
def recombination(parents: np.ndarray,
                  children: np.ndarray,
                  nb_parents: int,
                  nb_children: int,
                  n: int) -> np.ndarray:
    i = 0

    while i < nb_children:
        subset_indices = np.random.randint(0, n, 2)
        start, end = np.min(subset_indices), np.max(subset_indices)
        if end == start:
            continue
        parent_indices = np.random.randint(0, nb_parents, 2)
        parent1 = parents[parent_indices[0]]
        parent2 = parents[parent_indices[1]]
        chosen_parent = parent1 if np.random.randint(1, 3) == 1 else parent2
        not_chosen_parent = parent1 if chosen_parent is parent2 else parent2

        child = np.full(n, -1, dtype=np.int32)
        child[start:end] = chosen_parent[start:end]
        child_set = set(child[start:end])
        child_index = end

        for j in range(n):
            if not_chosen_parent[j] not in child_set:
                child[child_index] = not_chosen_parent[j]
                child_index = (child_index + 1) % n
        children[i] = child
        i += 1
    return children


def elimination(population: np.ndarray,
                children: np.ndarray,
                population_size: int,
                n: int,
                distance_matrix: np.ndarray,
                elimination_intensity: float,
                elimination_alpha: float,
                change_elimination: bool) -> np.ndarray:
    merged_population = np.concatenate((population, children), axis=0)
    merged_population_fitness = np.array([fitness(n, individual, distance_matrix) for individual in merged_population],
                                         dtype=np.float32)
    if change_elimination:
        return merged_population[np.argsort(merged_population_fitness)][:population_size]

    merged_population_size = merged_population.shape[0]
    window_size = int(n * elimination_intensity)
    distances = np.zeros(shape=(merged_population_size, merged_population_size), dtype=np.int32)

    ii = np.random.randint(0, n)
    while ii + window_size >= n:
        ii = np.random.randint(0, n)
    jj = ii + window_size

    for i in range(merged_population_size):
        dist = 0
        for j in range(merged_population_size):
            if i != j:
                if distances[i][j] == 0 and distances[j][i] == 0:
                    distances[i][j] = distance(merged_population[i][ii:jj], merged_population[j][ii:jj], ii, jj)
                    dist += distances[i][j]
                    distances[j][i] = distances[i][j]
                else:
                    dist += distances[i][j]
        ratio = (dist / (merged_population_size - 1)) / (jj - ii - 1)
        shared_fitness_factor = (1 + ratio) ** elimination_alpha
        merged_population_fitness[i] *= shared_fitness_factor
    return merged_population[np.argsort(merged_population_fitness)][:population_size]


@njit()
def distance(individual1: np.ndarray,
             individual2: np.ndarray,
             ii: int,
             jj: int) -> int:
    n = jj - ii
    dist = 0

    for i in range(n - 1):
        common_edge = False
        for j in range(n - 1):
            if individual1[i] == individual2[j] and individual1[(i + 1) % n] == individual2[(j + 1) % n]:
                common_edge = True
                break
        dist += 1 if not common_edge else 0
    return dist


@njit(fastmath=True)
def report(individuals_fitness: np.ndarray,
           population: np.ndarray):
    mean_objective = np.mean(individuals_fitness)
    best_objective = np.min(individuals_fitness)
    max_objective = np.max(individuals_fitness)
    best_id = np.where(individuals_fitness == best_objective)[0][0]
    best_solution = population[best_id]
    return mean_objective, best_objective, max_objective, best_id, best_solution


@njit(fastmath=True)
def diversity(individuals_fitness: np.ndarray) -> np.float64:
    return 1 - (np.min(individuals_fitness) / np.max(individuals_fitness))


@njit()
def mutate_seed_population(population: np.ndarray,
                           n: int,
                           population_size: int,
                           stuck: bool,
                           beta: float) -> np.ndarray:
    if stuck:
        for www in range(1, population_size):
            if np.random.random() < beta:
                if n < 500:
                    index = np.random.randint(0, n)
                    population[www][index], population[www][(index + 1) % n] = population[www][(index + 1) % n], population[www][index]
                else:
                    indexes = np.random.randint(0, n, 2)
                    min_index = np.min(indexes)
                    max_index = np.max(indexes)
                    population[www][min_index], population[www][max_index] = population[www][max_index], population[www][min_index]

    return population


class r0924352:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.population_size = 100
        self.alpha = 0.2
        self.beta = 0.5
        self.nb_parents = 20
        self.nb_children = 40
        self.inf_penalty = 1.1
        self.elimination_alpha = 0.4
        self.k = 5
        self.stuck = False
        self.change_elimination = False
        self.diversity = np.float64(0)

    def fitness(self, individual):
        return fitness(self.N, individual, self.distanceMatrix)

    def evaluate(self):
        self.individualsFitness = evaluate(self.population_size, self.population, self.individualsFitness,
                                           self.distanceMatrix, self.N)

    def initialization(self):
        self.population = initialization(self.N, self.distanceMatrix, self.population_size,
                                         self.population, self.init_k)

    def selection(self):
        self.parents = selection(self.nb_parents, self.population, self.individualsFitness, self.k,
                                 self.N, self.population_size)

    def calculate_diversity(self):
        self.diversity = diversity(self.individualsFitness)

    def recombination(self):
        self.children = recombination(self.parents, self.children, self.nb_parents, self.nb_children, self.N)

    def mutation(self):
        self.children = mutation_operators(self.children, self.alpha, self.N, self.nb_children, self.diversity)

    def lso(self):
        self.population = general_local_search_operator(self.population,
                                                        self.distanceMatrix,
                                                        self.N,
                                                        self.int_refinement,
                                                        self.population_size)

    def elimination(self, change_elimination):
        self.population = elimination(self.population, self.children, self.population_size, self.N, self.distanceMatrix,
                                      self.elimination_intensity, self.elimination_alpha, change_elimination)

    def optimize(self, filename):
        file = open(filename, "r")
        self.distanceMatrix = np.loadtxt(file, delimiter=",", dtype=np.float32)
        file.close()

        self.bestObjective = np.inf
        self.meanObjective = np.inf
        self.maxObjective = np.inf
        self.timeLeft = np.inf
        max_value = np.max(self.distanceMatrix[self.distanceMatrix != np.inf]) * self.inf_penalty
        self.distanceMatrix[self.distanceMatrix == np.inf] = max_value
        self.N = np.int32(self.distanceMatrix.shape[0])

        if self.N <= 100:
            self.stuck_amount = 50
            self.int_refinement = 0.05
            self.init_k = 25
            self.elimination_intensity = 0.05
            self.time_left_threshold = 200
        elif 100 < self.N <= 250:
            self.stuck_amount = 100
            self.int_refinement = 0.025
            self.init_k = 50
            self.elimination_intensity = 0.025
            self.time_left_threshold = 150
        elif 250 < self.N <= 500:
            self.stuck_amount = 100
            self.int_refinement = 0.025
            self.init_k = 75
            self.elimination_intensity = 0.025
            self.time_left_threshold = 100
        elif 500 < self.N <= 750:
            self.stuck_amount = 85
            self.int_refinement = 0.025
            self.init_k = 100
            self.elimination_intensity = 0.025
            self.time_left_threshold = 100
        elif 750 < self.N <= 1000:
            self.stuck_amount = 75
            self.int_refinement = 0.025
            self.init_k = 125
            self.elimination_intensity = 0.025
            self.time_left_threshold = 80
        else:
            self.stuck_amount = 50
            self.int_refinement = 0.025
            self.init_k = 150
            self.elimination_intensity = 0.025
            self.time_left_threshold = 70

        self.population = np.zeros((self.population_size, self.N), dtype=np.int32)
        self.individualsFitness = np.arange(self.population_size, dtype=np.float32)
        self.parents = np.zeros(shape=(self.nb_parents, self.N), dtype=np.int32)
        self.children = np.zeros(shape=(self.nb_children, self.N), dtype=np.int32)

        self.initialization()
        self.evaluate()
        self.calculate_diversity()

        improving = True
        i = 0
        index_at_which_best_sol_changed = 0
        previous_best = np.inf

        while improving:
            self.population = mutate_seed_population(self.population, self.N,
                                                     self.population_size, self.stuck, self.beta)
            self.evaluate()
            self.stuck = False

            self.selection()

            self.recombination()

            self.calculate_diversity()

            self.mutation()

            self.elimination(self.change_elimination)

            self.evaluate()

            self.lso()

            self.evaluate()

            if self.timeLeft < self.time_left_threshold:
                self.change_elimination = True
                if self.N <= 100:
                    self.stuck_amount = 2500
                    self.int_refinement = 0.3
                elif 100 < self.N <= 250:
                    self.stuck_amount = 1000
                    self.int_refinement = 0.35
                elif 250 < self.N <= 500:
                    self.stuck_amount = 1500
                    self.int_refinement = 0.3
                elif 500 < self.N <= 750:
                    self.stuck_amount = 1000
                    self.int_refinement = 0.3
                elif 750 < self.N <= 1000:
                    self.stuck_amount = 5000
                    self.int_refinement = 0.3
                else:
                    self.stuck_amount = 7000
                    self.int_refinement = 0.3

            self.meanObjective, self.bestObjective, self.maxObjective, best_id, best_solution \
                = report(self.individualsFitness, self.population)

            if round(previous_best, 2) != round(self.bestObjective, 2):
                index_at_which_best_sol_changed = i
            i += 1

            if abs(index_at_which_best_sol_changed - i) > self.stuck_amount:
                self.stuck = True

            previous_best = self.bestObjective
            self.timeLeft = self.reporter.report(self.meanObjective, self.bestObjective, best_solution)
            if self.timeLeft < 0:
                break
        return 0
