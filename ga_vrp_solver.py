import numpy as np
import matplotlib.pyplot as plt
import random
import re
import os
import urllib.request
import csv

# ---------------- Lecture de l'instance ---------------- #
def read_vrp_euc_2d(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    capacity = int(re.search(r'CAPACITY\s*:\s*(\d+)', ''.join(lines)).group(1))

    node_coord_start = next(i for i, line in enumerate(lines) if 'NODE_COORD_SECTION' in line) + 1
    demand_start = next(i for i, line in enumerate(lines) if 'DEMAND_SECTION' in line) + 1
    depot_start = next(i for i, line in enumerate(lines) if 'DEPOT_SECTION' in line) + 1

    coords = []
    for line in lines[node_coord_start:demand_start - 1]:
        parts = line.strip().split()
        coords.append((float(parts[1]), float(parts[2])))

    demands = []
    for line in lines[demand_start:depot_start - 1]:
        parts = line.strip().split()
        demands.append(int(parts[1]))

    depot = int(lines[depot_start].strip()) - 1

    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist_matrix[i][j] = math.hypot(dx, dy)

    return coords, demands, depot, capacity, dist_matrix

# ---------------- Outils AG ---------------- #
def calculate_total_distance(solution, dist_matrix):
    return sum(dist_matrix[route[i]][route[i+1]] for route in solution for i in range(len(route)-1))

def initialize_population(pop_size, num_clients):
    return [random.sample(range(1, num_clients), num_clients - 1) for _ in range(pop_size)]

def split_routes(individual, demands, capacity):
    routes = []
    route = [0]
    load = 0
    for client in individual:
        if load + demands[client] <= capacity:
            route.append(client)
            load += demands[client]
        else:
            route.append(0)
            routes.append(route)
            route = [0, client]
            load = demands[client]
    route.append(0)
    routes.append(route)
    return routes

def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    return min(selected, key=lambda x: x[1])[0]

def pmx_crossover(p1, p2):
    size = len(p1)
    c = [-1]*size
    cx1, cx2 = sorted(random.sample(range(size), 2))
    c[cx1:cx2+1] = p1[cx1:cx2+1]
    for i in range(cx1, cx2+1):
        if p2[i] not in c:
            val = p2[i]
            pos = i
            while True:
                val_in_p1 = p1[pos]
                if val_in_p1 not in c:
                    pos = p2.index(val_in_p1)
                else:
                    break
            c[pos] = val
    for i in range(size):
        if c[i] == -1:
            c[i] = p2[i]
    return c

# ---------------- Mutation guidée par DQN ---------------- #
# Fictive pour l’exemple : sélectionne la meilleure parmi N mutations aléatoires
def dqn_mutation(individual, dqn_model, dist_matrix, demands, capacity, mutation_rate=0.02):
    if random.random() < mutation_rate:
        k = 5
        candidates = []
        costs = []
        for _ in range(k):
            mutated = individual[:]
            i, j = sorted(random.sample(range(len(mutated)), 2))
            mutated[i], mutated[j] = mutated[j], mutated[i]
            routes = split_routes(mutated, demands, capacity)
            cost = calculate_total_distance(routes, dist_matrix)
            candidates.append(mutated)
            costs.append(cost)
        best_idx = np.argmin(costs)
        return candidates[best_idx]
    return individual

# ---------------- Algorithme génétique avec DQN ---------------- #
def genetic_algorithm_dqn(demands, depot, capacity, dist_matrix, dqn_model,
                          generations=100, pop_size=50, return_history=False):
    num_clients = len(demands)
    population = initialize_population(pop_size, num_clients)
    best_costs_per_gen = []

    for gen in range(generations):
        fitnesses = []
        for individual in population:
            routes = split_routes(individual, demands, capacity)
            cost = calculate_total_distance(routes, dist_matrix)
            fitnesses.append(cost)

        best_costs_per_gen.append(min(fitnesses))

        new_population = []
        for _ in range(pop_size):
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            child = pmx_crossover(p1, p2)
            child = dqn_mutation(child, dqn_model, dist_matrix, demands, capacity)
            new_population.append(child)

        population = new_population

    best_cost = float('inf')
    best_solution = None
    for individual in population:
        routes = split_routes(individual, demands, capacity)
        cost = calculate_total_distance(routes, dist_matrix)
        if cost < best_cost:
            best_cost = cost
            best_solution = routes

    if return_history:
        return best_solution, best_cost, best_costs_per_gen
    else:
        return best_solution, best_cost

# ---------------- Visualisation ---------------- #
def plot_routes(routes, coords):
    plt.figure(figsize=(8, 6))
    for route in routes:
        xs = [coords[i][0] for i in route]
        ys = [coords[i][1] for i in route]
        plt.plot(xs, ys, marker='o')
    plt.scatter(coords[0][0], coords[0][1], c='red', s=100, label='Depot')
    plt.title("Tournées optimisées (AG + DQN)")
    plt.legend()
    plt.grid()
    plt.show()

def plot_convergence_curve(costs_per_gen):
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(costs_per_gen)), costs_per_gen, marker='o', color='blue')
    plt.title("Courbe de convergence du coût")
    plt.xlabel("Génération")
    plt.ylabel("Meilleur coût")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------- MAIN ---------------- #
# Exemple fictif de dqn_model (aucun modèle réel nécessaire ici)
class DummyDQN:
    def predict(self, state_batch, verbose=0):
        # Retourne des valeurs aléatoires comme Q-values
        batch_size = len(state_batch)
        num_actions = len(state_batch[0]) if batch_size > 0 else 1
        return np.random.rand(batch_size, num_actions)


# Charger instance
coords, demands, depot, capacity, dist_matrix = read_vrp_euc_2d("F-n45-k4.vrp")

# Appliquer AG + DQN
start_time = time.time()
solution, cost, cost_history = genetic_algorithm_dqn(
    demands, depot, capacity, dist_matrix, dqn_model=DummyDQN(),
    generations=100, pop_size=50, return_history=True
)
end_time = time.time()

print(f"✅ Coût final = {cost:.2f}")
print(f"⏱️ Temps total : {end_time - start_time:.2f} secondes")

# Affichage
plot_routes(solution, coords)
plot_convergence_curve(cost_history)
