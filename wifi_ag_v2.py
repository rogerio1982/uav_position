import numpy as np
import matplotlib.pyplot as plt
import math
import time

# Grava o tempo de início
start_time = time.time()

# Parâmetros do problema
num_users = 100
num_wifi_points = 4
grid_size = 1000
#coverage_radius = 300  # Ajuste conforme necessário
min_distance_between_points = 10  # Distância mínima entre pontos WiFi
#altura = 50
signal_power = -80 #dBm
noise_power = 38  # Adicionamos uma potência de ruído constante
frequency = 2.5 #GHz
SNR_objetivo = 25 # dB

# Gerar posições aleatórias para usuários
np.random.seed(42)
user_positions = np.random.randint(0, grid_size, size=(num_users, 2))
np.random.seed(None)

import math
import numpy as np

# Função para calcular a distância entre dois pontos
def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Função para calcular o path loss usando o modelo SUI
def calculate_pathloss_sui(user_pos, wifi_pos, frequency, d):
    # Parâmetros do modelo SUI
    alpha = 4.0  # Expoente do caminho (path) - relacionado ao ambiente urbano
    beta_user = 0.0065 * 50  # Atenuação devida à altura da antena do usuário (50 metros)
    beta_bs = 0  # Atenuação devida à altura da antena da estação base
    gamma = 0  # Constante

    # Cálculo do path loss usando o modelo SUI
    delta_f = 20 * math.log10(frequency) + 20 * math.log10(d) + 20 * math.log10(4 * math.pi / 3e8) - 147.55
    pathloss = delta_f + 10 * alpha * math.log10(d) + beta_user - beta_bs - gamma

    return pathloss

def calculate_pathloss_atg(user_pos, wifi_pos, D):
    white_noise = 7.4e-13
    # Interefência gerada por outras células
    interference = 0
    frequency = 2.5
    I = 0  # Interferencia gerada por outras células
    receiving_antenna_height = 1.2  # Altura da antena receptora em metros
    base_station_height = 50  # Altura da EstaçãoBase  # Altura da estação base
    f = 2.4  # S.F / 1e9
    env = 2

    ax = [15, 11, 5, 5]
    bx = [.16, .18, .3, .3]
    a = ax[env]
    b = bx[env]
    # ====antenna loss=====================
    A = 1  # to calculate with, A=0 to calculate without antenna loss
    # =========max antenna gain=============%
    Go = 2.4  # 2.15
    # =============antenna 3db bandwidth=======%
    seta_3db = 20  # 76
    # ==reflection loss===================%
    L_r = .3

    # antigo
    WN = 7.4e-13  # Ruído Branco (CORRIGIR)
    D0 = 100  # Distância Referência
    Sv = 9.4  # 8.2 to 10.6 dB
    E = 16  # Equalizado

    #pathloss = math.atan((base_station_height - base_station_height) / D)
    seta = math.atan((base_station_height - receiving_antenna_height) / D)

    pathloss = (-147.5 + 20 * math.log10(f) + 20 * math.log10(D) - 20 * math.log10(math.cos(math.pi / 180 * seta))) \
           - A * (2 * Go - (12 * ((seta) / seta_3db) ** 2) - (12 * ((seta) / seta_3db) ** 2)) \
 \
           + 20 * math.log10(
        (10 ** ((-68.8 + 10 * math.log10(f) + 10 * math.log10(base_station_height - receiving_antenna_height) \
                 + 20 * math.log10(math.cos(math.pi * seta / 180)) - 10 * math.log10(
                    1 + math.sqrt(2) / (L_r ** 2))) / 20) * (1 - (1 / (a * math.exp(-b * (seta - a)) + 1)))) \
        + (1 * (1 / (a * math.exp(-b * (seta - a)) + 1))))
    rp=23
    Pw = 10 ** ((rp - pathloss) / 10) / 1000

    Pdbm = -(10 * math.log10(1000 * Pw))
    #print(Pdbm)

    return Pdbm

# Função para calcular o raio de cobertura com base no SNR
def calculate_coverage_radius(wifi_pos):
    distance = 1  # Começa com 1 metro
    while True:
        pathloss = calculate_pathloss_atg(user_positions[0], wifi_pos, distance)  # Calcular pathloss para o primeiro usuário
        received_power = signal_power - pathloss
        snr = received_power - noise_power
        if snr < SNR_objetivo:
           return distance
        distance += 1

# Função de aptidão
def fitness(wifi_positions, user_positions):

    total_users_connected = 0
    users_connected_per_point = np.zeros(len(wifi_positions))

    for user_pos in user_positions:
        max_snr = float('-inf')
        selected_antenna = -1

        for i, wifi_pos in enumerate(wifi_positions):
            distance = calculate_distance(user_pos, wifi_pos)
            #pathloss_sui = calculate_pathloss_sui(user_pos, wifi_pos, frequency, distance)
            pathloss_sui = calculate_pathloss_atg(user_pos, wifi_pos, distance)
            received_power = signal_power - pathloss_sui
            snr = received_power - noise_power

            #if snr > SNR_objetivo and users_connected_per_point[i] < num_users // num_wifi_points:
            if snr > SNR_objetivo:
                # Conectar usuário à antena se ela não atingiu a capacidade máxima
                #if snr > max_snr:
                    #max_snr = snr
                selected_antenna = i
                    #print("path loss: ", snr)

        if selected_antenna != -1:
            users_connected_per_point[selected_antenna] += 1
            total_users_connected += 1

    return total_users_connected, users_connected_per_point


# Função para inicializar a população
def initialize_population(population_size):
    population = []
    # Loop para gerar cada indivíduo na população
    for _ in range(population_size):
        # Criar um indivíduo aleatório representando a posição dos pontos de acesso
        individual = np.random.randint(0, grid_size, size=(num_wifi_points, 2))

        # Garantir que não existam pontos sobrepostos e que a distância mínima seja respeitada
        for i in range(num_wifi_points):
            for j in range(i):
                while calculate_distance(individual[i], individual[j]) < min_distance_between_points:
                    # Ajustar posição se a distância mínima não for respeitada
                    individual[i] = np.random.randint(0, grid_size, size=2)

        population.append(individual)
    return population


# Função de cruzamento (crossover)
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, num_wifi_points)
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]), axis=0)
    return child


# Função de mutação
def mutate(individual, mutation_rate=0.1):
    mutation_mask = np.random.rand(*individual.shape) < mutation_rate
    individual[mutation_mask] = np.random.randint(0, grid_size, size=np.sum(mutation_mask))
    while len(set(map(tuple, individual))) < num_wifi_points:
        # Garantir que não existam pontos sobrepostos
        individual[mutation_mask] = np.random.randint(0, grid_size, size=np.sum(mutation_mask))
    return individual


# Algoritmo genético
def genetic_algorithm(num_generations, population_size):
    population = initialize_population(population_size)

    for generation in range(num_generations):
        # Avaliar a aptidão de cada indivíduo na população
        fitness_scores = [fitness(individual, user_positions) for individual in population]

        # Selecionar os melhores indivíduos para reprodução
        selected_indices = np.argsort([score[0] for score in fitness_scores])[-population_size // 2:]
        selected_population = [population[i] for i in selected_indices]

        # Criar nova geração usando crossover e mutação
        new_population = []
        while len(new_population) < population_size:
            parent1 = selected_population[np.random.choice(len(selected_population))]
            parent2 = selected_population[np.random.choice(len(selected_population))]
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # Encontrar o melhor indivíduo na última geração
    best_individual = max(population, key=lambda x: fitness(x, user_positions)[0])

    return best_individual, fitness(best_individual, user_positions)


# Executar o algoritmo genético
time_start_ag = time.time()
best_wifi_positions, user_stats = genetic_algorithm(num_generations=50, population_size=50)
time_end_ag = time.time()
time_total_ag = time_end_ag - time_start_ag
# Inicializar a contagem total de usuários alocados
total_users_allocated = 0

# Imprimir a quantidade de usuários alocados por ponto de acesso e a quantidade de não alocados
for i, wifi_pos in enumerate(best_wifi_positions):
    users_connected = int(user_stats[1][i])
    print(f'Ponto de Acesso {i + 1}: {users_connected} usuários alocados')
    snr_values = []

    for user_pos in user_positions:
        distance = calculate_distance(user_pos, wifi_pos)
        #received_power = 100 - 0.2 * distance
        pathloss_sui = calculate_pathloss_atg(user_pos, wifi_pos, distance)
        received_power = signal_power - pathloss_sui
        snr = received_power - noise_power  # Assuming constant noise power of global variable
        snr_values.append(snr)

    print(f'Relação Sinal-Ruído Média: {np.mean(snr_values):.2f} dB\n')

    total_users_allocated += users_connected

users_not_allocated = num_users - total_users_allocated

print(f'Usuários Alocados: {total_users_allocated}')
print(f'Usuários não alocados: {users_not_allocated}')

# Grava o tempo de término
end_time = time.time()

# Calcula o tempo total de execução
execution_time = end_time - start_time

print(f"Tempo de Execução AG: {time_total_ag} segundos")
print(f"Tempo Total de Execução: {execution_time} segundos")

# Plotar o resultado
plt.figure(figsize=(6, 6))
plt.scatter(user_positions[:, 0], user_positions[:, 1], marker='o', label='Usuários')
plt.scatter(best_wifi_positions[:, 0], best_wifi_positions[:, 1], marker='x', color='red', label='UAVBS')
for wifi_pos, users_connected in zip(best_wifi_positions, user_stats[1]):
    plt.text(wifi_pos[0], wifi_pos[1], f'Conectados: {int(users_connected)}', ha='center', va='bottom')

# Adicionar círculos de raio de cobertura
for wifi_pos in best_wifi_positions:
    circle = plt.Circle((wifi_pos[0], wifi_pos[1]), calculate_coverage_radius(wifi_pos), color='gray', alpha=0.2)
    plt.gca().add_patch(circle)

plt.title('Alocação Ótima de UAVBS com Algoritmo Genético')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.legend()
plt.show()
