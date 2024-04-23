import numpy as np
import matplotlib.pyplot as plt
import math
import time

# Grava o tempo de início
start_time = time.time()

# Parâmetros do problema
num_users = 100
grid_size = 1000
SNR_objetivo = 25
signal_power = -80
noise_power = 38  # Potência de ruído constante
frequency = 2.4  # GHz

evaporation_rate = 0.1
alpha = 1  # Importância dos feromônios
beta = 1   # Importância da informação heurística
pheromone_deposit = 1  # Quantidade de feromônio depositada pelas formigas
pheromone_initial = 0.5  # Nível inicial de feromônio em todas as posições
max_iterations = 4
num_ants = 4

# Gerar posições aleatórias para usuários
np.random.seed(42)
user_positions = np.random.randint(0, grid_size, size=(num_users, 2))
np.random.seed(None)

# Função para calcular a distância entre dois pontos
def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Função para calcular o path loss usando o modelo ATG
def calculate_pathloss_atg(user_pos, wifi_pos, D):
    white_noise = 7.4e-13
    # Interefência gerada por outras células
    interference = 0
    frequency = 2.5
    I = 0  # Interferencia gerada por outras células
    receiving_antenna_height = 1.6  # Altura da antena receptora em metros
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

    return Pdbm

# Função para calcular o SNR
def calculate_snr(user_pos, wifi_pos, frequency, distance):
    pathloss = calculate_pathloss_atg(user_pos, wifi_pos, distance)
    received_power = signal_power - pathloss
    snr = received_power - noise_power
    return snr

# Função de aptidão
def fitness(wifi_positions, user_positions, SNR_Obj):
    total_users_connected = [0] * len(wifi_positions)  # Lista para armazenar o número de usuários conectados em cada ponto de acesso
    connected_users = set()

    min_distance = 100 # Distância mínima desejada entre os pontos de acesso

    # Calcular a penalidade pela proximidade dos pontos de acesso
    penalty = 0
    for i in range(len(wifi_positions)):
        for j in range(i + 1, len(wifi_positions)):
            dist = calculate_distance(wifi_positions[i], wifi_positions[j])
            if dist < min_distance:
                penalty += min_distance - dist
                print(penalty)

    for user_pos in user_positions:
        max_snr = float('-inf')
        for idx, wifi_pos in enumerate(wifi_positions):
            distance = calculate_distance(user_pos, wifi_pos)
            snr = calculate_snr(user_pos, wifi_pos, frequency, distance)
            if snr > SNR_Obj:
                connected_users.add(tuple(user_pos))
                total_users_connected[idx] += 1  # Incrementar o número de usuários conectados neste ponto de acesso
                break

    #total_users_connected += len(connected_users)
    total_users_connected = tuple(total_users_connected)
    fitness_score = sum(total_users_connected) - penalty  # Aplicar penalidade
    #fitness_score = total_users_connected - penalty  # Aplicar penalidade
    
    return fitness_score, len(wifi_positions), snr, total_users_connected


# Algoritmo ACO
def ant_colony_optimization(num_ants, max_iterations):
    pheromones = np.ones((grid_size, grid_size)) * pheromone_initial
    best_fitness = 0
    best_solution = None
    # Inicialização dos melhores resultados pessoais e globais
    personal_best_fitness = np.zeros(num_ants)
    personal_best_positions = [None] * num_ants  # Corrigido para inicializar uma lista de None
    global_best_fitness = -np.inf
    global_best_position = None  # Deve ser None inicialmente, ou uma posição específica se conhecida


    for iteration in range(max_iterations):
        ant_positions = np.random.randint(0, grid_size, size=(num_ants, 2))
        for ant_id in range(num_ants):
            fitness_score, _, _, _ = fitness([ant_positions[ant_id]], user_positions, SNR_objetivo)
            pheromones[ant_positions[ant_id][0], ant_positions[ant_id][1]] += fitness_score

        # Atualização do melhor pessoal
        if fitness_score > personal_best_fitness[ant_id]:
            personal_best_fitness[ant_id] = fitness_score
            personal_best_positions[ant_id] = ant_positions.copy()

        # Atualização do melhor global
        if fitness_score > global_best_fitness:
            global_best_fitness = fitness_score
            global_best_position = ant_positions.copy()   

        print(global_best_fitness)

        # Atualiza os feromônios com base nas melhores soluções encontradas
        pheromones *= (1 - evaporation_rate)  # Evaporação dos feromônios
        for pos in global_best_position:
            pheromones[pos[0], pos[1]] += pheromone_deposit

        best_solution = global_best_position
        best_fitness = global_best_fitness

    return best_solution, best_fitness

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

# Execução do algoritmo
best_solution, best_fitness = ant_colony_optimization(num_ants, max_iterations)
print("Melhor posição dos pontos de acesso:", best_solution)
print("Fitness máximo alcançado:", best_fitness)

# Calcular o número de usuários conectados em cada ponto de acesso
fitness_score, _, _, total_users_connected = fitness(best_solution, user_positions, SNR_objetivo)
print("Número de usuários conectados em cada ponto de acesso:", total_users_connected)
print("Número de usuários conectados:", sum(total_users_connected))

# Tempo total de execução
end_time = time.time()
total_time = end_time - start_time
print("Tempo total de execução: {:.2f} segundos".format(total_time))

# Plotar o resultado com o raio de cobertura real
plt.figure(figsize=(6, 6))

for i, pos in enumerate(user_positions):
    plt.annotate(str(i), (pos[0], pos[1]), textcoords="offset points", xytext=(0,5), ha='center')

plt.scatter(user_positions[:, 0], user_positions[:, 1], marker='o', label='Usuários') 
plt.scatter(best_solution[:, 0], best_solution[:, 1], marker='x', color='red', label='Ponto de Acesso')    

# Adicionar círculos de raio de cobertura
for wifi_pos in best_solution:
    circle = plt.Circle((wifi_pos[0], wifi_pos[1]), calculate_coverage_radius(wifi_pos), color='gray', alpha=0.2)
    plt.gca().add_patch(circle)  

plt.title('Área de Cobertura')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.legend()
plt.grid(True)
plt.show()
