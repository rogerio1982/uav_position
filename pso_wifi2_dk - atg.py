import numpy as np
import matplotlib.pyplot as plt
import math
import time

# Grava o tempo de início
start_time = time.time()

# Parâmetros do problema
num_users = 100
grid_size = 1000
#coverage_radius = 50  # Ajuste conforme necessário
#min_distance_between_points = 200  # Distância mínima entre pontos WiFi
#altura = 50
SNR_objetivo = 25

signal_power = -80
noise_power = 40  # Adicionamos uma potência de ruído constante para simplificar o exemplo
frequency = 2.4 #GHz - para um canal Wifi de 20MHz

# Gerar posições aleatórias para usuários
np.random.seed(42)
user_positions = np.random.randint(0, grid_size, size=(num_users, 2))
np.random.seed(None)

# Função para calcular o SNR
def calculate_snr(user_pos, wifi_pos, frequency, distance):
    #pathloss_sui = calculate_pathloss_sui(user_pos, wifi_pos, frequency, distance)
    pathloss_sui = calculate_pathloss_atg(user_pos, wifi_pos, distance)
    received_power = signal_power - pathloss_sui
    snr = received_power - noise_power

    return snr

# Função para calcular a distância entre dois pontos
def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2) #calculo da distância euclidiana entre o usuario e o uav


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
    seta = math.atan((base_station_height - base_station_height) / D)

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

# Função de aptidão
def fitness(wifi_positions, user_positions, SNR_Obj):
    total_users_connected = 0
    connected_users = set()

    for user_pos in user_positions:
        max_snr = float('-inf')

        for wifi_pos in wifi_positions:
            distance = calculate_distance(user_pos, wifi_pos)
            snr = calculate_snr(user_pos, wifi_pos, frequency, distance)

            if snr > SNR_Obj:
                connected_users.add(tuple(user_pos))
                break

    total_users_connected = len(connected_users)
    return total_users_connected, len(wifi_positions), snr

def max_coverage_distance(wifi_pos):
    # Iniciar com uma distância de teste e ajustar para encontrar o SNR = SNR_Objetivo
    test_distance = 1  # Começa com 1 metro
    while True:
        snr = calculate_snr(wifi_pos, wifi_pos, frequency, test_distance)
        if snr < SNR_objetivo:
            break
        test_distance += 1  # Incrementa a distância de teste até que o SNR caia abaixo do SNR_Objetivo
    return test_distance

# Função para inicializar as partículas
def initialize_particles(num_particles, space_boundaries, num_dimensions):
    particles = np.random.uniform(low=space_boundaries[0], high=space_boundaries[1],
                                  size=(num_particles, num_dimensions)) #aqui altera a quant uavs dk."...,num_dimensions)" indica que cada particula tem x dimensões, ou seja, cada partícula é representada por um vetor de x valores, que são x/2 posições (x,y)
    return particles

# Função de atualização da velocidade
def update_velocity(particle, velocity, personal_best, global_best, inertia_weight, cognitive_weight, social_weight):
    inertia_term = inertia_weight * velocity
    cognitive_term = cognitive_weight * np.random.rand() * (personal_best - particle)
    social_term = social_weight * np.random.rand() * (global_best - particle)
    new_velocity = inertia_term + cognitive_term + social_term
    return new_velocity


# Função de atualização da posição
def update_position(particle, velocity, space_boundaries):
    new_position = particle + velocity
    # Garantir que as novas posições estejam dentro dos limites do espaço
    new_position = np.clip(new_position, space_boundaries[0], space_boundaries[1])
    return new_position

# Parâmetros do PSO
num_particles = 100
max_iterations = 10
space_boundaries = (0, grid_size)  # Limites do espaço bidimensional
num_dimensions = 4  # Iniciando com x dimensões (x/2 pontos de acesso)

while True:
    # Inicialização das partículas e velocidades
    particles = initialize_particles(num_particles, space_boundaries, num_dimensions)
    particles_velocity = np.zeros_like(particles)  # Inicialização das velocidades das partículas

    # Inicialização dos melhores resultados pessoais e globais
    personal_best_positions = particles.copy()
    personal_best_fitness = np.zeros(num_particles)
    global_best_position = np.zeros(10)
    global_best_fitness = -np.inf

    # Lista para armazenar a evolução do fitness
    fitness_evolution = []

    # Algoritmo PSO
    start_time_pso = time.time()
    total_time_cov = 0

    for iteration in range(max_iterations):
        for i in range(num_particles):
            # Avaliação da função de fitness para cada partícula
            wifi_positions = particles[i].reshape(-1, 2)
            start_time_cov = time.time()
            coverage_radii = [max_coverage_distance(wifi_pos) for wifi_pos in wifi_positions]
            end_time_cov = time.time()
            total_time_cov += end_time_cov - start_time_cov
            total_users_connected, num_wifi_points, SNR_user = fitness(wifi_positions, user_positions,SNR_objetivo)
            fitness_score = total_users_connected

            # Atualização do melhor pessoal
            if fitness_score > personal_best_fitness[i]:
                personal_best_fitness[i] = fitness_score
                personal_best_positions[i] = particles[i].copy()

            # Atualização do melhor global
            if fitness_score > global_best_fitness:
                global_best_fitness = fitness_score
                global_best_position = particles[i].copy()

            # Atualização da velocidade e posição da partícula
            inertia_weight = 0.5
            cognitive_weight = 0.5
            social_weight = 0.8

            new_velocity = update_velocity(
                particles[i], particles_velocity[i], personal_best_positions[i], global_best_position,
                inertia_weight, cognitive_weight, social_weight)
            particles[i] = update_position(particles[i], new_velocity, space_boundaries)
        
        # Armazenar o fitness atual na lista de evolução
        fitness_evolution.append(global_best_fitness)

    end_time_pso = time.time()
    total_time_pso = end_time_pso - start_time_pso

    print(f"Qtd. UAV: {num_dimensions/2}, Usuários Conectados: {global_best_fitness}")
    print("Tempo de execução PSO: {:.2f} segundos".format(total_time_pso-total_time_cov))
    print("Tempo de Calculo da Cobertura do sinal: {:.2f} segundos".format(total_time_cov))

    if global_best_fitness >= num_users:
        break  # Finaliza se atingir o fitness desejado

    num_dimensions += 2  # Aumenta em 2 as dimensões para cada iteração, ou seja, incrementa 1 uav
    return 

# Resultado final
print("Melhor posição dos pontos de acesso:", global_best_position)
print("Fitness máximo alcançado:", global_best_fitness)

end_time = time.time()
total_time = end_time - start_time

print("Tempo total de execução: {:.2f} segundos".format(total_time))

# Plotar o resultado com o raio de cobertura real
plt.figure(figsize=(6, 6))
plt.scatter(user_positions[:, 0], user_positions[:, 1], marker='o', label='Usuários')

# Adicionar numeração para usuários
for i, pos in enumerate(user_positions):
    plt.annotate(str(i), (pos[0], pos[1]), textcoords="offset points", xytext=(0,5), ha='center')

best_wifi_positions = global_best_position.reshape(-1, 2)  # Usando a posição que foi resultado do PSO 
for i, wifi_pos in enumerate(best_wifi_positions):
    plt.scatter(wifi_pos[0], wifi_pos[1], marker='x', color='red', label='Ponto de Acesso' if i == 0 else "")
    coverage_circle = plt.Circle(wifi_pos, coverage_radii[i], color='gray', alpha=0.2)    
    plt.gca().add_patch(coverage_circle)
    
plt.title('Área de Cobertura')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.legend()
plt.grid(True)
plt.show()