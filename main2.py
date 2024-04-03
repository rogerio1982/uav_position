from users import Users
from uavs import Uavs
from calcchannel import calculate_channel

import random
import matplotlib.pyplot as plt
import numpy as np
import math


#define quantidade de uavbs
uav = 3
s = 10

# define qtd de resource blocks
resBlo = 100
areax = 1000
areay = 1000

# 01 Creating usuarios de forma randomica
# Atribuindo os valores para cada usuario: id,x,y, data rate, estacao,qos, snr, etc
alluser = []
t = 40
for i in range(s):
    taxa = random.randint(35, 40) # t = 400 * 1024;  # Taxa Requerida 400 Kbps
    us = Users(i, random.randint(1, areax), random.randint(1, areay), taxa, 1, 0, t, 0, 0, 0, False, 0, 0, 1, 0, 0, 0)
    alluser.append(us)

# 03 Creating uavbs em posicoes fixas
#Atribuindo os valores  cada uavbs: id, x,y,potencia,freq,banda,total,usuarios conect, estacao,totalprb, etc
alluav = []
posx = [540, 660, 780]
posy = [530, 650, 730]
h=100 #valor multiplicado por 10
for i in range(uav):
    uavs = Uavs(i, posx[i], posy[i], 23, 2.4, 18, 50, 30, True, 100, resBlo, False, "False", "implantação", h*2, h, 10, 5, 0) #estrategia pos e alt fixa sem kmeans
    alluav.append(uavs)

# 04 alocar usuarios a cada uavbs
#nessa etapa é calculado:
#distancia do usuario para uavbs, canal, prb, rsn, cqi, etc
teste = []
chanel = []
on = 0
for i in alluser:
    for x in alluav:
        if calculate_channel(i, x, alluav) != (0, 0, 0, 0, 0):
            chanel = calculate_channel(i, x, alluav)
            #if x.PRB_F >= i.PRB and (chanel[2] > 20 and chanel[2] <50 ) and i.C == False:
            if x.PRB_F >= i.PRB and i.C == False:

                i.EB = x.ID  # estacao base alocada
                i.ES = 1  # 1 small 2 macro
                i.CQI = chanel[1]
                i.SINR = chanel[2]
                i.PRX = x.RP - chanel[3]  # potencia recebida pelo usuario
                i.C = 'True'
                aux = 1
                x.PRB_F = x.PRB_F - chanel[0]  # subtrai os PRB do small cell
                x.U = x.U + 1
                on = on + 1
                x.MAX_U += 1
                i.DR=chanel[0]

#estatistica de alocacao
mediaSIN=0
mediapath=0
nub=0
for i in alluser:
    print("Status: ", i.C, "ID:", i.ID, i.X, i.Y, i.DR, i.R_DR, "UAV: ", i.EB, i.PRB, i.CQI, i.SINR, i.PRX, i.V, i.M,
          i.ES, i.EBC, i.Int,"perdas",i.PRX, "SINR",i.SINR, "Data Rate:",i.DR/ 10**6,"Mbps")
    if i.SINR:
        nub=nub+1
        mediaSIN=mediaSIN+i.SINR
        mediapath = mediapath + i.PRX

print("")
print("Estatística da proposta")
print('ON =', on)
print('OFF =', s - on)

print('MediaSIN =', mediaSIN/nub)
print('MediaPATH =', mediapath/nub)
for x in alluav:
    print("Drone:", x.ID, "MAX_USER: ", x.MAX_U, "PRB: ", x.PRB_F, "cob: ", x.Cob, "m", "Alt: ", x.H, "RecBlocks dipo",x.PRB_F)
import math
# Definindo as variáveis
snr_dB = 30  # SNR em dB quanto maior melhor a relacao sinal ruido
largura_banda_MHz = 40
largura_banda_prb_Hz = largura_banda_MHz * 10**6
#largura_banda_prb_Hz = 100 * 180 * 10**3  # Largura de banda por PRB em Hz (100 PRBs)
# Função para calcular a taxa máxima da rede em bps
def calcular_taxa_maxima(snr_dB, largura_banda_prb_Hz):
    snr_linear = 10 ** (snr_dB / 10)  # Convertendo SNR para escala linear
    taxa_maxima_bps = largura_banda_prb_Hz * math.log2(1 + snr_linear)  # Fórmula de Shannon
    return taxa_maxima_bps

# Calculando a taxa máxima da rede
taxa_maxima_rede_bps = calcular_taxa_maxima(snr_dB, largura_banda_prb_Hz)
taxa_maxima_rede_mbps = taxa_maxima_rede_bps / 10**6
print("Taxa máxima da rede:", taxa_maxima_rede_mbps, "Mbps")


# Plot todos os usuarios
plt.figure()
for x in alluser:
    plt.scatter(x.X, x.Y, marker='o', color='b')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Users')
plt.savefig('static/images/alluser.png')
#plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Simulando dados para teste
x = np.linspace(0, 1000, 100)
y = np.linspace(0, 1000, 100)
X, Y = np.meshgrid(x, y)

# Calculando a potência para cada ponto Wi-Fi na grade (simulação)
heatmap = np.zeros_like(X)
pot_ini = 10  # potência inicial do UAV

# Simulando alguns pontos UAV-BSs e usuários
#alluav = np.array([[10, 10], [20, 20], [30, 30]])  # coordenadas UAV-BSs
#alluser = np.array([[15, 15], [25, 25], [35, 35]])  # coordenadas usuários

for point in alluav:
    #x0, y0, power = point
    distance = np.sqrt((X - point.X)**2 + (Y - point.Y)**2)
    power_density = pot_ini * np.log10(distance)
    heatmap += power_density

# Plotando o heatmap
plt.figure()
plt.pcolormesh(X, Y, heatmap, cmap='viridis')
plt.colorbar(label='Potência (dBm)')

for i in range(uav):
    plt.scatter(posx[i], posy[i], color='yellow', marker='x')

for x in alluser:
    plt.scatter(x.X, x.Y, color='red', marker='o')


plt.xlabel('X')
plt.ylabel('Y')
plt.title('Heatmap de Potência UAV-BSs')
plt.legend()

# Salvando a figura
plt.savefig('static/images/heat.png')
plt.show()
