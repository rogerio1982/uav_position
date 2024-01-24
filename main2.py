from users import Users
from uavs import Uavs
from calcchannel import calculate_channel
import random
import matplotlib.pyplot as plt
import numpy as np


#define quantidade de uavbs
uav = 3
s = 200

# define qtd de resource blocks
resBlo = 100
areax = 1000
areay = 1000

# 01 Creating usuarios de forma randomica
# Atribuindo os valores para cada usuario: id,x,y, data rate, estacao,qos, snr, etc
alluser = []
T = 40  # t = 400 * 1024;  # Taxa Requerida 400 Kbps
for i in range(s):
    us = Users(i, random.randint(1, areax), random.randint(1, areay), 1, 1, 0, T, 0, 0, 0, False, 0, 0, 1, 0, 0, 0)
    alluser.append(us)

# 03 Creating uavbs em posicoes fixas
#Atribuindo os valores  cada uavbs: id, x,y,potencia,freq,banda,total,usuarios conect, estacao,totalprb, etc
alluav = []
posx = [140, 360, 280]
posy = [130, 350, 230]
h=300
for i in range(uav):
    uavs = Uavs(i, posx[i], posy[i], 23, 2.4, 20, 50, 30, True, 100, resBlo, False, "False", "implantação", h*2, h, 10, 5, 0) #estrategia pos e alt fixa sem kmeans
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

#estatistica de alocacao
for i in alluser:
    print("Status: ", i.C, "ID:", i.ID, i.X, i.Y, i.DR, i.R_DR, "UAV: ", i.EB, i.PRB, i.CQI, i.SINR, i.PRX, i.V, i.M,
          i.ES, i.EBC, i.Int)
print("")
print("Estatística da proposta")
print('ON =', on)
print('OFF =', s - on)

for x in alluav:
    print("Drone:", x.ID, "MAX_USER: ", x.MAX_U, "PRB: ", x.PRB_F, "cob: ", x.Cob, "m", "Alt: ", x.H, "m")


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
