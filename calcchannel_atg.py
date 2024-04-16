import math

def calculate_channel(U, S,alluav):
    #D = math.sqrt((U.X - S.X)**2 + (U.Y - S.Y)**2)**0.5  # Distancia de Euclides
    D = math.sqrt((U.X - S.X) ** 2 + (U.Y - S.Y) ** 2)  # Distancia de Euclides

    if D <= S.Cob and S.D:

        white_noise = 7.4e-13
        # Interefência gerada por outras células
        interference = 0
        I = 0  # Interferencia gerada por outras células
        receiving_antenna_height = 1.6  # Altura da antena receptora em metros
        base_station_height = S.H  # Altura da EstaçãoBase  # Altura da estação base
        f = 2.4#S.F / 1e9
        env = 2

        ax = [15, 11, 5, 5]
        bx = [.16, .18, .3, .3]
        a = ax[env]
        b = bx[env]
        # ====antenna loss=====================
        A = 1  # to calculate with, A=0 to calculate without antenna loss
        # =========max antenna gain=============%
        Go = 2.4#2.15
        # =============antenna 3db bandwidth=======%
        seta_3db = 20#76
        # ==reflection loss===================%
        L_r = .3

        #antigo
        WN = 7.4e-13  # Ruído Branco (CORRIGIR)
        D0 = 100  # Distância Referência
        Sv = 9.4  # 8.2 to 10.6 dB
        E = 16  # Equalizado

        seta = math.atan((base_station_height - base_station_height) / D)

        lost = (-147.5 + 20 * math.log10(f) + 20 * math.log10(D) - 20 * math.log10(math.cos(math.pi / 180 * seta))) \
        - A * (2 * Go - (12 * ((seta) / seta_3db) ** 2) - (12 * ((seta) / seta_3db) ** 2)) \

        + 20 * math.log10((10 ** ((-68.8 + 10 * math.log10(f) + 10 * math.log10(base_station_height - receiving_antenna_height) \
        + 20 * math.log10(math.cos(math.pi * seta / 180)) - 10 * math.log10(1 + math.sqrt(2) / (L_r ** 2))) / 20) * (1 - (1 / (a * math.exp(-b * (seta - a)) + 1)))) \
        + (1 * (1 / (a * math.exp(-b * (seta - a)) + 1))))



        #Pw = 10 ** ((base_station.transmit_power - lost) / 10) / 1000
        Pw = 10 ** ((S.RP - lost) / 10) / 1000

        for small in alluav:  # calculate intercell interference
            if ((small.D and small.ID) != S.ID):
                Da = math.sqrt((small.X - U.X)**2 + (small.Y - U.Y)**2)**0.5
                #LostA = small.RP - (A + 10 * math.log10(Da / D0) + Sv - E)
                LostA = small.RP - (92.45 + 20 * math.log10(D/1000) + 20 * math.log10(f/1e9))

                I = I + 10**(LostA / 10) / 1000
                print("inter",I)


        Pdbm = 10 * math.log10(1000 * Pw)
        WNdbm = -91.31
        Idbm = 10 * math.log10(1000 * I)
        SINRw = (Pw / (WN + I))
        SINR = 10 * math.log10(1000 * SINRw)

        C = S.B / S.PRB  # smallcell
        DR = C * math.log2(1 + SINRw)  # Datarate com apenas 1 PRB sendo usado/shannon.
        CQI = round(1 + (7 / 13) * (SINRw + 6))
        PRX = round(Pdbm)
        #print(DR)


    else:
        SINR = 0
        DR = 0
        CQI = 0
        I = 0
        PRX = 0

    return DR, CQI, SINR, PRX, I
