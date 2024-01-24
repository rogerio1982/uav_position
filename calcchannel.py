import math

def calculate_channel(U, S,alluav):
    D = math.sqrt((U.X - S.X)**2 + (U.Y - S.Y)**2)  # Distancia de Euclides

    if D <= S.Cob and S.D:
        WN = 7.4e-13  # Ruído Branco (CORRIGIR)
        I = 0  # Interferencia gerada por outras células

        D0 = 100  # Distância Referência
        Sv = 9.4  # 8.2 to 10.6 dB
        V = 3e8  # Velocidade da luz (m/s) no vacuo
        L = V / S.Fr  # Lambda
        Hr = 1.2  # Altura de recepção
        Hb = S.H  # Altura da EstaçãoBase
        E = 16  # Equalizado

        # Parâmetros Cenário SUI path loss
        a = 3.6
        b = 0.005
        c = 20 #largura de banda (em Hz).

        A = 20 * math.log10(4 * math.pi * D0 / L) #free space path loss
        Y = (a - (b * Hb)) + (c / Hb)

        Lost = A + 10 * Y * math.log10(D / D0) + Sv - E  # Perda no Canal sui model

        Pw = 10**((S.RP - Lost) / 10) / 1000

        for small in alluav:  # calculate intercell interference
            if ((small.D and small.ID) != S.ID):
                Da = math.sqrt((small.X - U.X)**2 + (small.Y - U.Y)**2)
                LostA = small.RP - (A + 10 * Y * math.log10(Da / D0) + Sv - E)
                I = I + 10**(LostA / 10) / 1000

        Pdbm = 10 * math.log10(1000 * Pw)
        WNdbm = -91.31
        Idbm = 10 * math.log10(1000 * I)
        SINRw = (Pw / (WN + I))
        SINR = 10 * math.log10(1000 * SINRw)

        C = S.B / S.PRB  # smallcell largura de banda
        DR = C * math.log2(1 + SINRw)  # Datarate (taxa de transferência)com apenas 1 PRB sendo usado/shannon.
        CQI = round(1 + (7 / 13) * (SINRw + 6))
        PRX = round(Pdbm)
    else:
        SINR = 0
        DR = 0
        CQI = 0
        I = 0
        PRX = 0

    return DR, CQI, SINR, PRX, I
