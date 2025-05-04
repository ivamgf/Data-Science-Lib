import numpy as np
import random
import copy

# Configurações iniciais
dias_semana = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]
horarios = ["19:00", "19:50", "20:50", "21:40"]

# Matriz inicial (5x4) com todos elementos iguais a 1
def criar_matriz_inicial():
    return np.ones((5, 4), dtype=int)

# Ajusta a matriz para dias não-letivos (0 = dia não letivo)
def ajustar_matriz_para_feriados(matriz, dias_nao_letivos):
    matriz_ajustada = matriz.copy()
    for dia in dias_nao_letivos:
        matriz_ajustada[dia, :] = 0
    return matriz_ajustada

# Dados dos professores
professores = [
    {"id": "P1",  "mat": 48273, "ch": 108, "tipo": 1, "saldo": 108, "rest": []},
    {"id": "P2",  "mat": 19385, "ch": 108, "tipo": 1, "saldo": 108, "rest": []},
    {"id": "P3",  "mat": 60714, "ch": 108, "tipo": 1, "saldo": 108, "rest": [3]},  # quinta
    {"id": "P4",  "mat": 84529, "ch": 72,  "tipo": 2, "saldo": 72,  "rest": []},
    {"id": "P5",  "mat": 21947, "ch": 72,  "tipo": 2, "saldo": 72,  "rest": []},
    {"id": "P6",  "mat": 73018, "ch": 72,  "tipo": 2, "saldo": 72,  "rest": []},
    {"id": "P7",  "mat": 59136, "ch": 36,  "tipo": 3, "saldo": 36,  "rest": []},
    {"id": "P8",  "mat": 36402, "ch": 36,  "tipo": 3, "saldo": 36,  "rest": []},
    {"id": "P9",  "mat": 10675, "ch": 36,  "tipo": 3, "saldo": 36,  "rest": []},
    {"id": "P10", "mat": 95821, "ch": 36,  "tipo": 3, "saldo": 36,  "rest": []},
]

# Reduz o saldo do professor
def reduzir_saldo(professor, tempo):
    professor['saldo'] -= tempo
    if professor['saldo'] < 0:
        professor['saldo'] = 0

# Seleciona professores para dias letivos com base em tipo e restrição
def selecionar_professores_para_dias(matriz_ajustada, professores):
    dias_disponiveis = [i for i, dia in enumerate(matriz_ajustada) if any(dia)]
    alocacao = [0]*5

    usados = set()
    tipo1 = [p for p in professores if p["tipo"] == 1 and p["saldo"] > 0]
    tipo2 = [p for p in professores if p["tipo"] == 2 and p["saldo"] > 0]
    tipo3 = [p for p in professores if p["tipo"] == 3 and p["saldo"] > 0]

    random.shuffle(dias_disponiveis)

    for i in range(min(2, len(dias_disponiveis))):
        dia = dias_disponiveis[i]
        candidato = next((p for p in tipo1 if dia not in p["rest"] and p["mat"] not in usados), None)
        if candidato:
            alocacao[dia] = candidato["mat"]
            usados.add(candidato["mat"])
            reduzir_saldo(candidato, 3.33)

    if len(dias_disponiveis) > 2:
        dia = dias_disponiveis[2]
        candidato = next((p for p in tipo2 if dia not in p["rest"] and p["mat"] not in usados), None)
        if candidato:
            alocacao[dia] = candidato["mat"]
            usados.add(candidato["mat"])
            reduzir_saldo(candidato, 3.33)

    for dia in dias_disponiveis:
        if alocacao[dia] == 0:
            candidatos_disponiveis = [p for p in tipo3 if dia not in p["rest"] and p["mat"] not in usados and p["saldo"] > 0]
            if candidatos_disponiveis:
                professor = random.choice(candidatos_disponiveis)
                alocacao[dia] = professor["mat"]
                usados.add(professor["mat"])
                reduzir_saldo(professor, 3.33)

    return alocacao

# Aplica o agendamento na matriz
def gerar_matriz_agendada(matriz_ajustada, alocacao):
    matriz_agendada = []
    for i in range(5):
        if all(matriz_ajustada[i] == 0):
            linha = [0]*4
        else:
            linha = [alocacao[i]]*4 if alocacao[i] != 0 else [0]*4
        matriz_agendada.append(linha)
    return np.array(matriz_agendada)

# Executa o agendamento completo
def agendar_semana(dias_nao_letivos=[]):
    M_init = criar_matriz_inicial()
    M_ajus = ajustar_matriz_para_feriados(M_init, dias_nao_letivos)
    alocacao = selecionar_professores_para_dias(M_ajus, professores)
    M_agen = gerar_matriz_agendada(M_ajus, alocacao)
    M_cal = M_agen.T  # transposta
    return M_agen, M_cal, professores

# Exemplo de execução
M_agen, M_cal, professores_atualizados = agendar_semana(dias_nao_letivos=[0, 4])

print("Matriz Agendada:")
print(M_agen)

print("\nMatriz Calendário (Transposta):")
print(M_cal)

print("\nSaldo dos Professores:")
for p in professores_atualizados:
    print(f"{p['id']} - {p['mat']} - Saldo: {p['saldo']:.2f} hrs")
