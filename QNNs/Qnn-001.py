# QNN 001

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Número de qubits (igual ao número de variáveis de entrada)
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

# Dados de entrada (features) e classes (labels)
X = np.array([[0.1, 0.2], [0.4, 0.4], [0.8, 0.8], [1.0, 0.9]])
Y = np.array([0, 0, 1, 1])  # Classes: 0 ou 1

# Codificação dos dados nos qubits (encoding layer)
def encode(x):
    qml.RY(x[0], wires=0)
    qml.RY(x[1], wires=1)

# Circuito quântico parametrizado (QNN)
@qml.qnode(dev)
def circuit(weights, x):
    encode(x)  # Codifica os dados de entrada
    qml.CNOT(wires=[0, 1])  # Entrelaçamento entre os qubits
    qml.RY(weights[0], wires=0)  # Parâmetros treináveis
    qml.RY(weights[1], wires=1)
    return qml.expval(qml.PauliZ(0))  # Mede a expectativa de Z no qubit 0

# Função de previsão
def predict(weights, x):
    return [circuit(weights, xi) for xi in x]

# Função de custo (erro quadrático médio)
def cost(weights, x, y):
    preds = predict(weights, x)
    return np.mean((preds - y) ** 2)

# Inicializa os pesos (parâmetros da rede)
weights = np.random.uniform(low=0, high=2*np.pi, size=(2,))

# Otimizador clássico (gradiente descendente)
opt = qml.GradientDescentOptimizer(stepsize=0.4)

# Lista para salvar os valores de custo
costs = []

# Treinamento da rede por 30 iterações
for i in range(30):
    weights = opt.step(lambda w: cost(w, X, Y), weights)
    current_cost = cost(weights, X, Y)
    costs.append(current_cost)
    if i % 5 == 0:
        print(f"Iteração {i}: custo = {current_cost:.4f}")

# Resultados
print("\nPesos finais treinados:", weights)
print("Previsões finais:", predict(weights, X))
print("Labels verdadeiros:", Y)

# Plot do custo ao longo das iterações
plt.plot(costs, marker='o')
plt.title("Evolução do custo durante o treinamento")
plt.xlabel("Iterações")
plt.ylabel("Custo")
plt.grid(True)
plt.show()
