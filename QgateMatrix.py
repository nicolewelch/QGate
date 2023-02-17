import numpy as np
import math
import cmath
pi = math.pi


# universal gate
#       | cos(θ/2) (-e^iλ)sin(θ/2) |
#       | (e^iφ)sin(θ/2) (e^i(λ+φ))cos(θ/2) |
def gU3(theta, phi, lambd):
    return np.array([[math.cos(theta/2), math.sin(theta/2) * -cmath.exp(1j * lambd)], 
                     [math.sin(theta/2) * cmath.exp(1j * phi), math.cos(theta/2) * cmath.exp(1j * (phi + lambd))]])

# ------------------------------

# NOT gate:  | 0 1 |
#            | 1 0 |
gNOT = np.array([[0, 1], 
                [1, 0]])

# Y gate:    | 0 -i |
#            | i 0  |
gY = np.array([[0, -1j], 
               [1j, 0]])

# ------------------------------

# Z gate:    | 1 0 |
#            | 0 -1 |
gZ = np.array([[1, 0], [0, -1]])

# S gate:    | 1 0 |
#            | 0 i |
gS = np.array([[1, 0], [0, 1j]])

# S dagger:    | 1 0 |
#            | 0 -i |
gSdagg = np.array([[1, 0], [0, -1j]])

# T gate:    | 1 0 |
#            | 0 e^i(pi/4) |
gT = np.array([[1, 0], [0, cmath.exp(1j * (pi/4))]])

# T dagger:    | 1 0 |
#            | 0 e^-i(pi/4) |
gTdagg = np.array([[1, 0], [0, cmath.exp(-1j * (pi/4))]])

# Rz gate:   | 1 0 |
#            | 0 e^iθ |
def gRz(theta):
    return np.array([[1, 0], [0, cmath.exp(1j * theta)]])

# ------------------------------

# H gate:    1/sqrt(2)| 1 1 |
#                     | 1 -1 |
gH = (1/math.sqrt(2)) * np.array([[1, 1], [1, -1]])

# Rx gate:   | cos(θ/2) -isin(θ/2) |
#            | -isin(θ/2) cos(θ/2) |
def gRx(theta):
    return np.array([[math.cos(theta/2), -1j * math.sin(theta/2)],
                     [-1j * math.sin(theta/2), math.cos(theta/2)]])

# Ry gate:   | cos(θ/2) -sin(θ/2) |
#            | -isin(θ/2) cos(θ/2) |
def gRy(theta):
    return np.array([[math.cos(theta/2), -math.sin(theta/2)], 
                     [-1j * math.sin(theta/2), math.cos(theta/2)]])

# ID gate:   | 1 0 |
#            | 0 1 |
gID = np.array([[1, 0], [0, 1]])

def getProbs(matrix):

    zero = matrix[0][0]
    one = matrix[1][0]

    zero = zero * np.conj(zero)
    one = one * np.conj(one)

    return((zero, one))


#-------------------------------------
# init qubit as idealized |0>
Qbit = np.array([[1], [0]])
mappings = {
    'H': gH,
    'X': gNOT,
    'Y': gY,
    'Z': gZ,
    'S': gS,
    'S+': gSdagg,
    'T': gT,
    'T+': gTdagg
}

thetaMappings = {
    'Rz': gRz,
    'Rx': gRx,
    'Ry': gRy
}

gates = []

print("----------------------------------------------------------")
print("Your qubit has been initialized to |0>")
print("Type the name of the gate to apply to the qubit, then hit enter")
print("Enter all angles in the form 3*pi/2 or pi/2")
print("Available gates: H, X, Y, Z, S, S+, T, T+, Rz, Rx, Ry, U3")
print("When you are finished adding gates, type 'end'")
print("----------------------------------------------------------")

while True:
    gate = input("Enter gate: ")

    if gate == 'end':
        break
    elif gate in mappings:
        gates.append(gate)
    elif gate in thetaMappings:
        theta = input('theta? ')
        gates.append((gate, theta))
    elif gate == "U3":
        theta = input('theta? ')
        phi = input('phi? ')
        lambd = input('lambda? ')
        gates.append((gate, theta, phi, lambd))
    else:
        print('invalid gate input')

print("Processing...")

if gates[-1] in mappings:
    psi = mappings[gates[-1]]
elif gates[-1][0] in thetaMappings:
    psi = thetaMappings[gates[-1][0]](eval(gates[-1][1]))
else:
    psi = gU3(eval(gates[-1][1]), eval(gates[-1][2]), eval(gates[-1][3]))

if len(gates) > 1:
    for i in range(len(gates)-2, -1, -1): 
        if gates[i] in mappings:
            psi = psi @ mappings[gates[i]]
        elif gates[i][0] in thetaMappings:
            psi = psi @ thetaMappings[gates[i][0]](eval(gates[i][1]))
        else:
            psi = psi @ gU3(eval(gates[i][1]), eval(gates[i][2]), eval(gates[i][3]))

psi = psi @ Qbit

print('Amplitudes: ')
print("( " + str(psi[0]) + " )")
print("( " + str(psi[1]) + " )")

probs = getProbs(psi)
print('Probabilities: ')
print('|0> : ' + str(probs[0]))
print('|1> : ' + str(probs[1]))