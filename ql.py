#!/usr/bin/env python3
import random
import time
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_D
from ev3dev2.sensor.lego import ColorSensor
from ev3dev2.sensor import INPUT_1, INPUT_2, INPUT_3

# --- Conexiones y configuración ---
# Nota: Usamos las clases directas para evitar ambigüedades si 'auto' falla
motor_der = LargeMotor(OUTPUT_A)
motor_izq = LargeMotor(OUTPUT_D)

ojo_der = ColorSensor(INPUT_1) 
ojo_med = ColorSensor(INPUT_2)
ojo_izq = ColorSensor(INPUT_3)

# Configurar modo REFLECTION
ojo_izq.mode = 'COL-REFLECT'
ojo_der.mode = 'COL-REFLECT'
ojo_med.mode = 'COL-REFLECT'

# --- Constantes ---
VER_NEGRO = 20   
VER_LIMITE = 40  
VEL_ALTA = -40    
VEL_MEDIA = -20
VEL_BAJA = -10

# --- Definición del Entorno ---
N_ESTADOS = 4
N_ACCIONES = 3

# Inicialización de la Q-Table sin Numpy
# Creamos una lista de 4 listas, cada una con 3 ceros [0.0, 0.0, 0.0]
q_table = [[0.0] * N_ACCIONES for _ in range(N_ESTADOS)]

# Hiperparámetros
alpha = 0.1     
gamma = 0.9     
epsilon_start = 1.0 
epsilon_min = 0.1   
decay = 0.995       

def obtener_estado():
    """Lee los 3 sensores y determina en qué estado discreto estamos"""
    val_i = ojo_izq.value()
    val_m = ojo_med.value()
    val_d = ojo_der.value()
    
    # Lógica de prioridades
    if val_m <= VER_NEGRO:
        return 1 # Estado: Centrado
    elif val_i <= VER_LIMITE:
        return 0 # Estado: Viendo izq
    elif val_d <= VER_LIMITE:
        return 2 # Estado: Viendo der
    else:
        return 3 # Estado: Perdido (blanco)

def avanza():
    motor_izq.on(VEL_ALTA)
    motor_der.on(VEL_ALTA)

def giraizq():
    # Pivote sobre una rueda o giro inverso
    motor_izq.on(VEL_MEDIA)
    motor_der.on(-VEL_ALTA)

def girader():
    motor_izq.on(-VEL_ALTA)
    motor_der.on(VEL_MEDIA)

def ev3action(accion):
    if accion == 0:
        avanza()
    elif accion == 1:
        giraizq()
    elif accion == 2:
        girader()

# --- FUNCIONES AUXILIARES (Reemplazo de Numpy) ---
def argmax(lista):
    """Devuelve el índice del valor más alto en una lista"""
    # Encuentra el valor máximo
    max_val = max(lista)
    # Devuelve el índice de ese valor
    return lista.index(max_val)

# --- BUCLE PRINCIPAL DE ENTRENAMIENTO ---
epsilon = epsilon_start
rondas = 100
s = obtener_estado()

print("Iniciando Q-Learning (Pure Python)...")

try:
    for x in range(rondas):
        # 2. Elegir Acción (Epsilon-Greedy)
        if random.random() < epsilon:
            a = random.randint(0, N_ACCIONES - 1) # Exploración
        else:
            # Explotación: Usamos nuestra función auxiliar en lugar de np.argmax
            # q_table[s] es la fila actual de valores para el estado s
            a = argmax(q_table[s]) 

        # 3. Ejecutar Acción
        ev3action(a)       
        
        # Pausa breve para dejar actuar a la física
        time.sleep(0.1) 
        
        # 4. Observar Nuevo Estado (S')
        sp = obtener_estado()
        
        # 5. Calcular Recompensa (R)
        r = 0
        if sp == 1:      # Centro
            r = 10
        elif sp == 0 or sp == 2: # Bordes
            r = -3
        elif sp == 3:    # Perdido
            r = -20
        
        # 6. Actualizar Q-Table (Ecuación de Bellman)
        # Acceso directo a listas: lista[fila][columna]
        q_old = q_table[s][a]
        
        # Reemplazo de np.max: usamos max() nativo de Python
        max_q_next = max(q_table[sp])
        
        # Cálculo aritmético estándar
        new_value = q_old + alpha * (r + gamma * max_q_next - q_old)
        q_table[s][a] = new_value
        
        # 7. Avanzar estado
        s = sp    
        
        # Decaer epsilon
        if epsilon > epsilon_min:
            epsilon *= decay
            
        # Debug opcional cada 10 pasos
        if x % 10 == 0:
            print("Paso: {}, Eps: {:.2f}, Estado: {}".format(x, epsilon, s))

except KeyboardInterrupt:
    print("Detenido por usuario.")

finally:
    motor_izq.off()
    motor_der.off()
    print("\n--- Tabla Q Final ---")
    for i, row in enumerate(q_table):
        print("Estado {}: {}".format(i, [round(val, 2) for val in row]))