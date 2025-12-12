#!/bin/python3
import numpy as np
import random
from ev3dev2.auto import * 
from time import sleep
# --- Conexiones y configuración ---
motor_der = LargeMotor(OUTPUT_A)
motor_izq = LargeMotor(OUTPUT_D)
ojo_der = ColorSensor(INPUT_1) 
ojo_med = ColorSensor(INPUT_2)
ojo_izq = ColorSensor(INPUT_3)
ojo_izq.mode = 'COL-REFLECT'
ojo_der.mode = 'COL-REFLECT'
ojo_med.mode = 'COL-REFLECT'
# --- Constantes ---
# Ajuste de umbrales: Para detectar NEGRO, el valor debe ser MENOR al umbral
VER_NEGRO = 20   # Umbral estricto para la línea
VER_LIMITE = 40  # Umbral para detectar si apenas tocamos la línea
VEL_ALTA = 40    
VEL_MEDIA = 20
VEL_BAJA = 10
# --- Definición del Entorno ---
N_ESTADOS = 4
N_ACCIONES = 3
# Mapeo de Estados:
# 0: Izquierda detecta linea (Desviado a la derecha)
# 1: Centro detecta linea (Correcto)
# 2: Derecha detecta linea (Desviado a la izquierda)
# 3: Ninguno (Perdido / Blanco)
# Mapeo de Acciones:
# 0: Avanza
# 1: Gira Izq
# 2: Gira Der

# Tabla Q inicializada en ceros
q_table = np.zeros([N_ESTADOS, N_ACCIONES])
# Hiperparámetros
alpha = 0.1     # Tasa de aprendizaje
gamma = 0.9     # Factor de descuento
epsilon_start = 1.0 # Empezar explorando al 100%
epsilon_min = 0.1   # Mínimo de exploración
decay = 0.995       # Qué tan rápido deja de explorar

def obtener_estado():
    """Lee los 3 sensores y determina en qué estado discreto estamos"""
    val_i = ojo_izq.value()
    val_m = ojo_med.value()
    val_d = ojo_der.value()
    # Prioridad: ¿Dónde está el negro? (Valores bajos indican negro)
    if val_m <= VER_NEGRO:
        return 1 # Estado: Centrado
    elif val_i <= VER_LIMITE:
        return 0 # Estado: Viendo izq
    elif val_d <= VER_LIMITE:
        return 2 # Estado: Viendo der
    else:
        return 3 # Estado: Perdido (blanco)

def avanza():
    motor_izq.run_forever(speed_sp=VEL_ALTA)
    motor_der.run_forever(speed_sp=VEL_ALTA)
def giraizq():
    motor_izq.run_forever(speed_sp=-VEL_MEDIA)
    motor_der.run_forever(speed_sp=VEL_ALTA)
def girader():
    motor_izq.run_forever(speed_sp=VEL_ALTA)
    motor_der.run_forever(speed_sp=-VEL_MEDIA)

def ev3action(accion):
    if accion == 0:
        avanza()
    elif accion == 1:
        giraizq()
    elif accion == 2:
        girader()

# --- BUCLE PRINCIPAL DE ENTRENAMIENTO ---
epsilon = epsilon_start
rondas = 100
# 1. Leemos el estado inicial antes de empezar
s = obtener_estado()
print("Iniciando Q-Learning...")
try:
    for x in range(rondas):
        # 2. Elegir Acción (Epsilon-Greedy)
        if random.random() < epsilon:
            a = random.randint(0, 2) # Exploración (Aleatorio 0, 1, 2)
        else:
            a = np.argmax(q_table[s]) # Explotación (Mejor valor conocido)
        # 3. Ejecutar Acción
        ev3action(a)       
        # Dar tiempo al robot para moverse físicamente
        sleep(0.10) 
        # 4. Observar Nuevo Estado (S')
        sp = obtener_estado()
        # 5. Calcular Recompensa (R) basada en el NUEVO estado
        r = 0
        terminal = False # ¿Se acabó el juego?
        if sp == 1:      # Está en el centro (Ideal)
            r = 10
        elif sp == 0 or sp == 2: # Está en los bordes (Aceptable pero peligroso)
            r = -3
        elif sp == 3:    # Se salió (Castigo fuerte)
            r = -20
            # Opcional: Si se sale mucho, podrías detener el episodio aquí
            # terminal = True 
        # 6. Actualizar Q-Table (Ecuación de Bellman)
        # Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
        q_old = q_table[s, a]
        max_q_next = np.max(q_table[sp])
        
        q_table[s, a] = q_old + alpha * (r + gamma * max_q_next - q_old)
        # 7. Avanzar estado
        s = sp    
        # Decaer epsilon (explorar menos con el tiempo)
        if epsilon > epsilon_min:
            epsilon *= decay

except KeyboardInterrupt:
    print("Detenido por usuario.")
finally:
    motor_izq.stop()
    motor_der.stop()
    print("Tabla Q final:")
    print(q_table)
