import os
import logging
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# DATOS (se cargan una sola vez para todos los modelos)
# ------------------------------------------------------------------
(X_train_raw, y_train), (X_test_raw, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train_raw / 255.0
X_test = X_test_raw / 255.0

resultados = {}   # aquí se guarda cada modelo para compararlos


# ------------------------------------------------------------------
# FUNCIÓN QUE CONSTRUYE, ENTRENA Y GUARDA UN MODELO
#   capas_ocultas: neuronas de cada capa oculta, ej. (128, 64)
#   lr: tasa de aprendizaje
# ------------------------------------------------------------------
def entrenar(nombre, capas_ocultas=(128, 64), lr=0.001, epocas=10):
    print(f"\n===== {nombre} =====")
    tf.keras.utils.set_random_seed(42)   # misma semilla = comparación justa

    modelo = models.Sequential()
    modelo.add(layers.Input(shape=(28, 28)))
    modelo.add(layers.Flatten())
    for neuronas in capas_ocultas:
        modelo.add(layers.Dense(neuronas, activation='relu'))
    modelo.add(layers.Dense(10, activation='softmax'))

    modelo.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    inicio = time.time()
    h = modelo.fit(X_train, y_train, epochs=epocas, batch_size=64,
                   validation_split=0.2, verbose=0)
    segundos = time.time() - inicio

    loss_test, acc_test = modelo.evaluate(X_test, y_test, verbose=0)

    resultados[nombre] = {
        "historial": h.history,
        "segundos": segundos,
        "parametros": modelo.count_params(),
        "loss_test": loss_test,
        "acc_test": acc_test,
    }
    print(f"Parámetros: {modelo.count_params():,}")
    print(f"Tiempo: {segundos:.1f} s")
    print(f"Loss entrenamiento (época {epocas}): {h.history['loss'][-1]:.4f}")
    print(f"Loss validación (época {epocas}):    {h.history['val_loss'][-1]:.4f}")
    print(f"Loss test: {loss_test:.4f} | Accuracy test: {acc_test*100:.2f}%")


# ------------------------------------------------------------------
# EXPERIMENTOS
# ------------------------------------------------------------------
entrenar("Base (128-64)", capas_ocultas=(128, 64))
entrenar("Exp 1 (128-128-128-64)", capas_ocultas=(128, 128, 128, 64))
entrenar("Exp 2 (16-8)", capas_ocultas=(16, 8))
entrenar("Exp 3A (lr=0.5)", lr=0.5)
entrenar("Exp 3B (lr=0.00001)", lr=0.00001)


# ------------------------------------------------------------------
# TABLA COMPARATIVA
# ------------------------------------------------------------------
print("\n" + "=" * 90)
print(f"{'Modelo':<26}{'Parámetros':>12}{'Tiempo (s)':>12}{'Loss train':>12}{'Loss val':>10}{'Acc test':>10}")
print("=" * 90)
for nombre, r in resultados.items():
    print(f"{nombre:<26}{r['parametros']:>12,}{r['segundos']:>12.1f}"
          f"{r['historial']['loss'][-1]:>12.4f}"
          f"{r['historial']['val_loss'][-1]:>10.4f}{r['acc_test']*100:>9.2f}%")
print("=" * 90)

# Pérdida de entrenamiento época por época del caso lr = 0.5
# (sirve para ver si hubo divergencia inicial o solo estancamiento)
print("\nLoss de entrenamiento por época, Exp 3A (lr=0.5):")
print([round(x, 3) for x in resultados["Exp 3A (lr=0.5)"]["historial"]["loss"]])


# ------------------------------------------------------------------
# GRÁFICA 1: arquitecturas (base, profundidad y capacidad)
# ------------------------------------------------------------------
arquitecturas = ["Base (128-64)", "Exp 1 (128-128-128-64)", "Exp 2 (16-8)"]

fig, ejes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for nombre in arquitecturas:
    r = resultados[nombre]
    x = range(1, len(r['historial']['loss']) + 1)
    ejes[0].plot(x, r['historial']['loss'], marker='o', label=nombre)
    ejes[1].plot(x, r['historial']['val_loss'], marker='s', label=nombre)

ejes[0].set_title('Pérdida de ENTRENAMIENTO vs épocas')
ejes[1].set_title('Pérdida de VALIDACIÓN vs épocas')
for ax in ejes:
    ax.set_xlabel('Época')
    ax.grid(True)
    ax.legend()
ejes[0].set_ylabel('Pérdida (L)')
plt.tight_layout()
plt.savefig('comparacion_perdida.png', dpi=150)


# ------------------------------------------------------------------
# GRÁFICA 2: efecto de la tasa de aprendizaje (escala logarítmica)
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5))
for nombre in ["Base (128-64)", "Exp 3A (lr=0.5)", "Exp 3B (lr=0.00001)"]:
    r = resultados[nombre]
    plt.plot(range(1, 11), r['historial']['loss'], marker='o', label=nombre)
plt.yscale('log')
plt.title('Efecto de la tasa de aprendizaje: pérdida de entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida (escala logarítmica)')
plt.grid(True, which='both')
plt.legend()
plt.savefig('comparacion_lr.png', dpi=150)


# ------------------------------------------------------------------
# GRÁFICA 3: precisión en test de todos los modelos
# ------------------------------------------------------------------
plt.figure(figsize=(9, 5))
nombres = list(resultados.keys())
accs = [resultados[n]['acc_test'] * 100 for n in nombres]
barras = plt.bar(nombres, accs)
plt.bar_label(barras, fmt='%.2f%%')
plt.ylabel('Precisión en test (%)')
plt.title('Precisión en test por experimento')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('comparacion_precision.png', dpi=150)

print("\nGráficas guardadas: comparacion_perdida.png, comparacion_lr.png, comparacion_precision.png")
plt.show()