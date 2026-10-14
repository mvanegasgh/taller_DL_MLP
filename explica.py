import os
import logging

# ==============================================================================
# CAMBIO 1: Supresión de logs del motor C++ de TensorFlow y GPU
# ==============================================================================
# En el original: No existían estas variables de entorno.
# Motivo: Evita avisos de optimizaciones de CPU (oneDNN/AVX) y la advertencia de GPU en Windows.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Cambiado de '2' a '3' para silenciar WARNINGs de C++
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Oculta dispositivos GPU para evitar el aviso de Windows

import tensorflow as tf

# ==============================================================================
# CAMBIO 2: Supresión del logger interno de Python de TensorFlow
# ==============================================================================
# En el original: No se reconfiguraba tf.get_logger().
# Motivo: Captura y silencia las alertas emitidas por tf_logging antes de imprimir en la consola.
tf.get_logger().setLevel(logging.ERROR)

from tensorflow.keras import layers, models, optimizers
import numpy as np

# Configuración de semillas para reproducibilidad (Igual al original)
tf.random.set_seed(42)
np.random.seed(42)

print(f"Versión de TensorFlow en uso: {tf.__version__}")

# Carga y preparación del conjunto de datos (Igual al original)
mnist = tf.keras.datasets.mnist
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

X_train = X_train_raw / 255.0
X_test = X_test_raw / 255.0

# ==============================================================================
# CAMBIO 3: Actualización de la sintaxis de la Capa de Entrada en Keras 3
# ==============================================================================
def construir_mlp():
    model = models.Sequential([
        # En el original: layers.Flatten(input_shape=(28, 28), name="Capa_Entrada_Flatten")
        # Motivo: Keras 3 depreca input_shape dentro de Flatten y requiere un Input explícito.
        layers.Input(shape=(28, 28), name="Capa_Entrada_Input"),
        layers.Flatten(name="Capa_Aplanado"),
        
        # Estructura idéntica al modelo original
        layers.Dense(128, activation='relu', name="Capa_Oculta_1"),
        layers.Dense(64, activation='relu', name="Capa_Oculta_2"),
        layers.Dense(10, activation='softmax', name="Capa_Salida_Softmax")
    ], name="MLP_UNINTEP")
    return model

modelo_mlp = construir_mlp()
modelo_mlp.summary()

# Compilación y entrenamiento (Igual al original)
optimizador_adam = optimizers.Adam(learning_rate=0.001)

modelo_mlp.compile(
    optimizer=optimizador_adam,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nIniciando entrenamiento del modelo base...")
historial = modelo_mlp.fit(
    X_train, y_train_raw,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# Evaluación del modelo (Igual al original)
loss_test, accuracy_test = modelo_mlp.evaluate(X_test, y_test_raw, verbose=0)
print(f"\n==========================================")
print(f"Pérdida en Test (Loss): {loss_test:.4f}")
print(f"Precisión en Test (Accuracy): {accuracy_test * 100:.2f}%")
print(f"==========================================")

# ==============================================================================
# VER POR DENTRO LA RED NEURONAL
# Pega este bloque AL FINAL de tu script (después de modelo_mlp.evaluate).
# Requiere matplotlib:  pip install matplotlib
# ==============================================================================
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1) CÓMO APRENDIÓ: curvas de precisión y error por época
# ------------------------------------------------------------------------------
fig, ejes = plt.subplots(1, 2, figsize=(11, 4))

ejes[0].plot(historial.history['accuracy'], label='Entrenamiento')
ejes[0].plot(historial.history['val_accuracy'], label='Validación')
ejes[0].set_title('Precisión por época (sube = aprende)')
ejes[0].set_xlabel('Época')
ejes[0].set_ylabel('Accuracy')
ejes[0].legend()

ejes[1].plot(historial.history['loss'], label='Entrenamiento')
ejes[1].plot(historial.history['val_loss'], label='Validación')
ejes[1].set_title('Error por época (baja = aprende)')
ejes[1].set_xlabel('Época')
ejes[1].set_ylabel('Loss')
ejes[1].legend()

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 2) QUÉ RESPONDE: la red adivina 10 imágenes de test que nunca vio
# ------------------------------------------------------------------------------
probabilidades = modelo_mlp.predict(X_test[:10], verbose=0)  # forma (10, 10)

fig, ejes = plt.subplots(2, 5, figsize=(11, 5))
for i, ax in enumerate(ejes.flat):
    prediccion = np.argmax(probabilidades[i])        # dígito con mayor probabilidad
    confianza = probabilidades[i][prediccion] * 100
    real = y_test_raw[i]
    ax.imshow(X_test_raw[i], cmap='gray_r')
    color = 'green' if prediccion == real else 'red'
    ax.set_title(f"Predice {prediccion} ({confianza:.0f}%)\nReal: {real}", color=color)
    ax.axis('off')
plt.suptitle('Verde = acertó, Rojo = se equivocó')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 3) EN QUÉ SE EQUIVOCA: primeras 10 imágenes que falló
# ------------------------------------------------------------------------------
todas_las_predicciones = np.argmax(modelo_mlp.predict(X_test, verbose=0), axis=1)
indices_error = np.where(todas_las_predicciones != y_test_raw)[0]
print(f"\nLa red falló en {len(indices_error)} de {len(y_test_raw)} imágenes de test.")

fig, ejes = plt.subplots(2, 5, figsize=(11, 5))
for ax, idx in zip(ejes.flat, indices_error[:10]):
    ax.imshow(X_test_raw[idx], cmap='gray_r')
    ax.set_title(f"Predijo {todas_las_predicciones[idx]}\nReal: {y_test_raw[idx]}", color='red')
    ax.axis('off')
plt.suptitle('Errores de la red (a veces hasta a un humano le cuesta leerlos)')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 4) POR DENTRO: qué "mira" cada neurona de la primera capa oculta
#    Cada neurona tiene 784 pesos (uno por píxel). Los reorganizamos en 28x28.
#    Zonas claras/oscuras = píxeles que esa neurona considera importantes.
# ------------------------------------------------------------------------------
pesos, sesgos = modelo_mlp.get_layer("Capa_Oculta_1").get_weights()   # pesos: (784, 128)

fig, ejes = plt.subplots(4, 8, figsize=(11, 6))
for i, ax in enumerate(ejes.flat):
    ax.imshow(pesos[:, i].reshape(28, 28), cmap='seismic')
    ax.axis('off')
plt.suptitle('32 de las 128 neuronas de la capa oculta 1: cada una busca un patrón distinto')
plt.tight_layout()
plt.show()