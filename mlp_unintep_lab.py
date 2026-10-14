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
