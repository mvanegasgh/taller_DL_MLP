Taller de Deep Learning: MLP sobre MNIST

Proyecto de la asignatura Inteligencia Artificial (UNINTEP, sede Roldanillo), orientado al resultado de aprendizaje RA2: diseñar e implementar redes neuronales artificiales y analizar cómo los hiperparámetros afectan su desempeño.

Se entrena un perceptrón multicapa (MLP) con TensorFlow / Keras para clasificar dígitos escritos a mano del dataset MNIST, y se mide el efecto de cambiar la profundidad, el ancho y la tasa de aprendizaje, un hiperparámetro a la vez.

Archivos
Archivo	Qué hace
mlp_unintep_lab.py	Modelo base (128 → 64). Entrena, evalúa y muestra curvas de aprendizaje, predicciones, errores y los pesos de la primera capa oculta.
experimento_1.py	Entrena los cinco modelos de los experimentos con una misma función, imprime la tabla comparativa y guarda las figuras.
comparacion_perdida.png	Pérdida de entrenamiento y de validación: base, Experimento 1 y Experimento 2.
comparacion_lr.png	Efecto de la tasa de aprendizaje (escala logarítmica).
comparacion_precision.png	Precisión en test de los cinco modelos.
exp1_perdida.png	Curva de pérdida del Experimento 1 (profundidad), ejecutado de forma independiente.
pyproject.toml, uv.lock	Dependencias del proyecto (gestionadas con uv).
Cómo ejecutarlo

Requiere uv instalado.

bash
git clone https://github.com/mvanegasgh/taller_DL_MLP.git
cd taller_DL_MLP
uv sync
uv run experimento_1.py

Para ejecutar solo el modelo base con sus visualizaciones:

bash
uv run mlp_unintep_lab.py

El script de experimentos entrena cinco modelos y tarda unos minutos en CPU. Las gráficas se abren en ventanas; ciérralas para que el script continúe. También se guardan como archivos .png en la carpeta del proyecto.

Modelo base
Entrada: imágenes de 28 × 28 píxeles, normalizadas al rango [0, 1] y aplanadas a 784 valores.
Capas ocultas: 128 neuronas y 64 neuronas, ambas con ReLU.
Salida: 10 neuronas con Softmax (una por dígito).
Parámetros: 109.386.
Entrenamiento: Adam (η = 0,001), entropía cruzada categórica esparsa, 10 épocas, lote de 64, 20 % de los datos de entrenamiento para validación y semilla 42.
Experimentos

Cada experimento cambia una sola cosa respecto al modelo base.

Modelo	Cambio	Parámetros	Pérdida val.	Precisión test	Diagnóstico
Base	Ninguno	109.386	0,1354	97,18 %	Sobreajuste leve
Exp 1	Capas 128 → 128 → 128 → 64	142.410	0,1246	97,31 %	Sin diferencia clara con el base
Exp 2	Capas 16 → 8	12.786	0,1869	94,57 %	Subajuste
Exp 3A	η = 0,5	109.386	2,3357	11,37 %	Colapso al nivel del azar
Exp 3B	η = 0,00001	109.386	0,3426	90,99 %	Convergencia lenta
Conclusiones
Profundidad: pasar de 2 a 4 capas ocultas aumentó los parámetros un 30 % y el tiempo de entrenamiento alrededor de un 15 %, sin una mejora ni un sobreajuste claros en 10 épocas.
Capacidad: con 16 y 8 neuronas hay subajuste: el error es alto ya en entrenamiento y la brecha con validación es pequeña.
Tasa de aprendizaje: η = 0,5 colapsó y η = 0,00001 convergió demasiado despacio. Solo η = 0,001 llegó a una buena precisión en 10 épocas.
Limitaciones
Cada configuración se entrenó una sola vez con una única semilla, así que las diferencias pequeñas (como 0,13 puntos entre el base y el Experimento 1) no son concluyentes.
Los tiempos de entrenamiento varían entre ejecuciones; solo la comparación entre arquitecturas muy distintas es informativa.
Todos los modelos usan 10 épocas; en el Experimento 2 y con η = 0,00001 la pérdida seguía bajando.
Autor

Milton Vanegas D. · UNINTEP,  Roldanillo
