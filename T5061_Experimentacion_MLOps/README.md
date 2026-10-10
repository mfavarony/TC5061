# TC5061 — Experimentación con MLflow

## Instalar dependencias
```bash
python3 -m pip install -r requirements.txt
```

## Ejecutar cinco corridas
Ejecuta estos comandos por separado desde esta carpeta y usando el mismo entorno Python:
```bash
python3 train.py --c 0.1 --max-iter 500 --class-weight none --random-state 42
python3 train.py --c 0.5 --max-iter 1000 --class-weight none --random-state 42
python3 train.py --c 1.0 --max-iter 1000 --class-weight none --random-state 42
python3 train.py --c 2.0 --max-iter 1500 --class-weight none --random-state 42
python3 train.py --c 1.0 --max-iter 1000 --class-weight balanced --random-state 42
```
Todas se agrupan en `TC5061_Titanic_LogisticRegression_Comparacion` y registran parámetros, métricas, matriz de confusión, JSON de métricas y modelo.

## Abrir MLflow
En otra terminal, desde la misma carpeta:
```bash
python3 -m mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
```
Abre `http://localhost:5001`, entra al experimento y selecciona los cinco runs para compararlos.

### Reproducibilidad y dependencias

Para favorecer la reproducibilidad, el script `train.py` permite configurar la semilla aleatoria mediante el argumento `--random-state`, cuyo valor predeterminado es 42. Esta semilla se utiliza tanto en la división estratificada de los datos de entrenamiento y prueba como en la configuración del modelo. De esta manera, al mantener los mismos datos, argumentos y versiones de las dependencias, las ejecuciones deberían producir las mismas métricas.

Para comprobar la reproducibilidad, se ejecuta dos veces el mismo comando:

python3 train.py --c 1.0 --max-iter 1000 --class-weight none --random-state 42

Se comparan las métricas accuracy, precision, recall y f1_score de ambas ejecuciones en MLflow. Los resultados deben coincidir, aunque cada ejecución tendrá un identificador de run diferente.

Las dependencias del proyecto se documentan en el archivo `requirements.txt`, generado a partir del entorno de Python utilizado. Este archivo permite identificar las versiones de las librerías y facilita la instalación del entorno en otro equipo. La reproducibilidad también requiere utilizar el mismo conjunto de datos y una configuración equivalente de Python y del sistema.

Para generar el archivo a partir del entorno actual, se utiliza:

python3 -m pip freeze > requirements.txt

Para instalar las dependencias en otro entorno de Python:

python3 -m pip install -r requirements.txt

Para garantizar resultados reproducibles, se recomienda utilizar las mismas versiones de Python y de las librerías, así como el mismo archivo de datos y los mismos argumentos de ejecución.

## Pruebas adicionales de sensibilidad del hiperparámetro C

Para analizar cómo influye la regularización en el desempeño de la Regresión Logística, se realizaron dos experimentos adicionales con valores extremos de C, manteniendo constantes los demás parámetros y la semilla aleatoria.

Prueba 1: regularización alta

python3 train.py --c 0.01 --max-iter 1000 --class-weight none --random-state 42

Resultados obtenidos:

Accuracy: 0.7821

Precision: 0.9167

Recall: 0.4783

F1-score: 0.6286

Prueba 2: regularización baja

python3 train.py --c 100 --max-iter 1000 --class-weight none --random-state 42

Resultados obtenidos:

Accuracy: 0.8101

Precision: 0.7966

Recall: 0.6812

F1-score: 0.7344

En esta comparación, el valor C = 100 obtuvo mayor accuracy, recall y F1-score, mientras que C = 0.01 alcanzó mayor precision. Esto muestra que la regularización influye en el equilibrio entre la identificación correcta de sobrevivientes y los falsos positivos. Los resultados corresponden a una sola partición de entrenamiento y prueba, por lo que no permiten concluir por sí solos que C = 100 sea la mejor configuración general.

