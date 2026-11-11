# Proyecto de Minería de Datos: predicción académica de diabetes

## Descripción
Aplicación académica que integra un modelo de clasificación entrenado con el conjunto Pima Indians Diabetes Database. Permite ingresar ocho variables, consultar la clase estimada y visualizar métricas y gráficas exploratorias.

## Objetivo
Integrar el pipeline de preprocesamiento y clasificación desarrollado en el notebook con una interfaz web en Streamlit, conservando una evaluación reproducible.

## Datos
Fuente: Pima Indians Diabetes Database, archivo CSV público:
https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

El notebook descarga los datos si `diabetes.csv` no está disponible localmente. La variable objetivo es `Outcome` (0 = sin diabetes según el conjunto, 1 = con diabetes según el conjunto).

## Integrantes
Completar con los nombres de todos los integrantes del grupo.

## Archivos
- `app.py`: aplicación Streamlit.
- `requirements.txt`: dependencias.
- `Proyecto mineria_actualizado.ipynb`: notebook con la sección para validación cruzada y generación de artefactos.
- `modelo_pipeline.joblib`: se genera al ejecutar el notebook.
- `metricas_modelo.json`: métricas, matriz de confusión y resultados de validación cruzada.
- `diabetes_limpio.csv`: datos limpios usados en las visualizaciones.
- `registro_prueba.csv` y `resultado_registro_prueba.json`: caso de prueba para comprobar consistencia.

## Cómo ejecutar
1. Abrir `Proyecto mineria_actualizado.ipynb` en Google Colab.
2. Ejecutar todas las celdas desde el inicio hasta el final.
3. Descargar los archivos generados (`modelo_pipeline.joblib`, `metricas_modelo.json`, `diabetes_limpio.csv`, `registro_prueba.csv` y `resultado_registro_prueba.json`) y agregarlos a la raíz del repositorio.
4. Instalar dependencias: `pip install -r requirements.txt`.
5. Ejecutar localmente: `streamlit run app.py`.

## Publicación
Crear un repositorio público en GitHub con todos los archivos anteriores. En Streamlit Community Cloud, seleccionar el repositorio, la rama y `app.py` como archivo principal. Verificar que la URL pública abra correctamente.

## Pruebas requeridas
- Caso normal: introducir valores dentro de los rangos permitidos y ejecutar una predicción.
- Entrada inválida: intentar valores fuera de los límites del formulario; Streamlit debe impedirlos. La aplicación también valida valores críticos positivos.
- Registro conocido: usar la sección de comprobación dentro de la aplicación. La predicción debe coincidir con la guardada desde el notebook.

## Evaluación del modelo
El notebook compara una línea base, regresión logística y árbol de decisión. La selección final se realiza usando el F1 promedio de validación cruzada, no solo accuracy. La aplicación presenta las métricas del conjunto de prueba y la matriz de confusión. Los valores finales se generan al ejecutar el notebook actualizado; no se deben inventar ni copiar métricas de otra ejecución.

## Uso de inteligencia artificial
- Herramienta: ChatGPT.
- Propósito: apoyar la organización del proyecto, la construcción de la interfaz Streamlit, la documentación y la revisión del código.
- Prompt principal: “Ayúdame a integrar en Streamlit el pipeline de clasificación de diabetes de mi notebook, conservando las variables, el preprocesamiento y las métricas, y prepara los archivos necesarios para GitHub”.
- Verificación: ejecutar el notebook de principio a fin, revisar que se generen los archivos, iniciar la aplicación y comparar la predicción de un registro de prueba con la salida del notebook.
- Problemas encontrados: el modelo y los archivos serializados deben generarse antes de iniciar Streamlit; si faltan, la aplicación muestra un mensaje explicativo.
- Ajustes realizados: carga del pipeline guardado en lugar de entrenar el modelo al abrir la aplicación; validación de entradas; visualizaciones y sección de métricas; comprobación de un registro conocido.

## Limitaciones y advertencia
Este proyecto es un prototipo académico. No es una herramienta médica, no diagnostica diabetes y no sustituye la valoración de un profesional de salud. El conjunto de datos es limitado y sus resultados no deben generalizarse a toda la población.
