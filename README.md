# Evento Evaluativo 4 · Detección de fraude en transacciones electrónicas

**Integrantes:** Isabel Johana Hoyos Echavarria · Jhon Eduardo Zabala Garzón · Sebastian Villa Castillo · Edwin Ramírez Gonzáles

**Video:** ENLACE_AL_VIDEO

## Problema
En una plataforma de pagos en línea, algunas transacciones son fraudulentas: alguien usa una tarjeta o una identidad que no es suya. Cada fraude que pasa cuesta dinero, pero bloquear una compra legítima también cuesta, porque molesta a un cliente real. El reto es detectar el fraude sin frenar las compras normales.

Lo difícil es que el fraude es muy poco frecuente (cerca de 3,5 de cada 100 transacciones): un modelo que dijera "todo es legítimo" acertaría cerca del 96 % de las veces y no serviría de nada. Por eso evaluamos con PR-AUC, recall y precision, y no con el porcentaje de aciertos.

Aplicamos dos enfoques sobre los mismos datos: **supervisado** (predecir `isFraud`) y **no supervisado** (buscar grupos y transacciones atípicas sin usar la etiqueta, para ver si coinciden con el fraude).

## Datos
- Dataset: **IEEE-CIS Fraud Detection** (Kaggle), archivos `train_transaction.csv` y `train_identity.csv` unidos por `TransactionID`.
- Descarga: Kaggle → competencia *IEEE-CIS Fraud Detection* → pestaña Data. Los CSV se colocan en `data/raw/` y no se suben al repositorio.
- Tamaño completo: 590.540 transacciones y 434 columnas tras unir; 3,50 % de fraude. Solo el 24,4 % de las transacciones tiene datos de identidad.
- Muestra usada: 150.000 transacciones elegidas al azar, porque el dataset completo es demasiado pesado para trabajar cómodamente. El muestreo conserva la proporción de fraude (verificado en el notebook 01).
- División por tiempo: 70 % más antiguo para entrenar (105.000) y 30 % más reciente para probar (45.000).

## Estructura
```
bi-ml-fraude-transacciones-entrega4/
├── notebooks/
│   ├── 01_comprension_eda_preprocesamiento.ipynb
│   ├── 02_supervisado.ipynb
│   ├── 03_no_supervisado.ipynb
│   └── 04_comparacion_conclusiones.ipynb
├── src/preprocesamiento.py      # funciones compartidas (sin fuga de datos)
├── resultados/                  # métricas en JSON generadas por los notebooks
├── data/raw/                    # CSV de Kaggle (no se suben)
└── requirements.txt
```

## Cómo ejecutarlo
1. `pip install -r requirements.txt`
2. Poner los CSV en `data/raw/`.
3. Ejecutar los notebooks **en orden** (01 → 04); el 01 guarda la muestra dividida que usan los demás.

## Decisiones técnicas clave
- División **por tiempo** (70 % antiguo para entrenar, 30 % reciente para probar) para evitar fuga de información.
- Todo el preprocesamiento se ajusta solo con entrenamiento.
- Métrica principal: **PR-AUC** (hay muy poco fraude); también precision, recall, F1, ROC-AUC y matriz de confusión.
- Umbral de decisión y selección del modelo elegidos en validación, no en prueba.
- En el notebook no supervisado, los valores estandarizados se recortan a ±5 porque algunas variables anónimas llegan a más de 300 desviaciones estándar y dominaban el agrupamiento.

## Hallazgos del análisis exploratorio
- **Desbalance:** 1 fraude por cada 28 transacciones legítimas.
- **Faltantes:** 229 de 437 columnas tienen más de 50 % de nulos, y que falten aporta información: con datos de identidad el fraude es 7,6 %, sin ellos 2,1 %.
- **Alta cardinalidad:** `DeviceInfo` tiene 997 categorías, `id_33` 108, `id_31` 97 e `id_30` 71; las categorías que aparecen en menos del 1 % se agrupan.
- **Dónde se concentra el fraude:** producto C (~12 %) frente a W (~2 %); crédito (~6,6 %) frente a débito (~2,5 %); móvil (~10 %) frente a escritorio (~6 %); correo outlook.com es el de más fraude.
- **Monto:** casi no separa fraude de legítimas (mediana 75 contra 69); los valores atípicos no se eliminan y se usa el logaritmo del monto.
- **Tiempo:** la tasa de fraude cambia a lo largo de los días, lo que justifica entrenar con lo antiguo y probar con lo reciente.

## Preprocesamiento y reducción de dimensionalidad
- De 434 columnas quedaron 311: se quitaron 12 casi vacías (más de 90 % de nulos), 1 constante y 110 muy correlacionadas (mayor a 0,95).
- Nulos: mediana en modelos lineales; en categóricas, una categoría "desconocido".
- Variables nuevas: hora, día de la semana y logaritmo del monto.
- Después de codificar quedan 398 columnas; con PCA bastan 105 componentes para conservar el 90 % de la varianza (86 en el notebook no supervisado, donde además se recortan los valores extremos).

## Principales resultados
**Supervisado** (prueba: 45.000 transacciones más recientes; un modelo sin información tendría PR-AUC de 0,036):

| Modelo | PR-AUC | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| Regresión logística | 0,216 | 0,821 | 0,39 | 0,30 |
| Gradient Boosting sin pesos | 0,475 | 0,868 | 0,65 | 0,37 |
| Gradient Boosting con pesos | 0,470 | 0,866 | 0,65 | 0,38 |
| **Gradient Boosting ajustado** | **0,481** | **0,873** | 0,58 | 0,39 |

- El mejor modelo es unas 13 veces mejor que el azar. Los pesos de clase y el ajuste de hiperparámetros casi no cambiaron el resultado.
- Variables que más pesan: `C1`, `C13`, `TransactionAmt`, `card6` y `V318`.
- Costo de errores (supuesto: un fraude que pasa cuesta 10 veces más que revisar una transacción legítima): el umbral que minimiza el costo detecta el 64 % de los fraudes (contra 39 %) y baja el costo total cerca de un 12 %, a cambio de más falsos positivos.

**No supervisado** (sin usar la etiqueta; se usa después solo para interpretar):
- Hay dos tipos de transacción: un grupo pequeño de producto C (13,2 % de fraude, 3,7 veces el promedio) y uno grande de producto W (2,5 %). La separación es débil (silhouette de 0,30).
- K-Means funcionó mejor que la mezcla gaussiana (silhouette 0,36 contra 0,09 en la prueba).
- Isolation Forest logra ROC-AUC de 0,75 y PR-AUC de 0,12 (unas 3 veces el azar): de las transacciones que marca como más raras (el 3 %), el 21 % son fraude.
- Con 86 componentes de PCA se conserva el 90 % de la varianza; las dos primeras conservan solo el 19 %.

## Conclusiones
- El fraude es muy poco frecuente (3,5 %), por lo que se evalúa con PR-AUC, recall y precision y no con el porcentaje de aciertos.
- El enfoque supervisado detecta mucho mejor el fraude (PR-AUC de 0,48 contra 0,12 del mejor método no supervisado), pero aun así deja pasar cerca de 6 de cada 10 fraudes con el umbral que equilibra precision y recall.
- El no supervisado no necesita etiquetas y encontró que lo atípico tiene más probabilidad de ser fraude, pero no todo lo atípico lo es.
- Recomendación para una aplicación real: usar el supervisado como detector principal, con un umbral elegido según el costo real de cada tipo de error, y el Isolation Forest como complemento para señalar casos raros; reentrenar con frecuencia porque el fraude cambia con el tiempo.
- Probamos también usar todas las transacciones y variables por cliente: el resultado se mantuvo en torno a 0,5 de PR-AUC, así que el límite parece estar en la información disponible.
