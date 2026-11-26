# 🧠 Aprendizaje No Supervisado: Setas 🍄 y Tarjetas de crédito 💳

Proyecto individual del bootcamp de IA de **Factoría F5**. Aplico las mismas técnicas de **machine learning no supervisado** a dos datasets muy distintos:

- **Reducción de dimensionalidad:** PCA y t-SNE
- **Clustering:** K-Means, Aglomerativo, GMM, DBSCAN (y HDBSCAN como extra)
- **Detección de anomalías:** Isolation Forest

La idea central es el **contraste**: en las setas hay **grupos reales** y una etiqueta para comprobarlos; en las tarjetas hay una **nube continua** sin etiqueta, y el clustering sirve para **dibujar segmentos útiles para el negocio**.

---

## 🗂️ Contenido

| Notebook | Dataset | Datos | ¿Etiqueta? |
|---|---|---|---|
| [`workshop-clustering-Mushrooms.ipynb`](workshop-clustering-Mushrooms.ipynb) | [`data/mushrooms.csv`](data/mushrooms.csv) · [Kaggle](https://www.kaggle.com/uciml/mushroom-classification) | 8124 setas · 22 variables **categóricas** | Sí (`class`), **solo para validar** |
| [`workshop-clustering-creditcard.ipynb`](workshop-clustering-creditcard.ipynb) | [`data/credit_card.csv`](data/credit_card.csv) · [Kaggle](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata) | 8950 clientes · 17 variables **numéricas** | **No** |

Cada notebook incluye explicaciones en markdown antes de cada paso y una **📝 Observación** después de cada resultado, y termina con **📝 Mis conclusiones**.

---

## 🍄 Parte 1: Setas (categóricas, con etiqueta)

**Preparación**
- **2480 nulos encubiertos** (`'?'` en `stalk-root`, 30,5 %) que `isnull()` no detecta → imputados con la **moda**.
- `veil-type` eliminada (un solo valor). **One-Hot Encoding**: de 21 a **115 columnas**.

**Resultados**

| Paso | Resultado |
|---|---|
| Random Forest (línea base supervisada) | **Accuracy 100 %**; bastan **8 componentes PCA** |
| PCA 2D | 29,6 % de la varianza, pero ya se ven grupos |
| t-SNE | **~15 subgrupos**, casi todos de una sola clase |
| Mejor k (silhouette) | **k = 9** (0,495), coincide con las "familias" de t-SNE |
| K-Means k = 2 vs etiqueta | **ARI 0,617 · NMI 0,566** sin haber visto la etiqueta |
| K-Means / Aglomerativo / GMM | ARI ≈ 0,60 en los tres → la estructura es **real** |
| DBSCAN con distancia **Jaccard** | **19 clusters casi puros** (homogeneidad 0,989) |
| Isolation Forest (5 %) | 407 setas atípicas, el 80 % comestibles → **raro ≠ venenoso** |

**Lección clave:** con datos one-hot la **distancia importa** (Jaccard > euclídea). Y un cluster "comestible" con un 16 % de setas venenosas **no es seguro** ☠️: el no supervisado sirve para **explorar**, no para sustituir a un modelo supervisado.

---

## 💳 Parte 2: Tarjetas de crédito (numéricas, sin etiqueta)

**Preparación**
- Nulos (`MINIMUM_PAYMENTS` 313, `CREDIT_LIMIT` 1) imputados con la **mediana**: los datos están muy **sesgados** (skew de `PURCHASES` = 8,1).
- **`StandardScaler`** imprescindible: las variables van de 0–1 (frecuencias) a decenas de miles de €.

**Resultados**

| Paso | Resultado |
|---|---|
| PCA | **7 componentes** para el 80 % de la varianza · **PC1 ≈ compras**, **PC2 ≈ adelantos de efectivo** |
| Mejor k (silhouette) | **k = 3** (0,251): silhouette **bajo** para todos los k |
| Comparativa | **K-Means gana en las 3 métricas internas** (silhouette, Davies-Bouldin, Calinski-Harabasz) |
| Dendrograma · DBSCAN · t-SNE | Sin saltos claros, un cluster con el 97 % de los clientes, zonas sin huecos |
| Isolation Forest (2 %) | **179 atípicos**, 154 coinciden con el ruido de DBSCAN; sobre todo **clientes premium** |

Cuatro herramientas distintas dicen lo mismo: **no hay grupos naturales**, es una nube continua. Los segmentos son **fronteras prácticas**, y se validan por su **interpretabilidad**.

### ⭐ Segmentos de negocio (K-Means, k = 3)

| Segmento | Clientes | Cliente típico | Estrategia |
|---|---|---|---|
| 🛍️ **Compradores activos (VIP)** | 1275 (14 %) | Compran casi cada mes (mediana 3056 €), límite alto, apenas usan efectivo | **Fidelizar**: cashback, subir límite, tarjeta premium |
| 💵 **Usuarios de efectivo (riesgo)** | 1561 (17 %) | Más de la mitad **no compra nada**: usan la tarjeta para sacar efectivo, saldo alto, casi nunca pagan el total | **Vigilar el riesgo** y acompañar: refinanciación, educación financiera |
| 😴 **Poco activos** | 6114 (68 %) | Saldo, límite y pagos bajos | **Activar**: promociones, campañas de uso |

Con **k = 4** los poco activos se dividen en **inactivos** y **compradores frecuentes a plazos**: peor silhouette (0,198) pero un segmento muy útil. Mantengo k = 3 como segmentación principal y propongo k = 4 como versión más detallada.

---

## 🚀 Extra: Para ir más allá

| Notebook | Extra | Resultado |
|---|---|---|
| 🍄 Setas | **HDBSCAN** (Jaccard) | 18 clusters, homogeneidad **0,994**, igual que el mejor DBSCAN pero **sin elegir `eps`** |
| 🍄 Setas | **Reglas de asociación** (Apriori, `mlxtend`) | «olor fétido → venenosa» al **100 %** (27 % de las setas); «sin olor → comestible» falla en ~120 setas |
| 💳 Tarjetas | **KPIs de ratio + `log1p`** | Segmentos **equilibrados** (2367 / 3379 / 3204) y sin clusters diminutos: 💵 **solo efectivo**, 🛍️ **transactors** (pagan todo) y 🔄 **revolvers** (aplazan la deuda) |

---

## 🧩 Setas vs Tarjetas

| | 🍄 Setas | 💳 Tarjetas |
|---|---|---|
| Variables | Categóricas | Numéricas |
| Preprocesado clave | Nulos encubiertos + One-Hot | Mediana + escalado (+ log) |
| Etiqueta | Sí (solo para validar) | No |
| Validación | ARI / NMI | Métricas internas + **interpretabilidad** |
| Estructura | **Islas** (grupos reales) | **Nube continua** (fronteras prácticas) |
| Silhouette del mejor k | 0,495 | 0,251 |
| Mejor distancia | Jaccard | Euclídea sobre datos escalados |
| Isolation Forest | Raro ≠ venenoso | Raro ≠ sospechoso (sobre todo clientes premium) |

**Conclusión:** no hay un algoritmo ni una métrica que gane siempre. Lo importante es elegir el **preprocesado**, la **distancia** y la **forma de validar** según el tipo de datos y la pregunta de negocio.

---

## 🔧 Tecnologías

- **Python 3.12** · Pandas · NumPy
- Matplotlib · Seaborn
- Scikit-learn: `PCA`, `TSNE`, `KMeans`, `AgglomerativeClustering`, `GaussianMixture`, `DBSCAN`, `HDBSCAN`, `IsolationForest`, `RandomForestClassifier`, métricas de clustering
- SciPy (`linkage`, `dendrogram`) · mlxtend (`apriori`, `association_rules`)

---

## 🚀 Cómo ejecutarlo

```bash
git clone https://github.com/IA-P1-BCN/IA-Project-Unsupervised-ML-Esra.git
cd IA-Project-Unsupervised-ML-Esra
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Abre los notebooks en **VS Code** o **Jupyter**, selecciona el kernel de `.venv` y ejecuta **Restart & Run All**. Recomiendo empezar por la **Parte 1 (setas)**: la Parte 2 da por sabido lo aprendido allí.

> 🔀 **Flujo de trabajo:** cada parte se desarrolló en su propia rama y se integró en `main` con un Pull Request (`parte1-setas`, `parte2-tarjetas`, `extra-parte1-setas`, `extra-parte2-tarjetas`, `docs-readme`). Las ramas están en este repositorio; los PRs (#1–#5) se pueden ver en mi repositorio personal: [esrakesken/unsupervised-ml](https://github.com/esrakesken/unsupervised-ml/pulls?q=is%3Apr).

---

## 📋 Enunciado original del taller

> Texto original del README del taller (proporcionado por el profesorado), sin cambios.

Este repositorio contiene un taller práctico de **machine learning no supervisado**: reducción de dimensionalidad (**PCA**, **t-SNE**), **clustering** (K-Means, Aglomerativo, GMM, DBSCAN) y **detección de anomalías** (Isolation Forest).

La novedad de esta versión es que el taller se divide en **dos notebooks complementarios**, cada uno con su propio dataset. La gracia está en el **contraste entre ambos**: un mismo conjunto de técnicas se comporta de forma muy distinta según el tipo de datos y según si tenemos o no una etiqueta de referencia.

---

### 🗂️ Estructura del repositorio

| Notebook | Dataset | Tipo de datos | ¿Hay etiqueta? |
|---|---|---|---|
| [`workshop-clustering-Mushrooms.ipynb`](workshop-clustering-Mushrooms.ipynb) | [`data/mushrooms.csv`](data/mushrooms.csv) | Categóricos | Sí — `class` (solo para **validar**) |
| [`workshop-clustering-creditcard.ipynb`](workshop-clustering-creditcard.ipynb) | [`data/credit_card.csv`](data/credit_card.csv) | Numéricos | No — segmentación **de verdad** |

> Hay que entregar **los dos notebooks**. No son independientes: la Parte 2 da por sabido lo aprendido en la Parte 1.

---

### 🍄 Parte 1 — Setas (datos categóricos, *con* etiqueta)

**Notebook:** [`workshop-clustering-Mushrooms.ipynb`](workshop-clustering-Mushrooms.ipynb) · **Dataset:** [`data/mushrooms.csv`](data/mushrooms.csv)
🔗 [Mushroom Dataset (Kaggle)](https://www.kaggle.com/uciml/mushroom-classification) · [UCI](https://archive.ics.uci.edu/ml/datasets/Mushroom)

Cada fila es un hongo descrito con **~22 variables, todas categóricas** (forma, color, olor, etc.). La variable `class` es **binaria**: `e` (comestible) / `p` (venenoso).

La clave pedagógica: **tenemos etiqueta, pero el clustering NO la usa**. La reservamos *solo para validar* a posteriori cuánta estructura real ha recuperado el modelo sin haberla visto.

**Qué se trabaja:**
- Carga, EDA y detección de nulos encubiertos (el valor `'?'`) y de columnas constantes (`veil-type`).
- Imputación con la moda y **One-Hot Encoding** (`pd.get_dummies`).
- **PCA** y **t-SNE** para visualizar un dataset de >100 dimensiones en 2D.
- **Random Forest** como *línea base supervisada* (¿cuánta información hay realmente?) y estudio de cuántas componentes PCA bastan para mantener la precisión.
- **Clustering**: K-Means (codo + *silhouette*), Aglomerativo (con **dendrograma**), GMM y DBSCAN.
- Lección con **DBSCAN**: con datos categóricos one-hot, la **distancia importa** (euclídea vs **Jaccard**).
- Validación con etiqueta: **Adjusted Rand Index (ARI)** y **NMI**.
- **Isolation Forest** para detección de anomalías.

---

### 💳 Parte 2 — Tarjetas de crédito (datos numéricos, *sin* etiqueta)

**Notebook:** [`workshop-clustering-creditcard.ipynb`](workshop-clustering-creditcard.ipynb) · **Dataset:** [`data/credit_card.csv`](data/credit_card.csv)
🔗 [Credit Card Dataset for Clustering (Kaggle)](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

Comportamiento de uso de ~9.000 titulares de tarjeta durante 6 meses, con **17 variables numéricas** (saldo, compras, adelantos de efectivo, límite, pagos…).

La clave pedagógica: **aquí NO hay etiqueta**. Es aprendizaje no supervisado «de verdad»: no se puede calcular ARI porque no existe una verdad de referencia. El éxito se mide con **métricas internas** y, sobre todo, con la **interpretabilidad** de los segmentos. El objetivo es **segmentar clientes** para una estrategia de marketing.

**Qué se trabaja:**
- Carga, EDA y tratamiento de nulos (imputación con la **mediana**, más robusta en datos sesgados).
- Observación del **sesgo** típico de datos financieros (histogramas).
- **Escalado** (`StandardScaler`) — imprescindible cuando las variables tienen escalas muy distintas.
- **PCA**: varianza explicada acumulada (*scree plot*) y proyección a 2D (aquí los datos son una **nube continua**, no grupos separados).
- **Clustering**: K-Means (codo + *silhouette*), Aglomerativo (dendrograma), GMM y DBSCAN.
- Validación **sin etiqueta**: *silhouette*, *Davies-Bouldin* y *Calinski-Harabasz*.
- Visualización de los segmentos con **t-SNE**.
- **Interpretación de perfiles** (heatmap de medias por cluster) → nombrar los segmentos en términos de negocio (VIP, riesgo, poco activos…). **Este es el entregable del caso.**
- **Isolation Forest** para detectar clientes atípicos.

---

### 🧩 ¿Por qué dos datasets?

| | 🍄 Setas | 💳 Tarjetas |
|---|---|---|
| Variables | Categóricas | Numéricas |
| Preprocesado clave | One-Hot Encoding | Escalado / imputación |
| Etiqueta | Sí (solo validar) | **No** |
| Cómo se valida | ARI / NMI (vs etiqueta) | Métricas internas + interpretabilidad |
| Estructura | Grupos separables | Nube continua |
| Distancia | Jaccard > euclídea | Euclídea sobre datos escalados |

**Conclusión transversal:** no hay un algoritmo ni una métrica que gane siempre. El acierto está en elegir el preprocesado, la distancia y la forma de validar **según el tipo de datos y el problema**.

---

### 🔧 Tecnologías

- Python · Pandas · NumPy
- Seaborn · Matplotlib (y opcionalmente Plotly)
- Scikit-learn: `PCA`, `TSNE`, `KMeans`, `AgglomerativeClustering`, `GaussianMixture`, `DBSCAN`, `IsolationForest`, `RandomForestClassifier` y métricas de clustering
- *(Opcional, para ir más allá)* `umap-learn`, `hdbscan`, `mlxtend`, SciPy (`linkage` / `dendrogram`)

---

### 📊 Evaluación

Se evaluarán las siguientes competencias **en ambos notebooks**:

**Competencia: Evaluar conjuntos de datos con herramientas de análisis y visualización**
- ✅ Uso y gestión de formato `.csv`
- ✅ Limpieza y preprocesado de datos
- ✅ Visualización de datos (Seaborn, Matplotlib, Plotly)
- ✅ Análisis exploratorio detallado (EDA)
- ✅ Técnicas de preprocesado (normalización, escalado, label/one-hot encoding)
- ✅ Técnicas avanzadas de limpieza (atípicos, imputación de faltantes)
- ✅ Técnicas de reducción de dimensionalidad (PCA, t-SNE)

**Competencia: Aplicar algoritmos de ML según el problema**
- ✅ Seleccionar las variables útiles y descartar las que no aportan
- ✅ Reconocer un caso de aprendizaje no supervisado
- ✅ Aplicar modelos de clustering
- ✅ Distinguir regresión / clasificación / clustering
- ✅ Separación de datos en train/test (Parte 1)
- ✅ Uso de modelos de *ensemble* (RandomForest como baseline en la Parte 1)
- ✅ Interpretación y validación de los resultados (ARI/NMI y métricas internas)

---

### 🚀 Cómo empezar

1. Clona el repositorio.
2. Los datasets están en la carpeta [`data/`](data/) (`data/mushrooms.csv` y `data/credit_card.csv`); al leerlos usa esa ruta, p. ej. `pd.read_csv("data/mushrooms.csv")`.
3. Abre cada notebook (Jupyter / VS Code) y **completa las celdas marcadas con comentarios** (`# ...`). Las celdas traen pistas, no la solución.
4. Empieza por la **Parte 1 (setas)** y luego haz la **Parte 2 (tarjetas)**.

> 💡 Cada notebook termina con una sección **«Para ir más allá»** con extensiones opcionales (UMAP, HDBSCAN, reglas de asociación, ingeniería de KPIs…) para quien quiera profundizar.
