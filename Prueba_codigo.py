# %% [markdown]
# # Análisis de consumo 

# %%
# Cargar las librerias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import f_oneway
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# %%
# Subir la base de datos 
consumo = pd.read_csv('Consumo.csv')

# %%
# Verificar la base de datos 
consumo.isnull().sum()
consumo.dtypes
consumo.head()
consumo.info()

# %% [markdown]
# ## Análisis descriptivo de los datos

# %%
consumo.describe()

# %% [markdown]
# En el análisis descriptivo de los datos, ya hemos encontrado información interesante. En promedio, los productos más consumidos son los productos frescos, con una media de 12.000 UM. Además, el valor máximo registrado alcanzó los 112.151 UM. La categoría más cercana a los productos frescos son los productos comestibles, los cuales también presentan cifras significativas. Por otro lado, los productos con menor consumo promedio son los delicatessen, probablemente debido a su exclusividad. Un aspecto común en todos los datos es su alta variabilidad; para cada una de las variables, la desviación estándar está por encima de la media, lo que indica que los datos están bastante dispersos. Esta alta variabilidad puede complicar algunos análisis predictivos.

# %%
plt.figure(figsize=(12, 8))
sns.boxplot(data=consumo, orient="h")
plt.title("Gastos por categoria")
plt.xlabel("Gasto anual (UM)")
plt.ylabel("Categoria del gasto")

# %% [markdown]
# En el boxplot que hemos creado, podemos observar la distribución gráfica de los datos. Notamos que la mayoría de los datos para todas las categorías se encuentran entre 3 UM y 18.000 UM aproximadamente, con una dispersión más alta en los productos frescos y comestibles. También podemos observar que fuera de los cuartiles hay una gran cantidad de valores atípicos, los cuales no son suficientes para entrar en los cuartiles, pero aún así deben ser tenidos en cuenta.

# %% [markdown]
# ### Análisis por canal y región

# %%
paleta = ["#FF6961","#77DD77","#FFC300"]
plt.figure(figsize=(12, 8))
sns.barplot(data=consumo, x='Channel', y='Fresh', hue='Region', errorbar=None, palette=paleta)
plt.title("Gasto en productos frescos por canal y región")
plt.xlabel("Canal")
plt.ylabel("Gasto anual en productos frescos (UM)")

# %% [markdown]
# En el siguiente gráfico de barras, podemos observar varias cosas. Para los productos frescos, separados por región y categoría, se observa que tienen una mayor representación en el mercado Horeca en cada una de las regiones. Es necesario recalcar que la región 'Otro' tiene una gran participación en este mercado.

# %%
paleta = ["#FF6961","#77DD77","#FFC300"]
plt.figure(figsize=(12, 8))
sns.barplot(data=consumo, x='Channel', y='Milk', hue='Region', errorbar=None, palette=paleta)
plt.title("Gasto en productos lacteos por canal y región")
plt.xlabel("Canal")
plt.ylabel("Gasto anual en productos lacteos (UM)")

# %% [markdown]
# Para los productos lácteos, se observa el caso contrario a los frescos: la mayor participación se encuentra en el canal minorista. Posiblemente, este cambio se deba a los patrones de consumo y costumbres de los hogares, o al problema del almacenamiento.

# %%
paleta = ["#FF6961","#77DD77","#FFC300"]
plt.figure(figsize=(12, 8))
sns.barplot(data=consumo, x='Channel', y='Grocery', hue='Region', errorbar=None, palette=paleta)
plt.title("Gasto en productos comestibles por canal y región")
plt.xlabel("Canal")
plt.ylabel("Gasto anual en productos comestibles (UM)")

# %% [markdown]
# Para la categoría de comestibles, nuevamente se observa una mayor representación del canal doméstico, lo cual tiene sentido ya que estos productos son más específicos y dirigidos a los hogares en particular.

# %%
paleta = ["#FF6961","#77DD77","#FFC300"]
plt.figure(figsize=(12, 8))
sns.barplot(data=consumo, x='Channel', y='Frozen', hue='Region', errorbar=None, palette=paleta)
plt.title("Gasto en productos congelados por canal y región")
plt.xlabel("Canal")
plt.ylabel("Gasto anual en productos congelados (UM)")

# %% [markdown]
# Las gráficas coinciden con lo esperado. Para el canal Horeca, se espera una mayor participación, ya que los productos congelados se utilizan mucho más en este sector, ya sean carnes, verduras o frutas. En el caso de los productos congelados, sería interesante analizar el mercado de Oporto, ya que muestra un alto gasto en estos productos para el sector Horeca. Se necesita más información para encontrar la razón.

# %%
paleta = ["#FF6961","#77DD77","#FFC300"]
plt.figure(figsize=(12, 8))
sns.barplot(data=consumo, x='Channel', y='Detergents_Paper', hue='Region', errorbar=None, palette=paleta)
plt.title("Gasto en productos detergentes_papel por canal y región")
plt.xlabel("Canal")
plt.ylabel("Gasto anual en productos detergentes_papel (UM)")

# %% [markdown]
# Como era de esperarse, el canal minorista es el mayor consumidor de este tipo de productos, con un gasto de más de 8.000 unidades monetarias anuales para las regiones de Lisboa y Oporto, y un poco menos para la región 'Otro'.

# %%
paleta = ["#FF6961","#77DD77","#FFC300"]
plt.figure(figsize=(12, 8))
sns.barplot(data=consumo, x='Channel', y='Delicassen', hue='Region', errorbar=None, palette=paleta)
plt.title("Gasto en productos delicatessen por canal y región")
plt.xlabel("Canal")
plt.ylabel("Gasto anual en productos delicatessen (UM)")

# %% [markdown]
# El caso de los productos delicatessen es muy interesante. Para la región de Oporto, se registran consumos similares en ambos canales, lo cual es curioso. Esto nos indica que las demandas de productos delicatessen no varían demasiado entre estos canales. Sin embargo, se puede notar que Lisboa y la región 'Otro' tienen posiblemente un nivel socioeconómico un poco más elevado que Oporto. Aunque estas son conclusiones preliminares.

# %% [markdown]
# ### Análisis de correlación 

# %%
data_horeca = consumo[consumo['Channel'] == 1]

# Calcular la matriz de correlación
correlaciones_horeca = data_horeca[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']].corr()

# Visualizar el mapa de calor de la matriz de correlación para Horeca
mask = np.triu(np.ones_like(correlaciones_horeca, dtype=bool))
plt.figure(figsize=(10, 8))
sns.heatmap(correlaciones_horeca, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, mask=mask)
plt.title("Mapa de calor de correlaciones entre categorías de gastos para Horeca")

# %% [markdown]
# En este análisis de correlación para el canal Horeca, podemos observar una interesante relación positiva entre los alimentos lácteos y los delicatessen. Esto puede deberse a algunos tipos de recetas en común o a que algunos de los productos delicatessen más comprados sean a la vez lácteos, lo que nos permite generar alguna campaña publicitaria específica.
# 
# Otra alta relación se encuentra entre los lácteos y los productos comestibles. Esto puede ser nuevamente una relación por categoría, ya que los lácteos también son comestibles, o puede ser un tema de complementariedad, ya que a la hora de comprar los productos de consumo en una misma canasta estén incluidos los productos lácteos para la preparación de alguna receta en el mercado Horeca, dejando una alta correlación en el consumo. Algo similar sucede con los productos comestibles y los detergentes y productos de papel, pero en menor medida.

# %%
data_minorista = consumo[consumo['Channel'] == 2]

# Calcular la matriz de correlación
correlaciones_minorista = data_minorista[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']].corr()

# Visualizar el mapa de calor de la matriz de correlación
mask = np.triu(np.ones_like(correlaciones_minorista, dtype=bool))
plt.figure(figsize=(10, 8))
sns.heatmap(correlaciones_minorista, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, mask=mask)
plt.title("Mapa de calor de correlaciones entre categorías de gastos para Minorista")

# %% [markdown]
# Para el canal minorista, encontramos varios datos significativamente altos. Se trata de los productos comestibles, detergentes y productos de papel, lácteos y detergentes y productos de papel, y comestibles con lácteos. Esto era de esperarse, ya que en la cesta de la compra de una familia, estos productos son comunes, por lo que no sorprende la similitud en el gasto entre ellos.

# %% [markdown]
# ## Análisis estadisticos 

# %%
# Prueba de normalidad 
variables = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

# Realizar el test de normalidad de Shapiro-Wilk para cada columna numérica
for columna in variables:
    stat, p_valor = shapiro(consumo[columna])
    alpha = 0.05
    print("Variable:", columna)
    print("Estadístico de prueba:", stat)
    print("Valor p:", p_valor)
    if p_valor > alpha:
        print("Los datos de", columna, "parecen provenir de una distribución normal (no se rechaza H0)")
    else:
        print("Los datos de", columna, "no parecen provenir de una distribución normal (se rechaza H0)")
    print()

# %% [markdown]
# Al realizar el test de normalidad de Shapiro-Wilk para cada una de las variables, nos damos cuenta de que no cumplen con el supuesto de normalidad. Por lo tanto, realizar pruebas estadísticas paramétricas no sería adecuado. Por ello, intentamos realizar pruebas no paramétricas, como la prueba de Wilcoxon.

# %%
#Segmentar las categorias
canal1 = consumo[consumo['Channel'] == 1]
lisboa1 = canal1[canal1['Region'] == 1]
oporto1 = canal1[canal1['Region'] == 2]
otro1 = canal1[canal1['Region'] == 3]
canal2 = consumo[consumo['Channel'] == 2]
lisboa2 = canal2[canal2['Region'] == 1]
oporto2 = canal2[canal2['Region'] == 2]
otro2 = canal2[canal2['Region'] == 3]
len(canal1), len(canal2)

# %% [markdown]
# A la hora de intentar la prueba de Wilcoxon, se requiere la misma cantidad de datos para cada variable a analizar, lo cual no es posible de realizar sin borrar datos. Por lo tanto, se procede a realizar las pruebas paramétricas normales, aunque no tengan mucha validez estadística.

# %%
# Agrupar las variables a análizar
variables_analisis = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

# Realiza el test de ANOVA para cada variable
for variable in variables_analisis:
    muestras = [canal1[variable], canal2[variable]]
    
    # Realiza el test de ANOVA
    stat, p_valor = f_oneway(*muestras)
    
    alpha = 0.05
    print("Variable:", variable)
    print("Estadístico de prueba:", stat)
    print("Valor p:", p_valor)
    if p_valor > alpha:
        print("No hay diferencias significativas entre los canales para la variable", variable)
    else:
        print("Hay diferencias significativas entre los canales para la variable", variable)
    print()

# %% [markdown]
# Estos resultados son congruentes con las gráficas que se presentaron al inicio del análisis, ya que para todas las variables hay diferencias significativas entre los canales Horeca y minorista, excepto para los productos delicatessen, para los cuales los datos eran curiosamente similares en ambos canales.

# %% [markdown]
# ## Análisis predictivo

# %%
for variable_objetivo in variables_analisis:
    # Seleccionar la variable objetivo actual
    y = consumo[variable_objetivo]
    
    # Seleccionar las características (X) correspondientes
    X = consumo.drop(['Channel', 'Region'], axis=1)
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=58)
    
    # Inicializar el modelo de regresión lineal
    model = LinearRegression()
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    # Hacer predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Calcular el error cuadrático medio (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print("Error cuadrático medio para", variable_objetivo, ":", mse)
    
    # Imprimir los coeficientes del modelo para ver su importancia
    print("Coeficientes del modelo para", variable_objetivo, ":", model.coef_)
    print()

# %% [markdown]
# Para realizar el análisis predictivo, se escogió un modelo de regresión lineal, ya que es el más común y, al no tener datos temporales, es el más acertado.
# 
# El modelo intenta explicar una variable en función de las demás variables presentes en el modelo. La idea es encontrar relaciones que nos permitan vincular la variable objetivo con las demás.
# 
# En este caso, no fue posible, ya que ninguno de los valores que arroja el modelo para cada una de las variables correspondientes resultó significativo, excepto para la variable en sí misma, pero ese resultado no nos dice mucho. Esto se puede explicar porque la mayoría de las variables tienen una alta desviación estándar, lo que hace difícil llegar a conclusiones válidas.
# 
# Al momento de realizar las regresiones, se segmentaron los datos por canal y región para cada una de las variables en particular, pero en todos los casos los valores eran similares y no significativos, por lo que se dejó el dataframe completo en el resultado final.

# %% [markdown]
# ## Conclusiones

# %% [markdown]
# Los datos proporcionados nos muestran los patrones de consumo anuales para ciertos productos en particular, del análisis podemos concluir.
# 
# 1. Los productos más consumidos son los productos frescos. Sería necesario realizar un análisis más profundo para decidir si los demás productos requieren una campaña publicitaria o para entender por qué los productos frescos están tan bien posicionados.
# 2. Los datos muestran una dispersión considerable, con desviaciones estándar más altas que la media, lo que complica la validez de los modelos predictivos.
# 3. Para ambos canales, los productos delicatessen tienen consumos similares en las tres regiones. Sería interesante analizar este segmento más a fondo para encontrar la razón en particular.
# 4. En el caso del canal Horeca, la matriz de correlación muestra una fuerte relación entre los lácteos y los productos delicatessen, así como también con los productos comestibles. Por lo tanto, organizar estos productos en los establecimientos de tal forma que los lácteos estén cerca de estos dos productos sería una buena estrategia para incentivar las ventas.
# 5. Para el canal minorista, existe una estrecha relación entre los productos de detergentes y papel y los comestibles. Sin embargo, colocarlos muy cerca no es necesario debido a la alta relación. Sería más efectivo realizar campañas de promoción entre lácteos y comestibles, ya que tendrían un buen impacto en las ventas de ambos.
# 6. Los datos no siguen una distribución normal y algunos tests no son muy confiables.


