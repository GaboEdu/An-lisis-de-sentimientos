import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

# ------------------------------
# 1. Cargar datos
# ------------------------------
try:
    df = pd.read_excel("analisis_sentimientos_trump_iran_otan.xlsx")
except FileNotFoundError:
    print("Error: El archivo no se encontró.")
    exit(1)
except pd.errors.EmptyDataError:
    print("Error: El archivo está vacío.")
    exit(1)

# ------------------------------
# 2. Exploración inicial
# ------------------------------
print("\nPrimeras filas del dataset:")
print(df.head())

print("\nInformación general:")
print(df.info())

print("\nEstadísticas descriptivas:")
print(df.describe(include='all'))

# ------------------------------
# 3. Limpieza de datos
# ------------------------------
df = df.drop_duplicates()
df = df.fillna({'Sentimiento': 'Desconocido', 'Palabra': 'Desconocido'})

# ------------------------------
# 4. Análisis univariado - Frecuencia de palabras
# ------------------------------
top_palabras = df['Palabra'].value_counts().head(10)

plt.figure(figsize=(8, 4))
sns.barplot(x=top_palabras.values, y=top_palabras.index, palette="Set2")
plt.title("Top 10 palabras más frecuentes")
plt.xlabel("Frecuencia")
plt.ylabel("Palabra")
plt.tight_layout()
plt.show()

# ------------------------------
# 5. Gráfica de pastel - Sentimientos
# ------------------------------
sentimiento_counts = df['Sentimiento'].value_counts()

colores = {
    'Positivo': '#2ecc71',
    'Negativo': '#e74c3c',
    'Neutro':   '#3498db'
}
colors = [colores.get(s, '#95a5a6') for s in sentimiento_counts.index]

plt.figure(figsize=(7, 7))
wedges, texts, autotexts = plt.pie(
    sentimiento_counts,
    labels=sentimiento_counts.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=140,
    explode=[0.05] * len(sentimiento_counts),
    shadow=True,
    textprops={'fontsize': 12}
)

for autotext in autotexts:
    autotext.set_fontweight('bold')
    autotext.set_color('white')

plt.title("Distribución de Sentimientos\nTrump - Irán - OTAN",
          fontsize=14, fontweight='bold', pad=20)

plt.legend(
    wedges,
    [f"{label}: {count}" for label, count in zip(sentimiento_counts.index, sentimiento_counts)],
    title="Sentimientos",
    loc="lower right",
    fontsize=10
)

plt.tight_layout()
plt.savefig("grafica_sentimientos.png", dpi=150, bbox_inches='tight')
plt.show()

# ------------------------------
# 6. Conteo de sentimientos (barra)
# ------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x='Sentimiento', data=df, palette=colores)
plt.title("Conteo de Sentimientos")
plt.xlabel("Sentimiento")
plt.ylabel("Cantidad")
plt.tight_layout()
plt.show()