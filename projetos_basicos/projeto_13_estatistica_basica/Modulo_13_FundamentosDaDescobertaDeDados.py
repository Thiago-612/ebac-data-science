import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly.io as pio

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('BASE_SUPERMERCADO.csv')#, delimiter=';')
print('\n', df.head().to_string(), '\n')

# Dicionário de tradução
traducao = {
    "belleza-y-cuidado-personal": "Beleza e Cuidados Pessoais",
    "comidas-preparadas": "Pratos Prontos",
    "congelados": "Congelados",
    "frutas": "Frutas",
    "instantaneos-y-sopas": "Sopas e Produtos Instantâneos",
    "lacteos": "Laticínios",
    "verduras": "Verduras"
}

# Aplica a tradução
df['Categoria'] = df['Categoria'].map(traducao)

print('\nVERIFICAÇÃO INICIAL: \n')
print(df.info())

print('\nANÁLISE DE DADOS ÚNICOS: \n')
print(df.nunique())

print('\nESTATÍSTICAS DOS DADOS: \n')
print(df.describe())

resumo = df.groupby('Categoria')['Preco_Normal'].agg(Média='mean', Mediana='median').reset_index()
print('\nMÉDIA E MEDIANA DO PREÇO POR CATEGORIA DE PRODUTOS: \n')
print(resumo)

media_maior = resumo[resumo['Média'] > resumo['Mediana']]
media_menor = resumo[resumo['Média'] < resumo['Mediana']]
print('\nMÉDIA MAIOR QUE A MEDIANA POR CATEGORIA DE PRODUTOS: \n')
print(media_maior)
print('\nMÉDIA MENOR QUE A MEDIANA POR CATEGORIA DE PRODUTOS: \n')
print(media_menor)

desvio_populacional = (
    df.groupby('Categoria')['Preco_Normal']
      .agg(Desvio_padrão=lambda x: x.std(ddof=0))  # ddof=0 → população
      .reset_index()
)
print('\nDESVIO PADRÃO DO PREÇO POR CATEGORIA DE PRODUTOS: \n')
print(desvio_populacional)

estatisticas = (
    df.groupby('Categoria')['Preco_Normal']
      .agg(
          Média='mean',
          Mediana='median',
          Desvio_padrão=lambda x: x.std(ddof=0)).reset_index()
)

estatisticas.plot(
    x='Categoria',
    y=['Média', 'Mediana', 'Desvio_padrão'],
    kind='bar',
    figsize=(12,8),
    color=['#4c72b0', '#55a868', '#c44e52']
)
plt.title('Média, Mediana e Desvio Padrão dos preços por Categoria')
plt.ylabel('Preço (R$)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.text(
    -1,                               # posição x (fora do gráfico, canto esquerdo)
    estatisticas[['Média', 'Desvio_padrão']].values.max() * 1.05,  # posição y acima do topo
    "📊 Interpretação:\n"
    "- Média ≈ Mediana: distribuição simétrica.\n"
    "- Média > Mediana: valores altos puxando a média.\n"
    "- Média < Mediana: valores baixos puxando a média.\n"
    "- Desvio Padrão alto: preços dispersos.\n"
    "- Desvio Padrão baixo: preços concentrados.",
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8)
)
plt.tight_layout()
plt.show()

df_filtrado = df[df['Categoria'] == 'Laticínios']
fig = px.box(df_filtrado, y="Preco_Normal",
    points="outliers",
    title=f"Distribuição de Preços - Laticínios",
    labels={"Preco_Normal": "Preço (R$)"},
    hover_data=["title", "Marca"]
)
fig.update_layout(
    yaxis_title="Preço dos produtos",
    xaxis_title="",
    title_font_size=20,
    template="plotly_white"
)
fig.show()

estatisticas2 = (
    df.groupby('Categoria')['Desconto']
      .agg(
          Média='mean')
          #Mediana='median',
          #Desvio_padrão=lambda x: x.std(ddof=0))
          .reset_index()
)

estatisticas2.plot(
    x='Categoria',
    y=['Média'],# 'Mediana', 'Desvio_padrão'],
    kind='bar',
    figsize=(10,6),
    color=['#4c72b0']#, '#55a868', '#c44e52']
)
plt.title('Média dos valores de desconto por Categoria')
plt.ylabel('Preço (R$)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

df_agg = (
    df.groupby(['Categoria', 'Marca'], as_index=False)
      .agg(Desconto_medio=('Desconto', 'mean'))
)

fig = px.treemap(
    df_agg,
    path=['Categoria', 'Marca'],
    values='Desconto_medio',             # área = média
    color='Desconto_medio',              # cor = média
    color_continuous_scale='RdBu',
    color_continuous_midpoint=df_agg['Desconto_medio'].mean(),
    hover_data={'Desconto_medio':':.2f'}
)
fig.update_layout(title="Treemap — Área e cor pelo desconto médio")
fig.show()


#USO PROPOSTO PELO TUTOR. SOMENTE PARA GERAR GRÁFICO NO JUPITER NOTEBOOK
# pio.renderers.default = 'notebook'
# pio.renderers.default = 'iframe_connected'
#
# desc_por_categoria_marca = df.groupby(['Categoria', 'Marca'])['Desconto'].mean().reset_index()
#
# desc_por_categoria_marca = desc_por_categoria_marca[desc_por_categoria_marca['Desconto'] > 0]
#
# fig = px.treemap(desc_por_categoria_marca,
#          path=['Categoria', 'Marca'],
#          values='Desconto',
#          title="Média de Desconto por Categoria e Marca",
#          color='Desconto',
#          color_continuous_scale='Viridis')
# fig.show()