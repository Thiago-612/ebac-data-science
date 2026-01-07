import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.linear_model import LinearRegression
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

pd.set_option('display.width', 120)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('CHURN_TELECON_FINAL.csv',
    #sep=';',
    encoding='utf-8',
    na_values=['', ' ', 'NA', 'None']
)

print('\n', df.head().to_string(), '\n')

print('\nVERIFICAÇÃO DO DATAFRAME: \n')
print(df.info())

#CONFERENCIA DE VALORES DAS COLUNAS

print('\nVERIFICAÇÃO DAS COLUNAS NUMÉRICAS: \n')
print('\n', df.describe().to_string(), '\n')

colunas = ['GÊNERO', 'TIPO_CONTRATO', 'CHURN',
           'IDOSO', 'CASADO', 'DEPENDENTES',
           'SERVIÇO TELEFÔNICO', 'SERVICO_INTERNET',
           'SERVICO_SEGURANCA', 'SUPORTE_TECNICO', 'STREAMINGTV',
           'MÉTODO DE PAGAMENTO']

print('\nCONFERINDO OS VALORES INSERIDOS NAS COLUNAS DE VARIÁVEIS CATEGÓRICAS: \n')
for c in colunas:
    print(f"\n📊 Coluna: {c}")
    print(df[c].value_counts(dropna=False).to_frame(name='Contagem'))

#CHECANDO POSSÍVEIS OUTLIERS ANÁLISANDO A MÉDIA COM A MEDIANA(50%)

"""
Análise:
A coluna TOTAL_PAGAMENTO tem indícios de outliers.
Após o boxplot verificou-se muitos valores altos, não são casos isolados.
Há uma quantidade razoável de clientes acima do terceiro quartil, não são outliers.
Após o histograma verifica-se que são clientes antigos.
Existe um padrão individual de cada variável na correlação com o churn, mas somente uma análise envolvendo todas
as variáveis juntas que mostrará um perfil de cliente positivo para o churn.
Foi decidido manter os valores fora dos quartis para entender melhor o perfil do cliente. Não são outliers e
a exclusão geraria uma amostra enviesada.
"""

colunas_numericas = [
    col for col in df.select_dtypes(include=['int64', 'float64']).columns
    if not col.endswith(('_CATCODES','_MEDIANA', '_MEDIA'))
]

# Listas para armazenar resultados
resultados = []

for col in colunas_numericas:
    media = df[col].mean()
    mediana = df[col].median()

    if pd.notna(media) and pd.notna(mediana):  # ignora colunas vazias
        diff = media - mediana
        resultados.append({
            'Coluna': col,
            'Média': media,
            'Mediana': mediana,
            'Diferença (Média - Mediana)': diff
        })

tabela = pd.DataFrame(resultados)

# Separar por tipo de assimetria
tabela_maior_media = tabela[tabela['Diferença (Média - Mediana)'] > 0].sort_values('Diferença (Média - Mediana)',ascending=False)
tabela_menor_media = tabela[tabela['Diferença (Média - Mediana)'] < 0].sort_values('Diferença (Média - Mediana)',ascending=True)

print('\nCOMPARANDO OS VALORES DA MÉDIA E MEDIANA INSERIDOS NAS COLUNAS DE VARIÁVEIS NUMÉRICAS:')
print('Valores altos na coluna Diferença indicam possíveis outliers')
print("\n📊 Colunas com MÉDIA MAIOR que MEDIANA (cauda à direita):")
print(tabela_maior_media.to_string(index=False))
print("\n📉 Colunas com MÉDIA MENOR que MEDIANA (cauda à esquerda):")
print(tabela_menor_media.to_string(index=False))

fig = px.box(
    data_frame=df,
    y="TOTAL_PAGAMENTO",
    points="all",
    title="Distribuição dos pagamentos totais",
    labels={"TOTAL_PAGAMENTO": "Total pago (R$)"}
)

fig.update_layout(
    yaxis_title="",
    xaxis_title="",
    title_font_size=20,
    template="plotly_white"
)

fig.show()

plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(
    df['TOTAL_PAGAMENTO'].dropna(),  #remove nulos
    bins=50,                         #menos bins = mais legível
    color='green',
    alpha=0.8,
    edgecolor='black'                #bordas visíveis
)

plt.title('Distribuição dos Pagamentos Totais', fontsize=14)
plt.xlabel('Valor Total Pago (R$)', fontsize=12)
plt.ylabel('Quantidade de Clientes', fontsize=12)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Exibir valores de contagem no topo das barras
for i in range(len(n)):
    if n[i] > 0:
        plt.text(bins[i], n[i], str(int(n[i])), fontsize=8, va='bottom')

plt.tight_layout()
plt.show()

# Cria faixas de tempo como cliente (exemplo: 0–12, 13–24, etc.)
df['TEMPO_FAIXA'] = pd.cut(
    df['TEMPO_COMO_CLIENTE'],
    bins=[0, 12, 24, 36, 48, 60, df['TEMPO_COMO_CLIENTE'].max()],
    labels=['0–12', '13–24', '25–36', '37–48', '49–60', '60+'],
    include_lowest=True
)

variaveis = [
    ('GÊNERO', 'Gênero'),
    ('CASADO', 'Casado'),
    ('DEPENDENTES', 'Dependentes'),
    ('TEMPO_FAIXA', 'Tempo como Cliente (faixas)')
]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f"CHURN por {label}" for _, label in variaveis],
    horizontal_spacing=0.10,
    vertical_spacing=0.15
)

for i, (col, label) in enumerate(variaveis):
    row = i // 2 + 1
    col_pos = i % 2 + 1

    # Tabela de frequências: quantos clientes por (CHURN, categoria)
    freq = (
        df
        .groupby(['CHURN', col])
        .size()
        .reset_index(name='QTDE')
        .dropna(subset=[col])  # evita categorias nulas no gráfico
    )

    # Para cada categoria da variável, cria uma barra separada
    for categoria in freq[col].unique():
        sub = freq[freq[col] == categoria]
        fig.add_trace(
            go.Bar(
                x=sub['CHURN'],
                y=sub['QTDE'],
                #name=f"{label}: {categoria}",
                name=str(categoria),  # nome curto na legenda
                legendgroup=label,  # agrupa legendas por variável
                showlegend=(row == 1 and col_pos == 1)  # legenda só no primeiro gráfico
            ),
            row=row,
            col=col_pos
        )

# Layout geral
fig.update_layout(
    title="Distribuição de CHURN por variáveis categóricas",
    title_font_size=22,
    template="plotly_white",
    height=900,
    width=1200,
    barmode='group',
    legend_title_text="Categorias",
    #showlegend=True,
    legend=dict(
        orientation="h",  # legenda horizontal
        yanchor="bottom",
        y=-0.12,
        xanchor="center",
        x=0.5,
        font=dict(size=11)
    )
)

# Ajustes de eixo
fig.update_xaxes(title_text="Churn (sim / não)")
fig.update_yaxes(title_text="Quantidade de clientes")

fig.show()

#VERIFICAÇÃO DO BALANCEAMENTO DAS VARIÁVEIS BOOLEANAS

"""
Análise:
Os gráficos de pizza mostram que a população possui um bom nível de balanceamento e que considera todas as 
características dos clientes, ocasionando uma diminuição do viés.
O serviço telefônico possui muitos NaN, pois o serviço não é necessário para poder ter internet e os
clientes que deixaram sem resposta podem ser um indicativo de que não possuem o serviço.
"""

categorias = [
    'GÊNERO', 'CASADO', 'DEPENDENTES', 'SERVIÇO TELEFÔNICO',
    'SERVICO_INTERNET', 'SERVICO_SEGURANCA', 'SUPORTE_TECNICO',
    'STREAMINGTV', 'TIPO_CONTRATO', 'MÉTODO DE PAGAMENTO',
    'CHURN', 'IDOSO'
]

fig = make_subplots(
    rows=3, cols=4,
    subplot_titles=categorias,
    specs=[[{'type': 'domain'}] * 4] * 3  # define que todos são gráficos de pizza
)

for i, col in enumerate(categorias):
    row = i // 4 + 1
    col_pos = i % 4 + 1
    contagem = df[col].value_counts(dropna=False)

    fig.add_trace(
        go.Pie(
            labels=contagem.index.astype(str),
            values=contagem.values,
            textinfo='percent+label',
            name=col
        ),
        row=row,
        col=col_pos
    )

fig.update_layout(
    height=900,
    width=1200,
    title_text="Distribuição das variáveis categóricas (Balanceamento)",
    showlegend=False,
    template="plotly_white"
)

fig.show()

#ANÁLISE BIVARIADA UTILIZANDO O PARCATS

"""
Análise:

Clientes que deram churn positivo (cancelaram o serviço):
Percebe-se um padrão nos clientes que deram churn.
O gênero não tem relevância, pessoas solteiras e/ou sem filhos têm uma têndencia ao cancelamento.
O serviço telefônico tem muitos clientes como NaN, existe a possibilidade deles não terem respondido
como uma forma de dizer que não possuem o serviço.
O serviço de internet por fibra óptica é predominante nos clientes que cancelaram, sendo um indicador
da insatisfação com o serviço (velocidade baixa ?).
Clientes que não tinham serviço de segurança cancelaram mais. Impossibilidade de definir o motivo de não
terem contratado. O preço elevado pode ser considerado como hipótese.
Clientes que não acionaram o suporte técnico cancelaram mais o serviço. Impossibilidade de definir o motivo. 
A contratação de streaming não tem relevância.
Contratos mês a mês correspondem a aproximadamente 90% dos clientes que cancelaram. Falta de fidelidade gera
um risco alto de cancelamento. A tolerância ao serviço de qualidade inferior fica menor, promoções de concorrentes
ou problemas financeiros levam ao cancelamento, dentre outros motivos.
Os métodos de pagamento estão desatualizados, cheques e transferências bancarias são poucos utilizados
atualmente. 
Clientes com idade inferior a 60 anos cancelam mais o serviço.

A falta de fidelidade nos contratos é o fator mais grave.
"""

#Colunas categóricas que vão virar dimensões
categorias = [
    'GÊNERO', 'CASADO', 'DEPENDENTES', 'SERVIÇO TELEFÔNICO',
    'SERVICO_INTERNET', 'SERVICO_SEGURANCA', 'SUPORTE_TECNICO',
    'STREAMINGTV', 'TIPO_CONTRATO', 'MÉTODO DE PAGAMENTO', 'IDOSO'
]

#Garante tudo como string
df[categorias + ['CHURN']] = df[categorias + ['CHURN']].astype(str)

#Separa os dataframes para uma melhor visualização
df_sim = df[df['CHURN'].str.upper() == 'SIM'].copy()
df_nao = df[df['CHURN'].str.upper() == 'NÃO'].copy()

#Função para criar lista de dimensões
def criar_dimensoes(df_local):
    return [
        dict(
            label=col,                       # nome no topo da dimensão
            values=df_local[col],            # valores (uma entrada por cliente)
            categoryorder='category ascending'
        )
        for col in categorias
    ]

dim_sim = criar_dimensoes(df_sim)
dim_nao = criar_dimensoes(df_nao)

#Parcats para CHURN = SIM
fig_sim = go.Figure(go.Parcats(
    dimensions=dim_sim,
    line=dict(
        color='crimson',    # todas as linhas vermelhas (clientes churn SIM)
        shape='hspline'
    ),
    hoveron='category',
    labelfont=dict(size=13, color='black'),
    tickfont=dict(size=11, color='gray'),
    arrangement='freeform'

))

fig_sim.update_layout(
    title="Perfil dos clientes com CHURN = SIM",
    title_font_size=20,
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=600
)

fig_sim.show()

#Parcats para CHURN = NÃO
fig_nao = go.Figure(go.Parcats(
    dimensions=dim_nao,
    line=dict(
        color='royalblue',  # todas as linhas azuis (clientes churn NÃO)
        shape='hspline'
    ),
    hoveron='category',
    labelfont=dict(size=13, color='black'),
    tickfont=dict(size=11, color='gray'),
    arrangement='freeform'

))

fig_nao.update_layout(
    title="Perfil dos clientes com CHURN = NÃO",
    title_font_size=20,
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=600
)

fig_nao.show()
