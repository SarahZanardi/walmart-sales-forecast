📈 Previsão de Vendas com Prophet
Este projeto utiliza Machine Learning para prever vendas semanais de uma loja e departamento específicos, com base em dados históricos e variáveis econômicas. O modelo escolhido é o Prophet, uma ferramenta robusta de séries temporais desenvolvida pelo Facebook.

🔍 Objetivo
Prever as vendas semanais da Loja 1 - Departamento 1 com base em:

Dados históricos de vendas

Fatores econômicos como CPI (Índice de Preços ao Consumidor) e Desemprego

Sazonalidade anual e semanal

🧰 Tecnologias Utilizadas
Python

Pandas

Matplotlib

NumPy

Prophet (Facebook)

Scikit-learn

📂 Estrutura dos Dados
Três arquivos .csv são utilizados:

train.csv: Dados históricos de vendas

features.csv: Variáveis adicionais (CPI, Desemprego, feriados)

stores.csv: Informações das lojas (não foi utilizado diretamente no modelo)

⚙️ Etapas do Projeto
Carregamento dos dados
Leitura dos arquivos CSV e verificação inicial da estrutura.

Tratamento dos dados

Preenchimento de valores ausentes em CPI e Unemployment

Merge entre dados de vendas e features

Filtro para Loja 1 e Departamento 1

Conversão e renomeação de colunas para uso com o Prophet

Visualização Inicial
Gráfico de vendas semanais para análise visual de tendências e sazonalidade.

Separação de Conjuntos
Divisão em treino (80%) e teste (20%) para validação do modelo.

Modelagem com Prophet

Inclusão de regressores externos (CPI e Desemprego)

Ajuste de sazonalidade anual e semanal

Previsão e Visualização
Geração de previsões com o Prophet e comparação com dados reais.

Avaliação do Modelo
Métricas utilizadas:

MAE (Erro Médio Absoluto)

RMSE (Raiz do Erro Quadrático Médio)

Exportação dos Resultados
Salvamento das previsões em previsao_loja1_depto1.csv.

📊 Resultados
As previsões são visualizadas em gráficos comparativos e avaliadas com métricas quantitativas. O modelo demonstra capacidade razoável de captura de padrões temporais com o uso de regressores.

📁 Arquivos Gerados
previsao_loja1_depto1.csv: Previsões semanais com intervalos de confiança.

Gráficos de linha com comparação entre valores reais e previstos.

🚀 Próximos Passos
Avaliar outros departamentos/lojas

Testar outros modelos de séries temporais (ARIMA, LSTM)

Otimizar hiperparâmetros do Prophet

Explorar sazonalidades personalizadas.
