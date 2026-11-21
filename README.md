# 🍷 Classificação de Qualidade de Vinho com Boosting + SMOTE

Projeto de **classificação binária desbalanceada** para identificar vinhos de alta qualidade (`NOTA = 1`) a partir de suas propriedades físico-químicas. Compara dois algoritmos de *boosting* (**AdaBoost** e **Gradient Boosting**) com balanceamento via **SMOTE** e ajuste de hiperparâmetros com **GridSearchCV**.

## 🎯 Objetivo

Prever se um vinho recebe a classe de alta qualidade (`NOTA = 1`) e descobrir quais características físico-químicas mais influenciam essa classificação, priorizando a capacidade de **encontrar os vinhos bons** (recall e F1 da classe positiva).

## 📂 Estrutura do repositório

| Arquivo | Descrição |
|---|---|
| `hands_com_parametros.ipynb` | Notebook com modelagem, tuning, avaliação e gráficos |
| `nota_vinho.csv` | Base de dados com variáveis físico-químicas e a coluna `NOTA` |
| `matriz_confusao_*.png` | Matrizes de confusão (teste) de cada modelo |
| `importancia_*.png` | Importância das variáveis de cada modelo |
| `pairplot_nota_*.png` | Pairplots das variáveis por classe |

## 📊 Dados

- **Variáveis preditoras (11):** `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`.
- **Alvo:** `NOTA` (binária), criada a partir da coluna original `quality`.
- ⚠️ **Prevenção de vazamento:** a coluna `quality` foi **removida** das features, pois a `NOTA` foi derivada dela e deixaria o problema trivial.
- A classe positiva é minoritária, o que justifica o uso de SMOTE e de F1/recall como métricas principais.

## 🔬 Metodologia

1. **Split** 70/30 estratificado (`random_state=42`).
2. **Pipeline (`imblearn`)** com **SMOTE + modelo**. Como o SMOTE está dentro do pipeline, ele só é aplicado nos dados de treino de cada fold, evitando vazamento na validação cruzada.
3. **GridSearchCV** com `StratifiedKFold` de 5 folds, otimizando **F1**.
4. **Validação cruzada** do melhor modelo sobre o treino e avaliação final no teste.
5. **Métricas:** Acurácia, Recall, Precision e F1, com as diferenças Treino–Teste–CV para diagnosticar overfitting.
6. **Interpretação:** importância das variáveis, matrizes de confusão e pairplots.

### Hiperparâmetros testados

| Modelo | Grade de busca | Melhor combinação |
|---|---|---|
| **AdaBoost** (base: árvore) | `n_estimators` [100, 200, 300] · `learning_rate` [0,05, 0,1, 0,5, 1,0] · `max_depth` [1, 2, 3] | 200 estimadores · lr 0,5 · depth 3 |
| **Gradient Boosting** | `n_estimators` [100, 200] · `learning_rate` [0,05, 0,1] · `max_depth` [2, 3, 4] · `subsample` [0,8] · `min_samples_leaf` [5, 10, 20] | 200 estimadores · lr 0,1 · depth 3 · leaf 5 · subsample 0,8 |

## 📈 Resultados

| Modelo | Conjunto | Acurácia | Recall | Precision | F1 |
|---|---|---|---|---|---|
| **AdaBoost** | Treino | 0,939 | 0,921 | 0,714 | 0,805 |
| | Teste | 0,896 | 0,739 | 0,593 | 0,658 |
| | Validação cruzada | 0,843 | 0,612 | 0,448 | 0,514 |
| **Gradient Boosting** | Treino | 0,964 | 0,934 | 0,826 | 0,877 |
| | Teste | 0,906 | 0,708 | 0,639 | 0,672 |
| | Validação cruzada | 0,852 | 0,599 | 0,472 | 0,525 |

### Importância das variáveis (top 5)

| # | AdaBoost | Gradient Boosting |
|---|---|---|
| 1 | alcohol (0,197) | **alcohol (0,474)** |
| 2 | free sulfur dioxide (0,113) | sulphates (0,145) |
| 3 | total sulfur dioxide (0,111) | volatile acidity (0,095) |
| 4 | residual sugar (0,095) | total sulfur dioxide (0,068) |
| 5 | volatile acidity (0,085) | free sulfur dioxide (0,044) |

## 💡 Conclusões

- **Os dois modelos têm desempenho muito próximo.** O Gradient Boosting leva vantagem leve em acurácia, precision e F1 no teste (0,672 vs 0,658). O AdaBoost tem recall um pouco maior no teste (0,739 vs 0,708).
- **Álcool é a variável mais importante nos dois modelos**, de forma consistente. Sulfatos, acidez volátil e dióxido de enxofre aparecem em seguida.
- **Há overfitting relevante.** O F1 cai de ~0,80–0,88 no treino para ~0,66–0,67 no teste e ~0,51–0,52 na validação cruzada. O modelo "decora" parte do treino, sobretudo por causa dos exemplos sintéticos do SMOTE.
- **A acurácia engana** em bases desbalanceadas. Por isso F1 e recall são as métricas de referência, e elas mostram que o problema ainda é difícil: a precision da classe positiva fica entre 0,45 e 0,64.
- O teste ficou melhor que a validação cruzada. Como a classe positiva tem poucos exemplos no teste, essa diferença tem variância alta, e a **CV é a estimativa mais conservadora**.

## ⚠️ Limitações

- Poucos exemplos da classe positiva, o que deixa as métricas instáveis.
- Diferença grande entre treino e validação, indicando espaço para mais regularização.
- A melhor `max_depth` do AdaBoost (3) e o melhor `min_samples_leaf` do Gradient Boosting (5) ficaram na **borda da grade**, então a busca pode não ter encontrado o ótimo.

## 🚀 Próximos passos

- Ampliar a grade de hiperparâmetros (ou usar `RandomizedSearchCV`/Optuna) e testar maior regularização;
- Comparar com **Random Forest, XGBoost e LightGBM**;
- Testar `class_weight`/ajuste de limiar de decisão como alternativa ao SMOTE e analisar a curva Precisão-Recall;
- Explicar as previsões com **SHAP**;
- Repetir a validação com `RepeatedStratifiedKFold` para reduzir a variância das métricas.

## 🧰 Tecnologias

Python · Pandas · Scikit-learn · Imbalanced-learn (SMOTE) · Seaborn · Matplotlib · Jupyter Notebook

## ▶️ Como executar

```bash
pip install pandas scikit-learn imbalanced-learn seaborn matplotlib jupyter
jupyter notebook hands_com_parametros.ipynb
```

Mantenha o arquivo `nota_vinho.csv` na mesma pasta do notebook.
