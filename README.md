# Desafios Práticos de Machine Learning com XGBoost

Este repositório documenta 5 desafios práticos de Machine Learning utilizando a poderosa biblioteca XGBoost e o ecossistema Scikit-learn, cobrindo técnicas essenciais em regressão, classificação, tratamento de dados desbalanceados e engenharia de modelos.

---

## 1. O Corretor de Imóveis: Regressão (XGBRegressor)

**Cenário:** Previsão do preço mediano de casas na Califórnia. Este desafio foca em prever um valor contínuo (regressão).

* **Dataset:** `sklearn.datasets.fetch_california_housing`
* **Algoritmo:** `XGBRegressor`
* **Métricas de Avaliação:** **RMSE** (Root Mean Squared Error) e **MAE** (Mean Absolute Error). O foco é interpretar o erro médio da previsão em dólares.
* **Dica:** O objetivo do modelo deve ser `objective='reg:squarederror'`.

---

## 2. O Sommelier de Vinhos: Classificação Multiclasse

**Cenário:** Classificar quimicamente vinhos em uma de três categorias de cultivadores (Classe 0, 1 ou 2).

* **Dataset:** `sklearn.datasets.load_wine`
* **Algoritmo:** `XGBClassifier`
* **Requisitos:** Configurar o modelo para classificação multiclasse (`objective='multi:softmax'` e `num_class=3`).
* **Avaliação:** Geração e análise da **Matriz de Confusão** para identificar padrões de erros entre as classes.

---

## 3. O Detector de Fraudes: Dados Desbalanceados

**Cenário:** Detecção de fraude em transações financeiras, onde a classe positiva (fraude) representa menos de 1% dos dados. O desafio é superar a armadilha da acurácia alta, mas inútil.

* **Dataset:** Dados sintéticos gerados por `sklearn.datasets.make_classification` com proporção de classes de 99:1.
* **Técnicas:** Uso de `stratify=y` na divisão dos dados para manter a proporção das classes.
* **Avaliação:** Foco na métrica **Recall** (Revocação) da classe 1 (fraude).
* **Comparação:** Treinar um modelo sem ajuste e outro usando o parâmetro **`scale_pos_weight`** para dar maior importância à classe minoritária.

---

## 4. Duelo de Modelos: Árvore vs. XGBoost

**Cenário:** Comparar a performance e o custo-benefício de um modelo simples (`DecisionTreeClassifier`) contra um modelo de *ensemble* avançado (`XGBClassifier`) no mesmo conjunto de dados.

* **Dataset:** `sklearn.datasets.load_breast_cancer`
* **Algoritmos:** `DecisionTreeClassifier` e `XGBClassifier`.
* **Requisitos:** Treinar ambos com a mesma restrição de complexidade (`max_depth=3`) para uma comparação justa.
* **Avaliação:** Comparativo lado a lado da **Acurácia** de ambos os modelos no conjunto de teste.

---

## 5. Early Stopping e Salvamento: Engenharia de ML

**Cenário:** Otimizar o processo de treinamento em termos de tempo e evitar *overfitting* (sobreajuste), além de simular o processo de salvar e carregar o modelo para uso em produção.

* **Dataset:** Qualquer um dos anteriores (utilizado o `load_breast_cancer`).
* **Requisitos:**
    1.  Implementar **Early Stopping** (`early_stopping_rounds=10`) durante o treino para parar o processo automaticamente.
    2.  **Salvar o modelo** treinado (`bst.save_model("modelo.json")`).
    3.  **Carregar o modelo** salvo (`bst_carregado.load_model(...)`) e validar sua capacidade de previsão.
* **Nota Técnica:** Para garantir a funcionalidade do Early Stopping em todas as versões do XGBoost, utiliza-se a API nativa (`xgb.DMatrix` e `xgb.train`).