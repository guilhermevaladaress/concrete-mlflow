# Treinamento observável: Concrete + MLP (PyTorch) + MLflow

Atividade avaliativa de Sistemas de Informação (UNITINS). O script de MLP da aula (`mlp_torch_avaliacao.py`, Iris, classificação) foi portado para um pipeline **observável** com MLflow: cada experimento vira uma run com parâmetros, métricas por época, artefatos e um trace com as etapas do pipeline.

## Dataset e tarefa

- **Concrete Compressive Strength** (UCI, id 165), baixado via `ucimlrepo`.
- 1030 amostras, 8 features numéricas (cimento, escória, cinza volante, água, superplastificante, agregado graúdo, agregado miúdo, idade em dias), sem valores ausentes.
- Alvo: resistência à compressão do concreto, em **MPa**.
- Tarefa: **regressão**. Comparado ao script original, a saída tem 1 neurônio, a loss é `MSELoss` e as métricas são RMSE, MAE e R² em MPa (no lugar de acurácia, F1 e matriz de confusão).

## Ambiente

- Python **3.10.11** (Windows, PowerShell)

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
# se o PowerShell bloquear a ativação:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
pip install -r requirements.txt
```

> No Windows, o alias `python` fora do venv abre a Microsoft Store. Ative o venv antes.
> Se aparecerem caracteres estranhos no terminal (acentos), rode `$env:PYTHONUTF8=1`.
> Se o `pip install` falhar com `OSError: [Errno 2] No such file or directory` num arquivo do `torch`, o caminho passou do limite de 260 caracteres do Windows. Clone o projeto numa pasta de caminho curto (ex.: `C:\Users\<voce>\concrete-mlflow`) ou habilite caminhos longos no Windows (`LongPathsEnabled`).

## Estrutura

```
concrete-mlflow/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── common.py      # dados, split, modelo MLP, métricas, seed
│   ├── train.py       # treino + validação, registrado no MLflow (sem teste)
│   └── evaluate.py    # avalia o teste UMA vez na run escolhida
├── configs/
│   ├── run_a.yaml
│   ├── run_b.yaml
│   └── run_c.yaml
├── run_all.ps1        # roda as 3 runs
└── evidencias/        # prints do MLflow UI
```

## Experimentos

Base comum: Adam, treino full-batch, 1000 épocas, 1 camada oculta de 16 neurônios com ReLU, seed 33, split 60/20/20.

| Run | lr | L2 (weight_decay) | Pergunta |
|---|---|---|---|
| A, referência | 0.01 | 0 | Qual é a linha de base? |
| B, taxa menor | 0.001 | 0 | Uma taxa menor converge melhor ou fica subtreinada em 1000 épocas? |
| C, com L2 | 0.01 | 1e-5 | O L2 reduz o gap entre a loss de treino e a de validação? |

## Como rodar

Todos os comandos são executados **na raiz do projeto** (é lá que o `mlflow.db` é criado):

```powershell
.\run_all.ps1
# ou uma run de cada vez:
python src/train.py --config configs/run_a.yaml
```

Cada run registra no MLflow (`sqlite:///mlflow.db`, experimento `concrete-mlp`):

- **Parâmetros:** lr, weight_decay, epochs, hidden_sizes, seed, otimizador, loss, split, device.
- **Métricas por época:** `train_loss`, `val_loss` (MSE na escala padronizada), `val_rmse`, `val_mae`, `val_r2` (em MPa).
- **Métricas finais:** `final_val_rmse`, `final_val_mae`, `final_val_r2`.
- **Tags de versão:** `git_commit` (hash do commit), `git_dirty` (se havia alterações não commitadas, com a lista em `git_dirty_files`), `python_version`, `torch_version`, `mlflow_version` e `sklearn_version`.
- **Artefatos:** `config.json`, `preprocessing.json` (médias/desvios do scaler e tamanhos do split), `requirements.txt` e `model/model.pt`.
- **Trace:** um span `pipeline` com os filhos `preparar`, `treinar` e `validar`.

### MLflow UI

Em outro terminal, com o venv ativo e na pasta do projeto:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abra http://localhost:5000 (se a porta estiver ocupada, use `--port 5001`).

### Avaliação no teste (uma única vez)

Depois de escolher a run pela validação:

```powershell
python src/evaluate.py --run_id <RUN_ID_ESCOLHIDA>
```

O script recarrega o modelo e o pré-processamento da run, refaz o mesmo split e registra `test_rmse`, `test_mae` e `test_r2` na própria run. Ele **recusa** uma segunda execução na mesma run: o teste só é olhado uma vez.

## Decisões de projeto

- **Scaler ajustado só no treino.** O script original ajustava o `StandardScaler` com todos os dados, o que vaza informação da validação e do teste para o treino. Aqui, o scaler das features e o do alvo usam `fit` apenas em `X_train`/`y_train` e só `transform` nos outros conjuntos. Os parâmetros ficam salvos em `preprocessing.json`.
- **Alvo padronizado.** A rede treina com o alvo padronizado (`MSELoss` na escala z), e as métricas em MPa são calculadas após desfazer a padronização.
- **Seed 33 e split 60/20/20** (618 treino / 206 validação / 206 teste), como no script da aula. O split usa `train_test_split` com `random_state=33` fixo, então os três conjuntos são sempre os mesmos em todas as runs e no `evaluate.py`.
- **Sem divisão estratificada.** A estratificação vista nos slides vale para classificação: ela mantém a proporção de cada classe nos conjuntos. Aqui o alvo (resistência em MPa) é contínuo e não tem classes, então a divisão é aleatória simples.
- **Versão do código e do ambiente em cada run.** O `train.py` grava como tags o commit do git, se havia alterações não commitadas e as versões de Python, torch, MLflow e scikit-learn, além do `requirements.txt` como artefato. As 3 runs registradas foram treinadas antes de a pasta virar um repositório git. Por isso, as tags delas (commit `7dd8ba6`) foram adicionadas depois, com `MlflowClient.set_tag`, e marcadas com `git_commit_retroativo`. Antes disso, conferi que `common.py`, `train.py`, `configs/` e `requirements.txt` desse commit são idênticos aos usados no treino.
- **Teste reservado.** O `train.py` não calcula nada no teste. A escolha da run é feita só com a validação, e o teste é avaliado uma vez pelo `evaluate.py`.
- **Seed sem o módulo `random`.** O `set_seed` semeia NumPy e PyTorch, mas não o `random` global do Python. Na primeira execução, `random.seed(33)` fazia o OpenTelemetry (usado pelo tracing do MLflow) gerar **o mesmo trace ID** em todas as runs. O trace da run B colidia com o da A e se perdia. O pipeline não usa `random`, então a reprodutibilidade não muda: as métricas saíram idênticas antes e depois da correção.

## Resultados

Execução real em CPU. Valores de validação na época 1000 (206 amostras):

| Run | lr | L2 | val RMSE (MPa) | val MAE (MPa) | val R² | gap val_loss − train_loss (ép. 1000) |
|---|---|---|---|---|---|---|
| A-referencia | 0.01 | 0 | 5.668 | 4.105 | 0.890 | 0.062 |
| B-taxa-menor | 0.001 | 0 | 6.431 | 4.788 | 0.859 | 0.020 |
| C-com-L2 | 0.01 | 1e-5 | **5.540** | **4.088** | **0.895** | 0.058 |

Respostas às perguntas:

- **A (linha de base):** val RMSE de 5.67 MPa e R² de 0.89.
- **B (taxa menor):** ficou **subtreinada**. Em 1000 épocas ela ainda está melhorando (val RMSE 6.62 → 6.43 MPa entre as épocas 900 e 1000; train_loss 0.148 → 0.136). O gap pequeno vem do subtreino, não de uma generalização melhor.
- **C (com L2):** o efeito do L2 de 1e-5 é **pequeno**. O gap caiu de 0.062 para 0.058, e o val RMSE ficou 0.13 MPa menor que o da A. A curva da C também ficou mais estável no fim (≈5.54 MPa entre as épocas 800 e 1000).

**Escolha:** run **C-com-L2**, por ter o menor RMSE de validação. A diferença para a A (0.13 MPa) é pequena demais para ser conclusiva com 206 amostras, então a curva mais estável serviu como critério de desempate.

Teste (avaliado uma única vez, run C, 206 amostras):

| Run | test RMSE (MPa) | test MAE (MPa) | test R² |
|---|---|---|---|
| C-com-L2 | 5.15 | 3.81 | 0.913 |

O teste saiu um pouco melhor que a validação (5.15 contra 5.54 MPa). Com 206 amostras em cada conjunto, essa diferença é compatível com a variação natural do split, e não indica vazamento: o scaler e a escolha da run não usaram o teste.

## Limitações e próximos passos

- Uma única seed e um único split. Repetir com várias seeds (ou validação cruzada) daria intervalos de confiança para comparar A e C.
- A run B precisaria de mais épocas para uma comparação justa de taxa de aprendizado.
- O modelo final é o da época 1000. Early stopping pela validação (a C teve o melhor val RMSE, 5.45 MPa, na época 508) seria um próximo passo.
- Arquitetura pequena (1 camada de 16 neurônios). Vale testar mais neurônios ou camadas.
