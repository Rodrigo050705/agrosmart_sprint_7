# 🌽 AgroSmart Core Hub

Sistema inteligente para monitoramento contínuo e diagnóstico fitossanitário em lavouras de milho. A solução integra modelos de visão computacional profunda com técnicas de explicabilidade visual, mineração de processos (*Process Mining*) para observabilidade do ciclo de vida dos dados, conteinerização resiliente e esteira automatizada de DevSecOps.

---

## 📌 Visão Geral do Projeto

O **AgroSmart Core** processa imagens capturadas por sensores de campo, classifica a sanidade foliar e destaca visualmente as regiões patológicas em folhas infectadas. O sistema foi desenvolvido para operar de forma autônoma, registrando métricas em tempo real e auditando os gargalos de execução através de telemetria e mineração de processos.

### Principais Funcionalidades

* **Diagnóstico com Deep Learning:** Classificação de patologias em milho utilizando a arquitetura convolucional **DenseNet-121**.
* **Explicabilidade Visual (Grad-CAM++):** Geração automática de mapas de calor e máscaras de segmentação sobrepondo as lesões ativas quando uma patologia é detectada.
* **Deduplicação de Imagens:** Mecanismo criptográfico com **SHA-256** para evitar reprocessamento desnecessário de amostras idênticas.
* **Mineração de Processos (Process Mining):** Extração de *Event Log* estruturado (padrão IEEE XES/CSV) mapeando etapas, latência e gargalos computacionais.
* **Alertas Epidemiológicos Automatizados:** Notificações via SMTP quando um limite crítico de focos de infecção é atingido em determinado talhão.
* **Painel Interativo Streamlit:** Dashboard com monitoramento temporal, galeria visual, exportação de dados em CSV e console de telemetria.

---

## 🛠️ Arquitetura e Tecnologias

* **Linguagem:** Python 3.10
* **Framework Web / UI:** Streamlit
* **Visão Computacional & IA:** PyTorch, Torchvision, OpenCV, Pillow, Grad-CAM++
* **Análise de Dados:** Pandas, NumPy, Plotly Express
* **Persistência de Dados:** SQLite3
* **Conteinerização:** Docker e Docker Compose
* **Qualidade e Segurança (CI/CD):** GitHub Actions, Flake8, Bandit (SAST) e Trivy (Scanner de Vulnerabilidades de Contêiner)

---

## 📂 Estrutura de Diretórios

```text
├── .github/
│   └── workflows/
│       └── pipeline.yml          # Pipeline CI/CD (Linting, SAST, Build e Trivy)
├── dataset_teste/                # Amostras para simulação contínua
├── models/                       # Pesos e metadados da DenseNet-121
│   ├── maize_densenet121_meta.json
│   └── maize_densenet121_state_dict.pt
├── mvp/                          # Módulos de visão computacional, hash e banco
│   ├── annotate.py
│   ├── campp.py
│   ├── db.py
│   ├── hashing.py
│   ├── masking.py
│   └── model.py
├── app_dashboard.py              # Aplicação principal Streamlit
├── docker-compose.yml            # Orquestração do contêiner e volumes
├── Dockerfile                    # Receita da imagem do contêiner com healthcheck
├── requirements.txt              # Dependências do projeto
└── results.sqlite                # Base de dados relacional persistida

```

---

## 🚀 Como Executar o Projeto com Docker (Recomendado)

### Pré-requisitos

* [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado e ativo.
* [Git](https://git-scm.com/) configurado.

### 1. Clonar o Repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd agrosmart-core

```

### 2. Inicializar o Arquivo de Banco de Dados

Para evitar que o Docker monte o arquivo SQLite como um diretório no host Windows/Linux, crie o arquivo em branco antes de subir o serviço:

* **PowerShell (Windows):**
```powershell
if (!(Test-Path "results.sqlite")) { New-Item -ItemType File -Name "results.sqlite" }

```


* **Linux / macOS:**
```bash
touch results.sqlite

```



### 3. Construir e Iniciar o Contêiner

Execute a construção e a inicialização em segundo plano:

```bash
docker compose up --build -d

```

### 4. Verificar o Status e a Integridade (Healthcheck)

Aguarde cerca de 30 segundos e confira se o serviço alcançou o status de integridade esperado:

```bash
docker compose ps

```

(O status deve exibir `Up` acompanhado de `(healthy)`).

### 5. Acessar a Aplicação

Abra no navegador:

```text
http://localhost:8501

```

---

## 🧪 Roteiro de Testes e Validação Funcional

### 1. Teste da Simulação Contínua

1. Acesse o menu lateral do dashboard.
2. Na seção **Simulação de Sensores**, clique no botão **▶ Ligar**.
3. O sistema passará a simular o envio contínuo de amostras do diretório `dataset_teste/`.
4. Navegue pela aba **Painel Analítico & Temporal** para acompanhar os gráficos de sazonalidade e índices de sanidade sendo alimentados em tempo real.

### 2. Teste da Mineração de Processos (Process Mining)

1. Com a simulação ativa por alguns ciclos, clique na aba **🔍 Mineração de Processos**.
2. Avalie as métricas operacionais consolidadas:
* **Casos Processados:** Volume de amostras auditadas.
* **Descoberta de Gargalos:** O gráfico horizontal evidenciará o tempo despendido em cada microetapa (destaque para a latência da explicabilidade Grad-CAM++ vs. deduplicação de hash).
* **Variantes de Processo:** Tabela de rotas percorridas pelas amostras (*happy path* com Grad-CAM++, *fast path* de folhas sadias e amostras descartadas por deduplicação).


3. Clique em **📥 Baixar Event Log Estruturado (CSV)** para obter a base pronta para ferramentas de processo (Celonis, Disco, ProM, pm4py).

### 3. Teste da Galeria de Explicabilidade

1. Acesse a aba **Galeria Visual (Grad-CAM++)**.
2. Filtre por diagnóstico e fazenda para inspecionar os mapas de atenção gerados pelo modelo sobre as folhas infectadas.

---

## 🔒 Testes Locais de Qualidade e Segurança (DevSecOps)

Caso queira validar as etapas de auditoria localmente antes de enviar commits:

### 1. Linting e Validação Sintática (Flake8)

```powershell
python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=dataset_teste,annotated_images,input_images

```

(A saída esperada é `0` erros).

### 2. Auditoria Estática de Segurança (Bandit - SAST)

```powershell
python -m bandit -r mvp/ app_dashboard.py -ll --exit-zero

```

(Inspeciona potenciais brechas de segurança no código-fonte).

### 3. Análise de Vulnerabilidades em Contêineres (Trivy)

```powershell
docker run --rm -v //var/run/docker.sock:/var/run/docker.sock aquasec/trivy:latest image agrosmart-core:latest

```

(Gera o relatório com inventário de pacotes e CVEs identificadas no contêiner).

---

## ⚙️ Esteira de CI/CD (GitHub Actions)

A cada `push` ou `pull request` na branch `main`, o workflow `.github/workflows/pipeline.yml` dispara automaticamente:

1. **Job 1 (Qualidade & SAST):** Validação de sintaxe com **Flake8** e análise de riscos com **Bandit**.


2. **Job 2 (Build & Trivy Scan):** Compilação automatizada da imagem Docker e varredura de vulnerabilidades de contêiner com **Trivy**.