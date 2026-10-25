# 🌽 AgroSmart Core — Monitoramento Fitossanitário Inteligente

Plataforma integrada de visão computacional e engenharia de dados voltada ao monitoramento em tempo real da saúde da cultura do milho. O sistema processa fluxos contínuos de imagens captadas em campo, realiza diagnósticos fitossanitários com explicabilidade visual via **Grad-CAM++**, armazena as inferências de forma relacional e automatiza alertas agronômicos por e-mail quando limiares críticos de infestação são atingidos.

---

## 📌 Funcionalidades Principais

* **Simulação Contínua de Sensores:** Emulação do envio de imagens de campo por dispositivos IoT/câmeras em intervalos ajustáveis, com sorteio de amostras e metadados contextuais (talhão, fazenda e data).
* **Classificação Multiclasse com PyTorch:** Modelo convolucional **DenseNet121** pré-treinado para diferenciar folhas sadias (*Healthy*) de patologias como Ferrugem Comum (*Common Rust*), Mancha de Cercospora (*Gray Leaf Spot*) e Helmintosporiose do Milho (*Northern Leaf Blight*).
* **Explicabilidade Visual (Grad-CAM++):** Destaque interpretável das regiões causadoras da doença sobre a folha através de mapas de calor e contornos adaptativos.
* **Persistência Relacional com Deduplicação:** Banco de dados **SQLite** com validação de hash criptográfico **SHA-256** para evitar reprocessamento desnecessário.
* **Painel Analítico e Temporal:** Métricas executivas consolidadas, distribuição geral de patologias e análise cronológica para identificação de sazonalidade e picos epidemiológicos.
* **Alertas Automatizados via SMTP:** Gatilho com regras de negócio (> 2 focos por fazenda) e controle antispam por sessão para envio imediato de e-mails técnicos.
* **Inspeção Visual & Exportação:** Galeria com filtros dinâmicos de imagens processadas e download dos relatórios tabulares em **CSV**.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Framework Web:** Streamlit
* **Visão Computacional & Deep Learning:** PyTorch, Torchvision, Pillow (PIL), NumPy, OpenCV
* **Explicabilidade (XAI):** Grad-CAM++
* **Visualização de Dados:** Plotly Express, Pandas
* **Banco de Dados:** SQLite3
* **Comunicação & Mensageria:** `smtplib`, `email.mime`
* **Virtualização (Opcional):** Docker / Docker Compose

---

## 📂 Estrutura de Diretórios

```plaintext
corn_predictions/
├── app_dashboard.py            # Aplicação principal Streamlit
├── docker-compose.yml          # Orquestração de contêineres Docker
├── Dockerfile                  # Imagem base de execução
├── requirements.txt            # Dependências do projeto
├── results.sqlite              # Banco de dados local de inferências
├── dataset_teste/              # Banco de imagens brutas de teste para simulação
├── input_images/               # Fila ativa de processamento contínuo
├── annotated_images/           # Imagens resultantes com máscaras Grad-CAM++
├── models/
│   ├── maize_densenet121_state_dict.pt  # Pesos treinados do modelo
│   └── maize_densenet121_meta.json      # Metadados e índices das classes
└── mvp/
    ├── annotate.py             # Rotinas de desenho de anotações e contornos
    ├── campp.py                # Implementação do Grad-CAM++
    ├── db.py                   # Operações de inicialização e inserção no SQLite
    ├── hashing.py              # Cálculo de hash SHA-256
    ├── masking.py              # Segmentação e limiares de máscaras adaptativas
    └── model.py                # Carregamento e inferência da DenseNet121

## 🚀 Como Executar o Projeto (Via Arquivo .ZIP)

### 1. Pré-requisitos
* **Python 3.10 ou superior** instalado ([Download oficial do Python](https://www.python.org/downloads/)).
  > ⚠️ Durante a instalação no Windows, lembre-se de marcar a opção **"Add Python to PATH"**.

---

### 2. Preparação do Ambiente

1. Extraia o conteúdo do arquivo `.zip` em uma pasta de sua preferência.
2. Certifique-se de que a estrutura básica contém os seguintes diretórios:
   * `models/`: contendo os pesos (`maize_densenet121_state_dict.pt`) e os metadados (`maize_densenet121_meta.json`).
   * `dataset_teste/`: contendo as imagens de folhas de milho para teste da simulação.
   * `mvp/`: contendo os scripts auxiliares do pipeline.

---

### 3. Instalação das Dependências e execução

Abra o terminal o projeto dentro da pasta descompactada e execute:

pip install -r requirements.txt
python -m streamlit run app_dashboard.py