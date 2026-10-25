import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MVP_DIR = ROOT_DIR / "mvp"
if str(MVP_DIR) not in sys.path:
    sys.path.append(str(MVP_DIR))

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import torch
from PIL import Image
from datetime import datetime, timezone, timedelta
import random
import json
import time
import shutil
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import plotly.express as px
import plotly.graph_objects as go

# Imports dos módulos de IA do projeto
from mvp.model import load_meta, build_preprocess, load_densenet121_from_state_dict, predict_softmax
from mvp.campp import GradCAMPlusPlus, densenet_target_layer
from mvp.masking import cam_to_mask_adaptive, cam_is_flat
from mvp.annotate import annotate_with_mask, draw_label
from mvp.hashing import sha256_file
from mvp.db import init_db, insert_result, has_hash

# ---------------- CONFIGURAÇÃO DA PÁGINA ----------------
st.set_page_config(page_title="AgroSmart Core Hub", page_icon="🌽", layout="wide")

MODEL_PATH = ROOT_DIR / "models" / "maize_densenet121_state_dict.pt"
META_PATH = ROOT_DIR / "models" / "maize_densenet121_meta.json"
DB_PATH = ROOT_DIR / "results.sqlite"
DATASET_TESTE = ROOT_DIR / "dataset_teste"
INPUT_DIR = ROOT_DIR / "input_images"
OUTPUT_DIR = ROOT_DIR / "annotated_images"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_TESTE.mkdir(parents=True, exist_ok=True)

# ---------------- BANCO DE DADOS & EVENT LOG ----------------
def init_all_databases():
    """Inicializa as tabelas padrão e a tabela dedicada para Process Mining."""
    c = init_db(DB_PATH)
    with sqlite3.connect(DB_PATH) as conn_local:
        conn_local.execute("""
            CREATE TABLE IF NOT EXISTS event_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                activity TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                duration_ms REAL NOT NULL,
                resource TEXT NOT NULL,
                status TEXT NOT NULL
            );
        """)
        conn_local.commit()
    return c

conn = init_all_databases()

def registrar_evento(case_id: str, activity: str, duration_ms: float, resource: str, status: str = "COMPLETED"):
    agora = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn_local:
        conn_local.execute(
            "INSERT INTO event_logs (case_id, activity, timestamp, duration_ms, resource, status) VALUES (?, ?, ?, ?, ?, ?)",
            (case_id, activity, agora, round(duration_ms, 2), resource, status)
        )
        conn_local.commit()

# ---------------- ESTADO DA SESSÃO ----------------
if "simulando" not in st.session_state:
    st.session_state["simulando"] = False
if "alertas_enviados" not in st.session_state:
    st.session_state["alertas_enviados"] = set()
if "telemetria" not in st.session_state:
    st.session_state["telemetria"] = {
        "ultima_latencia_ms": 0,
        "total_ciclos": 0,
        "ultimo_evento": "Aguardando início...",
        "logs": []
    }

def adicionar_log(tipo: str, mensagem: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entrada = f"[{timestamp}] [{tipo}] {mensagem}"
    st.session_state["telemetria"]["logs"].append(entrada)
    if len(st.session_state["telemetria"]["logs"]) > 40:
        st.session_state["telemetria"]["logs"].pop(0)
    st.session_state["telemetria"]["ultimo_evento"] = entrada

# ---------------- CARREGAMENTO DO MODELO ----------------
@st.cache_resource
def carregar_pipeline():
    meta = load_meta(META_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_densenet121_from_state_dict(MODEL_PATH, meta, device)
    model.eval()
    tfm = build_preprocess(meta)
    campp = GradCAMPlusPlus(model, densenet_target_layer(model))
    return meta, model, tfm, campp, device

meta, model, tfm, campp, device = carregar_pipeline()
classes = meta["classes"]

# ---------------- FUNÇÕES DE APOIO ----------------
def gerar_data_aleatoria():
    start = datetime(2026, 1, 1)
    end = datetime(2026, 12, 31)
    dias = random.randint(0, (end - start).days)
    return (start + timedelta(days=dias)).strftime("%Y-%m-%d")

def disparar_email_alerta(destinatario: str, doenca: str, fazenda: str, qtd_focos: int, remetente: str, senha_app: str):
    if not destinatario or not remetente or not senha_app:
        return False, "Credenciais ausentes."
    assunto = f"🚨 [AgroSmart] Alerta Crítico: {doenca} na {fazenda}"
    corpo = f"Ocorrências confirmadas: {qtd_focos} focos. Priorizar inspeção presencial imediata."
    msg = MIMEMultipart()
    msg["From"] = remetente
    msg["To"] = destinatario
    msg["Subject"] = assunto
    msg.attach(MIMEText(corpo, "plain", "utf-8"))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(remetente, senha_app)
            server.send_message(msg)
        return True, "E-mail enviado!"
    except Exception as e:
        return False, str(e)

def processar_ciclo_sensor():
    """Executa um ciclo único rastreando os eventos de processo."""
    imagens = [p for p in DATASET_TESTE.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if not imagens:
        adicionar_log("AVISO", "dataset_teste vazio.")
        return False

    case_id = f"CASE_{int(time.time()*1000)}"
    inicio_total = time.perf_counter()

    # 1. Atividade: Aquisição da Imagem (Sensor)
    t0 = time.perf_counter()
    sorteada = random.choice(imagens)
    destino = INPUT_DIR / f"{case_id}.jpg"
    shutil.copy(sorteada, destino)
    fazenda = random.choice(["Fazenda Sol Nascente", "Fazenda Capão Rico", "Fazenda Recanto"])
    talhao = f"PLT{random.randint(1, 15)}"
    data_sorteada = gerar_data_aleatoria()
    registrar_evento(case_id, "1_Aquisicao_Sensor", (time.perf_counter() - t0) * 1000, "Sensor_Edge")

    # 2. Atividade: Deduplicação SHA-256
    t0 = time.perf_counter()
    sha = sha256_file(destino)
    with sqlite3.connect(DB_PATH) as c_local:
        duplicado = has_hash(c_local, sha)
    dur_sha = (time.perf_counter() - t0) * 1000

    if duplicado:
        registrar_evento(case_id, "2_Deduplicacao_SHA", dur_sha, "Hash_Engine", status="DUPLICATED")
        adicionar_log("INFO", f"Amostra duplicada ({sha[:8]}). Ciclo abreviado.")
        return True
    
    registrar_evento(case_id, "2_Deduplicacao_SHA", dur_sha, "Hash_Engine")

    # 3. Atividade: Inferência DenseNet121
    t0 = time.perf_counter()
    image = Image.open(destino).convert("RGB")
    rgb = np.array(image)
    H, W = rgb.shape[:2]
    x = tfm(image).unsqueeze(0).to(device)
    probs_t = predict_softmax(model, x)
    probs = probs_t.detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    pred_prob = float(probs[pred_idx])
    dur_inf = (time.perf_counter() - t0) * 1000
    registrar_evento(case_id, "3_Inferencia_DenseNet", dur_inf, "PyTorch_Model")

    # 4. Atividade: Grad-CAM++ (apenas patologias)
    mask = np.zeros((H, W), dtype=np.uint8)
    notes = None
    if pred_label.strip().lower() != "healthy":
        t0 = time.perf_counter()
        x2 = x.clone().requires_grad_(True)
        cam = campp(x2, class_idx=pred_idx).detach().cpu().numpy().astype(np.float32)
        if cam_is_flat(cam):
            notes = "CAM flat"
        else:
            mask = cam_to_mask_adaptive(cam, H, W, q_start=0.92)
        annot = annotate_with_mask(rgb, mask, highlight_strength=0.55, draw_contour=True)
        annot = draw_label(annot, f"{pred_label} ({pred_prob:.2f})")
        annot_file = OUTPUT_DIR / f"annot_{destino.stem}.png"
        Image.fromarray(annot).save(annot_file)
        dur_cam = (time.perf_counter() - t0) * 1000
        registrar_evento(case_id, "4_Explicabilidade_GradCAM", dur_cam, "GradCAM_Module")
    else:
        annot_file = OUTPUT_DIR / f"annot_{destino.stem}.png"
        annot = draw_label(rgb, f"{pred_label} ({pred_prob:.2f})")
        Image.fromarray(annot).save(annot_file)

    # 5. Atividade: Persistência no SQLite
    t0 = time.perf_counter()
    created_at = datetime.now(timezone.utc).isoformat()
    probs_json = json.dumps({classes[i]: float(probs[i]) for i in range(len(classes))})
    with sqlite3.connect(DB_PATH) as c_local:
        insert_result(c_local, created_at, data_sorteada, fazenda, talhao, str(destino), sha,
                      pred_label, pred_prob, probs_json, str(annot_file), notes)
    dur_sql = (time.perf_counter() - t0) * 1000
    registrar_evento(case_id, "5_Persistencia_Dados", dur_sql, "SQLite_Database")

    # Atualiza telemetria geral
    latencia_total = (time.perf_counter() - inicio_total) * 1000
    st.session_state["telemetria"]["ultima_latencia_ms"] = round(latencia_total, 1)
    st.session_state["telemetria"]["total_ciclos"] += 1
    adicionar_log("SUCESSO", f"Ciclo {case_id} finalizado em {latencia_total:.0f}ms | {pred_label}")
    return True

# ---------------- BARRA LATERAL ----------------
st.sidebar.title("🌽 AgroSmart Core")
st.sidebar.caption("Pipeline Fitossanitário & Process Mining")

# Registra troca de tela para sincronização limpa
if "aba_atual" not in st.session_state:
    st.session_state["aba_atual"] = "Painel Analítico & Temporal"

menu = st.sidebar.radio(
    "Navegação",
    [
        "Painel Analítico & Temporal",
        "🔍 Mineração de Processos",
        "Galeria Visual (Grad-CAM++)",
        "Visualizador de Dados (CSV)",
        "🖥️ Telemetria & Logs"
    ],
    index=["Painel Analítico & Temporal", "🔍 Mineração de Processos", "Galeria Visual (Grad-CAM++)", "Visualizador de Dados (CSV)", "🖥️ Telemetria & Logs"].index(st.session_state.get("aba_atual", "Painel Analítico & Temporal"))
)

# Detectou que o usuário acabou de trocar de aba
trocou_de_aba = (menu != st.session_state["aba_atual"])
if trocou_de_aba:
    st.session_state["aba_atual"] = menu

st.sidebar.divider()
st.sidebar.subheader("📡 Simulação de Sensores")
col_b1, col_b2 = st.sidebar.columns(2)
if col_b1.button("▶ Ligar", type="primary", use_container_width=True):
    st.session_state["simulando"] = True
    adicionar_log("STATUS", "Simulador iniciado.")
if col_b2.button("⏹ Desligar", use_container_width=True):
    st.session_state["simulando"] = False
    adicionar_log("STATUS", "Simulador desligado.")

if st.session_state["simulando"]:
    st.sidebar.success("🟢 Sensores ativos")
else:
    st.sidebar.info("⚪ Sensores inativos")

velocidade = st.sidebar.slider("Intervalo (segundos)", 1, 8, 3)
auto_refresh = st.sidebar.checkbox("Atualização Automática", value=True)
refresh_interval = st.sidebar.number_input("Taxa de atualização (seg)", 2, 20, 3)

st.sidebar.divider()
st.sidebar.subheader("📬 Alertas por E-mail")
email_destinatario = st.sidebar.text_input("Destinatário", placeholder="agronomo@agro.com")
with st.sidebar.expander("⚙️ Credenciais SMTP"):
    email_remetente = st.text_input("Gmail Remetente", placeholder="seu_email@gmail.com")
    senha_remetente = st.text_input("Senha de App", type="password")

st.sidebar.divider()
if st.sidebar.button("🗑️ Reiniciar Banco do Zero", use_container_width=True):
    with sqlite3.connect(DB_PATH) as c:
        c.execute("DELETE FROM inferences;")
        c.execute("DELETE FROM event_logs;")
        c.commit()
    for f in OUTPUT_DIR.glob("*"):
        if f.is_file(): f.unlink()
    for f in INPUT_DIR.glob("*"):
        if f.is_file(): f.unlink()
    st.session_state["alertas_enviados"] = set()
    st.session_state["telemetria"]["logs"] = []
    st.session_state["telemetria"]["total_ciclos"] = 0
    adicionar_log("RESET", "Dados zerados.")
    st.sidebar.success("Ambiente limpo!")
    time.sleep(1)
    st.rerun()

# Executa o ciclo se a simulação estiver ligada
if st.session_state["simulando"]:
    processar_ciclo_sensor()

# ---------------- CONTÊINER MESTRE (EVITA TELAS SOBREPOSTAS) ----------------
# st.empty() força a limpeza completa da tela antes de desenhar a aba selecionada
main_placeholder = st.empty()

with main_placeholder.container():
    # ---------------- TELA 1: PAINEL ANALÍTICO ----------------
    if menu == "Painel Analítico & Temporal":
        st.header("📊 Inteligência Fitossanitária & Monitoramento Temporal")
        df = pd.read_sql_query("SELECT * FROM inferences", conn)

        if df.empty:
            st.info("Aguardando dados... Ative os sensores na barra lateral.")
        else:
            df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
            df["mes_ano"] = df["date_parsed"].dt.strftime("%Y-%m")

            total = len(df)
            doentes = df[df["pred_label"] != "Healthy"]
            taxa_saude = ((total - len(doentes)) / total) * 100 if total > 0 else 0
            talhao_critico = doentes["plot"].mode()[0] if not doentes.empty else "Nenhum"

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Amostras Avaliadas", total)
            m2.metric("Índice de Sanidade", f"{taxa_saude:.1f}%")
            m3.metric("Focos de Doenças", len(doentes))
            m4.metric("Talhão Mais Afetado", talhao_critico)

            st.divider()

            if not doentes.empty:
                alertas = doentes.groupby(["farm_name", "pred_label"]).size().reset_index(name="focos")
                criticos = alertas[alertas["focos"] > 2]
                if not criticos.empty:
                    for _, row in criticos.iterrows():
                        chave = f"{row['farm_name']}_{row['pred_label']}"
                        st.error(f"🚨 **Alerta de Infestação:** {row['pred_label']} na {row['farm_name']} com **{row['focos']} focos detetados**.")
                        st.info("💡 **Ação Recomendada:** Priorizar inspeção presencial imediata nos talhões afetados.")

                        if chave not in st.session_state["alertas_enviados"]:
                            if email_destinatario and email_remetente and senha_remetente:
                                t_smtp = time.perf_counter()
                                sucesso, msg = disparar_email_alerta(email_destinatario, row["pred_label"], row["farm_name"], row["focos"], email_remetente, senha_remetente)
                                dur_smtp = (time.perf_counter() - t_smtp) * 1000
                                registrar_evento(f"ALERT_{int(time.time())}", "6_Disparo_Alerta_SMTP", dur_smtp, "SMTP_Service", status="SUCCESS" if sucesso else "FAILED")
                                if sucesso:
                                    adicionar_log("SMTP", f"Alerta enviado para {email_destinatario}.")
                                    st.session_state["alertas_enviados"].add(chave)
                else:
                    st.success("Nenhuma patologia excedeu o limiar de risco (>2 focos).")

            st.divider()
            st.subheader("📈 Sazonalidade das Doenças")
            df_temporal = df[df["pred_label"] != "Healthy"].dropna(subset=["mes_ano"])
            if not df_temporal.empty:
                contagem = df_temporal.groupby(["mes_ano", "pred_label"]).size().reset_index(name="quantidade").sort_values("mes_ano")
                fig_bar_tempo = px.bar(
                    contagem, x="mes_ano", y="quantidade", color="pred_label", barmode="group",
                    labels={"mes_ano": "Mês / Ano", "quantidade": "Ocorrências", "pred_label": "Patologia"}
                )
                st.plotly_chart(fig_bar_tempo, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Distribuição Geral")
                st.plotly_chart(px.pie(df, names="pred_label", hole=0.45), use_container_width=True)
            with c2:
                st.subheader("Ocorrências por Fazenda")
                st.plotly_chart(px.bar(df, x="farm_name", color="pred_label", barmode="group"), use_container_width=True)

    # ---------------- TELA 2: MINERAÇÃO DE PROCESSOS ----------------
    elif menu == "🔍 Mineração de Processos":
        st.header("🔍 Mineração de Processos & Desempenho Operacional")
        st.markdown("Auditoria do fluxo do pipeline fitossanitário com base nos registos do **Event Log**.")

        df_events = pd.read_sql_query("SELECT * FROM event_logs", conn)

        if df_events.empty:
            st.info("Nenhum evento registrado ainda. Ative o simulador para gerar rastros de processo.")
        else:
            # 1. Métricas Globais do Processo
            casos_totais = df_events["case_id"].nunique()
            tempo_medio_caso = df_events.groupby("case_id")["duration_ms"].sum().mean()
            ativ_mais_lenta = df_events.groupby("activity")["duration_ms"].mean().idxmax()
            pior_latencia = df_events.groupby("activity")["duration_ms"].mean().max()

            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("Casos Processados", casos_totais)
            pm2.metric("Tempo Médio Total", f"{tempo_medio_caso:.1f} ms")
            pm3.metric("Gargalo Identificado", ativ_mais_lenta.split("_", 1)[-1])
            pm4.metric("Duração do Gargalo", f"{pior_latencia:.1f} ms")

            st.divider()

            # 2. Análise de Gargalos por Atividade
            st.subheader("⏱️ Tempo Médio por Etapa do Fluxo (Descoberta de Gargalos)")
            stats_ativ = df_events.groupby("activity").agg(
                tempo_medio=("duration_ms", "mean"),
                execucoes=("id", "count")
            ).reset_index()

            fig_gargalos = px.bar(
                stats_ativ,
                x="tempo_medio",
                y="activity",
                orientation="h",
                text="tempo_medio",
                labels={"tempo_medio": "Tempo Médio (ms)", "activity": "Etapa do Processo"},
                color="tempo_medio",
                color_continuous_scale="Reds"
            )
            fig_gargalos.update_traces(texttemplate='%{text:.1f} ms', textposition='outside')
            st.plotly_chart(fig_gargalos, use_container_width=True)

            # 3. Variantes do Processo (Happy Path vs Desvios)
            st.subheader("🔀 Análise de Variantes e Caminhos do Processo")
            caminhos = df_events.groupby("case_id")["activity"].apply(lambda x: " ➔ ".join(x)).value_counts().reset_index()
            caminhos.columns = ["Fluxo / Rota de Execução", "Quantidade de Casos"]
            st.dataframe(caminhos, use_container_width=True)

            # 4. Tabela Estruturada do Event Log com Exportação
            st.subheader("📄 Registos do Event Log (Padrão IEEE XES / CSV)")
            st.dataframe(df_events.tail(15).iloc[::-1], use_container_width=True)

            csv_eventos = df_events.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Baixar Event Log Estruturado (CSV)",
                data=csv_eventos,
                file_name=f"event_log_agrosmart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    # ---------------- TELA 3: GALERIA VISUAL ----------------
    elif menu == "Galeria Visual (Grad-CAM++)":
        st.header("🖼️ Inspeção Visual Detalhada (Grad-CAM++)")
        df = pd.read_sql_query("SELECT * FROM inferences", conn)

        if df.empty:
            st.info("Nenhuma imagem processada até o momento.")
        else:
            cg1, cg2 = st.columns(2)
            f_diag = cg1.selectbox("Filtrar por Diagnóstico", ["Todos"] + list(df["pred_label"].unique()))
            f_faz = cg2.selectbox("Filtrar por Fazenda", ["Todas"] + list(df["farm_name"].unique()))

            df_g = df.copy()
            if f_diag != "Todos": df_g = df_g[df_g["pred_label"] == f_diag]
            if f_faz != "Todas": df_g = df_g[df_g["farm_name"] == f_faz]

            amostras = df_g.tail(6).iloc[::-1]
            cols = st.columns(3)
            for idx, (_, item) in enumerate(amostras.iterrows()):
                caminho_anot = Path(item["annotated_path"])
                with cols[idx % 3]:
                    if caminho_anot.exists():
                        st.image(str(caminho_anot), caption=f"{item['farm_name']} | {item['plot']} - {item['pred_label']} ({item['pred_prob']*100:.1f}%)", use_container_width=True)

    # ---------------- TELA 4: CSV ----------------
    elif menu == "Visualizador de Dados (CSV)":
        st.header("📁 Base de Dados Estruturada e Exportação")
        df = pd.read_sql_query("SELECT id, created_at, date, farm_name, plot, pred_label, pred_prob, notes FROM inferences", conn)

        if df.empty:
            st.info("Nenhum registo persistido.")
        else:
            st.dataframe(df, use_container_width=True)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Baixar Dados (CSV)", data=csv_bytes, file_name=f"inferences_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", type="primary")

    # ---------------- TELA 5: TELEMETRIA & LOGS ----------------
    elif menu == "🖥️ Telemetria & Logs":
        st.header("🖥️ Monitoramento Operacional & Logs em Tempo Real")
        t1, t2, t3, t4 = st.columns(4)
        status_ia = "Ativo (CUDA/GPU)" if torch.cuda.is_available() else "Ativo (CPU)"
        t1.metric("Motor de Inferência", status_ia)
        t2.metric("Latência do Pipeline", f"{st.session_state['telemetria']['ultima_latencia_ms']} ms")
        tam_db_kb = round(DB_PATH.stat().st_size / 1024, 1) if DB_PATH.exists() else 0
        t3.metric("Tamanho da Base SQLite", f"{tam_db_kb} KB")
        t4.metric("Ciclos Processados", st.session_state["telemetria"]["total_ciclos"])
        st.divider()
        st.subheader("📜 Consola de Eventos")
        logs_texto = "\n".join(st.session_state["telemetria"]["logs"][-30:]) if st.session_state["telemetria"]["logs"] else "Aguardando primeiros eventos..."
        st.text_area("Eventos Registados", logs_texto, height=320)

# ---------------- AUTO-REFRESH CONTROLADO ----------------
if (st.session_state["simulando"] or auto_refresh) and not trocou_de_aba:
    tempo_espera = max(velocidade if st.session_state["simulando"] else refresh_interval, 2)
    time.sleep(tempo_espera)
    st.rerun()