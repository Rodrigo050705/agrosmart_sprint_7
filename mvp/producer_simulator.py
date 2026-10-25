import time
import shutil
import random
from pathlib import Path

# Configuração de caminhos
PASTA_DATASET = Path("dataset_teste")    # Onde ficam guardadas suas fotos de teste
PASTA_INPUT = Path("input_images") # Onde o consumer.py escuta

PASTA_INPUT.mkdir(parents=True, exist_ok=True)

print("⏳ Simulador AgroSmart iniciado. Gerando fluxo contínuo de imagens...")

while True:
    # Lista todas as imagens disponíveis no seu dataset de teste
    imagens = [p for p in PASTA_DATASET.glob("*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    if imagens:
        img_sorteada = random.choice(imagens)
        # Cria um nome único para simular uma nova foto entrando no fluxo
        novo_nome = f"cam_{int(time.time())}_{img_sorteada.name}"
        
        # Copia para a pasta de entrada do sistema
        shutil.copy(img_sorteada, PASTA_INPUT / novo_nome)
        print(f"📸 Nova imagem enviada para processamento: {novo_nome}")
    else:
        print("⚠ Nenhuma imagem encontrada na pasta 'dataset_teste'")
        
    # Simula o intervalo de envio de fotos (ex: a cada 10 segundos)
    time.sleep(10)
