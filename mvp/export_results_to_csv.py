import argparse
import csv
import json
import sqlite3
from pathlib import Path
import pandas as pd
import gspread

def export_sqlite_to_csv(db_path: Path, output_csv: Path, flatten_probs: bool = False):
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.execute("SELECT * FROM inferences;")
    rows = cur.fetchall()

    if not rows:
        print("Atenção: Nenhum dado encontrado no banco de dados.")
        return False

    base_columns = rows[0].keys()
    prob_keys = set()
    if flatten_probs:
        for row in rows:
            if row["probs_json"]:
                try:
                    probs = json.loads(row["probs_json"])
                    prob_keys.update(probs.keys())
                except: pass
        prob_keys = sorted(prob_keys)

    header = list(base_columns)
    if flatten_probs: header += [f"prob_{k}" for k in prob_keys]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            rd = dict(row)
            if flatten_probs:
                probs = json.loads(rd.get("probs_json", "{}"))
                for k in prob_keys: rd[f"prob_{k}"] = probs.get(k)
            writer.writerow(rd)
    
    conn.close()
    print(f"📊 CSV gerado com sucesso: {output_csv.resolve()}")
    return True

def upload_to_google_sheets(csv_path, sheet_name):
    print(f"☁️ Sincronizando com Google Sheets via Conta de Serviço...")
    
    # 🚀 Usa o arquivo de credenciais fixo para o Docker
    gc = gspread.service_account(filename="credentials.json")
    
    df = pd.read_csv(csv_path)
    df = df.fillna('') 
    
    try:
        sh = gc.open(sheet_name)
    except gspread.exceptions.SpreadsheetNotFound:
        sh = gc.create(sheet_name)
        
    worksheet = sh.get_worksheet(0)
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())
    print(f"✨ Dashboard atualizado! Planilha: {sheet_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default="results.sqlite")
    parser.add_argument("--output_csv", type=str, default="results.csv")
    parser.add_argument("--flatten_probs", action="store_true")
    parser.add_argument("--sheet_name", type=str, default="Dashboard_Saude_Milho")
    args = parser.parse_args()

    # Executa a exportação
    if export_sqlite_to_csv(Path(args.db_path), Path(args.output_csv), args.flatten_probs):
        # Se exportou com sucesso, sobe para o Sheets
        upload_to_google_sheets(args.output_csv, args.sheet_name)

if __name__ == "__main__":
    main()
