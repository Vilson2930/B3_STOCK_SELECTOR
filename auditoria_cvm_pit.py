"""B3 STOCK SELECTOR — Auditoria read-only de metadados PIT dos ZIPs CVM.

Executar na raiz do repositório: python auditoria_cvm_pit.py
Não baixa arquivos, não escreve no cache, não certifica publicações.
Saídas: outputs/cvm_pit_audit/inventario.csv e resumo.json.
"""
from __future__ import annotations
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'cvm' / 'raw'
OUTPUT = ROOT / 'outputs' / 'cvm_pit_audit'
DATASETS = ('DFP', 'ITR')
YEARS = range(2021, 2026)
TOKENS = ('REFER', 'VERSAO', 'RECEB', 'ENTREG', 'PUBLIC', 'PROTOC', 'DOCUMENT', 'LINK', 'URL', 'NUM_SEQ')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_header(stream):
    line = stream.readline()
    for enc in ('utf-8-sig', 'latin1'):
        try:
            decoded = line.decode(enc)
            return next(csv.reader([decoded], delimiter=';'))
        except (UnicodeError, csv.Error):
            continue
    raise ValueError('Cabeçalho não decodificável')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, zips = [], []
    for dataset in DATASETS:
        for year in YEARS:
            path = DATA / f'{dataset.lower()}_cia_aberta_{year}.zip'
            record = {'dataset': dataset, 'year': year, 'path': str(path.relative_to(ROOT)),
                      'exists': path.is_file(), 'sha256': None, 'files': 0, 'error': None}
            if not path.is_file():
                zips.append(record)
                continue
            try:
                if not zipfile.is_zipfile(path):
                    raise ValueError('ZIP inválido')
                record['sha256'] = digest(path)
                with zipfile.ZipFile(path) as archive:
                    bad = archive.testzip()
                    if bad:
                        raise ValueError(f'Arquivo interno corrompido: {bad}')
                    for member in sorted(archive.namelist()):
                        if not member.lower().endswith('.csv'):
                            continue
                        with archive.open(member) as stream:
                            cols = [c.strip().upper() for c in read_header(stream)]
                        candidates = [c for c in cols if any(t in c for t in TOKENS)]
                        rows.append({'dataset': dataset, 'year': year, 'source_file': member,
                                     'column_count': len(cols), 'candidate_columns': '|'.join(candidates),
                                     'has_reference': any('REFER' in c for c in cols),
                                     'has_version': any('VERSAO' in c for c in cols),
                                     'has_delivery_candidate': any(any(t in c for t in ('RECEB','ENTREG','PUBLIC','PROTOC')) for c in cols),
                                     'zip_sha256': record['sha256']})
                        record['files'] += 1
            except Exception as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
            zips.append(record)
    out_csv = OUTPUT / 'inventario.csv'
    with out_csv.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset','year','source_file','column_count','candidate_columns',
                                                'has_reference','has_version','has_delivery_candidate','zip_sha256'])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'tool': 'B3 STOCK SELECTOR - CVM PIT evidence inventory',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'DFP/ITR 2021-2025; ZIPs locais somente; nenhuma rede',
        'zips': zips, 'csv_files_inspected': len(rows),
        'files_with_delivery_candidate': sum(r['has_delivery_candidate'] for r in rows),
        'candidate_column_frequency': dict(Counter(c for r in rows for c in r['candidate_columns'].split('|') if c)),
        'source_authenticated': False, 'pit_approved': False,
        'limitation': 'Colunas candidatas não comprovam divulgação; sem protocolo e correspondência documental verificados, o fail-safe deve permanecer ativo.',
        'next_source': 'https://www.rad.cvm.gov.br/ENETWeb/frmConsultaExternaCVM.aspx',
    }
    out_json = OUTPUT / 'resumo.json'
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'ZIPs encontrados: {sum(x["exists"] for x in zips)}/{len(zips)}')
    print(f'CSVs inspecionados: {len(rows)}')
    print(f'Colunas candidatas de entrega: {summary["files_with_delivery_candidate"]} arquivos')
    print(f'Inventário: {out_csv}')
    print(f'Resumo: {out_json}')
    print('PIT: PENDENTE — nenhuma evidência promovida automaticamente')
    if not any(x['exists'] for x in zips):
        print('AVISO: nenhum ZIP local encontrado; execute no repositório que contém data/cvm/raw.')


if __name__ == '__main__':
    main()
