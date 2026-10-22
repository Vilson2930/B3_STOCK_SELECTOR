"""B3 STOCK SELECTOR — auditoria documental exploratória CVM, somente leitura.

Executar: python auditoria_cvm_pit.py
Entrada: data/cvm/raw/{dfp,itr}_cia_aberta_{2021..2025}.zip
Saídas: outputs/cvm_pit_audit/{inventario.csv,resumo.json,documentos.csv,
        correspondencias.csv,anomalias.csv}.

NÃO autentica publicação, NÃO produz publication_registry.csv e NÃO libera PIT.
Os vínculos com tickers são apenas candidatos por denominação, nunca identidade
jurídica definitiva. Datas de recebimento não são automaticamente datas de
publicação ou disponibilidade pública.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'cvm' / 'raw'
OUTPUT = ROOT / 'outputs' / 'cvm_pit_audit'
DATASETS = ('DFP', 'ITR')
YEARS = range(2021, 2026)
TOKENS = ('REFER', 'VERSAO', 'RECEB', 'ENTREG', 'PUBLIC', 'PROTOC', 'DOCUMENT', 'LINK', 'URL', 'NUM_SEQ')
MAIN_FIELDS = ('CNPJ_CIA', 'DENOM_CIA', 'DT_REFER', 'VERSAO', 'DT_RECEB', 'LINK_DOC')
DOCUMENT_FIELDS = ('dataset', 'year', 'source_file', 'ticker_candidate', 'denom_cia',
                   'cnpj_cia', 'dt_refer', 'versao', 'dt_receb', 'link_doc',
                   'link_official_domain', 'document_key', 'row_number', 'status')
MATCH_FIELDS = ('dataset', 'year', 'ticker_candidate', 'cnpj_cia', 'dt_refer',
                'versao', 'source_file', 'main_rows', 'accounting_rows',
                'accounting_files', 'status')
ANOMALY_FIELDS = ('dataset', 'year', 'source_file', 'row_number', 'issue', 'details')

# Correspondências textuais exploratórias: não substituem cadastro oficial B3/CVM.
COMPANIES = {
    'TOTS3': ('TOTVS',),
    'B3SA3': ('B3 S A BRASIL BOLSA BALCAO', 'B3 BRASIL BOLSA BALCAO'),
    'FLRY3': ('FLEURY',),
    'TGMA3': ('TEGMA',),
    'YDUQ3': ('YDUQS', 'YDUQ'),
}


def normalize(value: str) -> str:
    value = unicodedata.normalize('NFKD', str(value or ''))
    value = ''.join(c for c in value if not unicodedata.combining(c))
    return re.sub(r'[^A-Z0-9]+', ' ', value.upper()).strip()


def ticker_candidate(name: str) -> str:
    norm = normalize(name)
    matches = [ticker for ticker, aliases in COMPANIES.items()
               if any(norm == alias or norm.startswith(alias + ' ') for alias in aliases)]
    return matches[0] if len(matches) == 1 else ''


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def open_csv(archive: zipfile.ZipFile, member: str):
    raw = archive.open(member)
    # Os CSVs oficiais frequentemente usam Latin-1; UTF-8 tem prioridade.
    sample = raw.read(8192)
    raw.close()
    encoding = 'utf-8-sig'
    try:
        sample.decode(encoding)
    except UnicodeError:
        encoding = 'latin1'
    binary = archive.open(member)
    text = io.TextIOWrapper(binary, encoding=encoding, newline='', errors='replace')
    return text


def valid_date(value: str) -> bool:
    if not value:
        return False
    try:
        datetime.strptime(value[:10], '%Y-%m-%d')
        return bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}', value[:10]))
    except ValueError:
        return False


def is_official_link(url: str) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(url.strip())
        host = (parsed.hostname or '').lower()
        return parsed.scheme.lower() == 'https' and (host == 'cvm.gov.br' or host.endswith('.cvm.gov.br'))
    except ValueError:
        return False


def save_csv(name: str, fields: tuple[str, ...], records: list[dict]) -> None:
    with (OUTPUT / name).open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def issue(items: list[dict], dataset: str, year: int, source: str,
          row: int, code: str, details: str = '') -> None:
    items.append(dict(dataset=dataset, year=year, source_file=source,
                      row_number=row, issue=code, details=details[:400]))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inventory: list[dict] = []
    documents: list[dict] = []
    anomalies: list[dict] = []
    matches: list[dict] = []
    zips: list[dict] = []
    stats = Counter()

    for dataset in DATASETS:
        for year in YEARS:
            path = DATA / f'{dataset.lower()}_cia_aberta_{year}.zip'
            z = dict(dataset=dataset, year=year, path=str(path.relative_to(ROOT)),
                     exists=path.is_file(), sha256=None, files=0, error=None)
            if not path.is_file():
                issue(anomalies, dataset, year, z['path'], 0, 'ZIP_AUSENTE')
                zips.append(z)
                continue
            try:
                if not zipfile.is_zipfile(path):
                    raise ValueError('ZIP inválido')
                z['sha256'] = digest(path)
                with zipfile.ZipFile(path) as archive:
                    corrupt = archive.testzip()
                    if corrupt:
                        raise ValueError(f'Membro corrompido: {corrupt}')
                    members = sorted(n for n in archive.namelist() if n.lower().endswith('.csv'))
                    main_name = f'{dataset.lower()}_cia_aberta_{year}.csv'
                    principal = [m for m in members if Path(m).name.lower() == main_name]
                    if len(principal) != 1:
                        issue(anomalies, dataset, year, z['path'], 0,
                              'ARQUIVO_PRINCIPAL_AUSENTE_OU_AMBIGUO', str(principal))
                    main_keys = Counter()
                    main_meta: dict[tuple[str, str, str], dict] = {}
                    accounting_keys = Counter()
                    accounting_sources = defaultdict(set)
                    for member in members:
                        with open_csv(archive, member) as stream:
                            reader = csv.DictReader(stream, delimiter=';')
                            cols = [c.strip().upper() for c in (reader.fieldnames or [])]
                            candidates = [c for c in cols if any(t in c for t in TOKENS)]
                            inventory.append(dict(dataset=dataset, year=year,
                                                  source_file=member, column_count=len(cols),
                                                  candidate_columns='|'.join(candidates),
                                                  has_reference=any('REFER' in c for c in cols),
                                                  has_version='VERSAO' in cols,
                                                  has_delivery_candidate=any(any(t in c for t in
                                                      ('RECEB', 'ENTREG', 'PUBLIC', 'PROTOC')) for c in cols),
                                                  zip_sha256=z['sha256']))
                            z['files'] += 1
                            is_main = member in principal
                            required = MAIN_FIELDS if is_main else ('CNPJ_CIA', 'DT_REFER', 'VERSAO')
                            if any(c not in cols for c in required):
                                if is_main:
                                    issue(anomalies, dataset, year, member, 0,
                                          'COLUNAS_PRINCIPAIS_AUSENTES',
                                          ','.join(c for c in required if c not in cols))
                                continue
                            # Arquivos não principais: apenas índices de identidade e versão;
                            # não guardar linhas contábeis nem interpretar valores financeiros.
                            for line_number, original in enumerate(reader, start=2):
                                r = {str(k).strip().upper(): (v or '').strip()
                                     for k, v in original.items() if k is not None}
                                key = (r.get('CNPJ_CIA', ''), r.get('DT_REFER', ''),
                                       r.get('VERSAO', ''))
                                if is_main:
                                    ticker = ticker_candidate(r.get('DENOM_CIA', ''))
                                    if not ticker:
                                        continue
                                    stats['main_target_rows'] += 1
                                    main_keys[key] += 1
                                    link = r.get('LINK_DOC', '')
                                    receipt = r.get('DT_RECEB', '')
                                    flags = []
                                    if not all(key):
                                        flags.append('CHAVE_INCOMPLETA')
                                    if not valid_date(receipt):
                                        flags.append('DT_RECEB_AUSENTE_OU_INVALIDA')
                                    if valid_date(receipt) and valid_date(key[1]) and receipt[:10] < key[1][:10]:
                                        flags.append('RECEBIMENTO_ANTES_DA_REFERENCIA')
                                    if not link:
                                        flags.append('LINK_DOC_AUSENTE')
                                    elif not is_official_link(link):
                                        flags.append('LINK_DOMINIO_NAO_OFICIAL_OU_NAO_HTTPS')
                                    for flag in flags:
                                        issue(anomalies, dataset, year, member, line_number,
                                              flag, f'{ticker}: {key}')
                                    record = dict(dataset=dataset, year=year,
                                                  source_file=member, ticker_candidate=ticker,
                                                  denom_cia=r.get('DENOM_CIA', ''),
                                                  cnpj_cia=key[0], dt_refer=key[1], versao=key[2],
                                                  dt_receb=receipt, link_doc=link,
                                                  link_official_domain=is_official_link(link),
                                                  document_key='|'.join(key), row_number=line_number,
                                                  status='PENDENTE_PIT_' + ('COM_ANOMALIA' if flags else 'METADADOS_PRESENTES'))
                                    documents.append(record)
                                    main_meta.setdefault(key, record)
                                else:
                                    if all(key):
                                        accounting_keys[key] += 1
                                        accounting_sources[key].add(member)
                    for key, count in main_keys.items():
                        doc = main_meta[key]
                        linked = accounting_keys[key]
                        status = 'CHAVE_ENCONTRADA' if linked else 'SEM_LINHAS_CONTABEIS_CORRESPONDENTES'
                        if count > 1:
                            status = 'CHAVE_DUPLICADA_NO_PRINCIPAL'
                            issue(anomalies, dataset, year, main_name, 0,
                                  'CHAVE_DUPLICADA_NO_PRINCIPAL', f'{key}: {count} linhas')
                        if not linked:
                            issue(anomalies, dataset, year, main_name, 0,
                                  'SEM_LINHAS_CONTABEIS_CORRESPONDENTES', str(key))
                        matches.append(dict(dataset=dataset, year=year,
                                            ticker_candidate=doc['ticker_candidate'],
                                            cnpj_cia=key[0], dt_refer=key[1], versao=key[2],
                                            source_file=main_name, main_rows=count,
                                            accounting_rows=linked,
                                            accounting_files=len(accounting_sources[key]),
                                            status=status))
            except Exception as exc:
                z['error'] = f'{type(exc).__name__}: {exc}'
                issue(anomalies, dataset, year, z['path'], 0, 'FALHA_LEITURA_ZIP', z['error'])
            zips.append(z)

    save_csv('inventario.csv', ('dataset', 'year', 'source_file', 'column_count',
             'candidate_columns', 'has_reference', 'has_version',
             'has_delivery_candidate', 'zip_sha256'), inventory)
    save_csv('documentos.csv', DOCUMENT_FIELDS, documents)
    save_csv('correspondencias.csv', MATCH_FIELDS, matches)
    save_csv('anomalias.csv', ANOMALY_FIELDS, anomalies)
    summary = {
        'tool': 'B3 STOCK SELECTOR - auditoria documental exploratoria CVM',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'DFP/ITR 2021-2025; cinco empresas candidatas; ZIPs locais; sem rede',
        'target_tickers': list(COMPANIES),
        'zips': zips,
        'csv_files_inspected': len(inventory),
        'files_with_delivery_candidate': sum(bool(r['has_delivery_candidate']) for r in inventory),
        'candidate_column_frequency': dict(Counter(c for r in inventory
            for c in r['candidate_columns'].split('|') if c)),
        'target_document_rows': len(documents),
        'document_rows_by_ticker': dict(Counter(d['ticker_candidate'] for d in documents)),
        'document_rows_with_official_domain_link': sum(d['link_official_domain'] for d in documents),
        'document_rows_with_valid_receipt_date': sum(valid_date(d['dt_receb']) for d in documents),
        'identity_version_matches': len(matches),
        'identity_version_matches_with_accounting_rows': sum(m['accounting_rows'] > 0 for m in matches),
        'anomalies': len(anomalies),
        'anomalies_by_type': dict(Counter(a['issue'] for a in anomalies)),
        'source_authenticated': False,
        'pit_approved': False,
        'limitations': [
            'DT_RECEB é metadado de recebimento; não foi demonstrada equivalência com divulgação pública.',
            'LINK_DOC e domínio oficial não comprovam acessibilidade histórica, protocolo ou versão do conteúdo.',
            'Correspondência textual entre denominação e ticker não autentica identidade societária.',
            'Chave CNPJ_CIA/DT_REFER/VERSAO comum a CSVs não prova que o conteúdo era conhecido naquela data.',
            'Nenhum arquivo publication_registry.csv é criado; fail-safe de produção permanece ativo.',
        ],
        'next_source': 'https://www.rad.cvm.gov.br/ENETWeb/frmConsultaExternaCVM.aspx',
    }
    (OUTPUT / 'resumo.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'ZIPs encontrados: {sum(z["exists"] for z in zips)}/{len(zips)}')
    print(f'CSVs inspecionados: {len(inventory)}')
    print(f'Documentos candidatos das cinco empresas: {len(documents)}')
    print(f'Correspondências de chave com linhas contábeis: '
          f'{summary["identity_version_matches_with_accounting_rows"]}/{len(matches)}')
    print(f'Anomalias: {len(anomalies)}')
    print(f'Relatórios: {OUTPUT}')
    print('PIT: PENDENTE — nenhum registro autenticado ou aprovado automaticamente')


if __name__ == '__main__':
    main()
