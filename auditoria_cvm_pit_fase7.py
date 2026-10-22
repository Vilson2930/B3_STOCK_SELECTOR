"""B3 STOCK SELECTOR — Fase 7: auditoria numérica exploratória XML x CSV CVM.

Read-only: baixa somente os documentos divergentes da fase 3, extrai valores
numéricos com trilha XML e procura candidatos nos CSVs DFP/ITR do mesmo ano.
NÃO considera valores equivalentes sem unidade, escala, escopo, período e
semântica comprovados. NÃO cria registro PIT e NÃO altera o robô principal.

Execução: python auditoria_cvm_pit_fase7.py
Dependências: Python 3.11 e auditoria_cvm_pit_fase5.py (fetch/official_url).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from xml.etree import ElementTree as ET

from auditoria_cvm_pit_fase5 import fetch, official_url

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'cvm_pit_audit'
RAW = ROOT / 'data' / 'cvm' / 'raw'
MAX_DOCS = 15
MAX_XML_BYTES = 15 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED = 90 * 1024 * 1024
MAX_MEMBERS = 100
MAX_XML_VALUES_PER_DOC = 12000
MAX_CVM_CANDIDATES_PER_DOC = 25000
MAX_REPORT_ROWS = 100000
NUMERIC_TAGS = {'ULTIMOEXERCICIO', 'PENULTIMOEXERCICIO', 'ANTEPENULTIMOEXERCICIO',
                'VALOR', 'VALORCONTA', 'VALORATUAL', 'VALORANTERIOR'}
ACCOUNT_TAGS = {'CODIGOCONTA', 'DESCRICAOCONTA'}
CSV_VALUE_FIELDS = ('VL_CONTA', 'VL_CONTA1', 'VL_CONTA2', 'VL_CONTA3')
CSV_ACCOUNT_FIELDS = ('CD_CONTA', 'DS_CONTA')


def read_csv(name):
    path = OUT / name
    if not path.is_file():
        raise FileNotFoundError(f'Relatório necessário ausente: {path}')
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter=','))


def write_csv(name, fields, records):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def tagname(tag):
    return tag.rsplit('}', 1)[-1].upper() if isinstance(tag, str) else ''


def normalize_number(value):
    s = (value or '').strip().replace('\u00a0', '').replace(' ', '')
    if not s or not re.fullmatch(r'[+-]?(?:\d+(?:[.,]\d+)?|\d{1,3}(?:\.\d{3})+(?:,\d+)?)', s):
        return None
    if ',' in s:
        s = s.replace('.', '').replace(',', '.')
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def key(record):
    return tuple((record.get(f) or '').strip() for f in ('dataset', 'cnpj_cia', 'dt_refer', 'versao'))


def xml_accounts(raw, member):
    """Extrai somente folhas numéricas ligadas a um ancestral Conta/ContaFixa.

    Não atribui escopo, período ou unidade sem conhecer o schema documental.
    """
    if len(raw) > MAX_XML_BYTES:
        return [], 'XML_ACIMA_LIMITE'
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        return [], f'XML_INVALIDO:{str(exc)[:100]}'
    extracted = []
    def walk(node, stack, depth):
        if depth > 90 or len(extracted) >= MAX_XML_VALUES_PER_DOC:
            return
        name = tagname(node.tag)
        active = stack
        if name in ('CONTA', 'CONTAFIXA'):
            direct = {tagname(c.tag): (c.text or '').strip() for c in node if tagname(c.tag) in ACCOUNT_TAGS}
            active = stack + [(direct.get('CODIGOCONTA', ''), direct.get('DESCRICAOCONTA', ''))]
        if name in NUMERIC_TAGS and active:
            value = normalize_number(node.text)
            if value is not None:
                code, description = active[-1]
                extracted.append({'member': member, 'xml_tag': name, 'xml_account_code': code,
                                  'xml_account_description': description[:160],
                                  'xml_value': str(value), 'xml_value_sha256': hashlib.sha256((node.text or '').encode()).hexdigest()})
        for child in node:
            walk(child, active, depth + 1)
    walk(root, [], 0)
    return extracted, 'EXTRACAO_LIMITADA' if len(extracted) >= MAX_XML_VALUES_PER_DOC else 'EXTRAIDO'


def cvm_candidates(dataset, year, company_key):
    path = RAW / f'{dataset.lower()}_cia_aberta_{year}.zip'
    if not path.is_file():
        return [], 'ZIP_CVM_AUSENTE'
    results = []
    cnpj, refer, version = company_key
    with zipfile.ZipFile(path) as z:
        for member in sorted(z.namelist()):
            low = member.lower()
            if not low.endswith('.csv') or low == f'{dataset.lower()}_cia_aberta_{year}.csv':
                continue
            if not any(token in low for token in ('bpa_', 'bpp_', 'dre_', 'dfc_', 'dva_', 'dmpl_', 'drai_')):
                continue
            with z.open(member) as binary:
                text = io.TextIOWrapper(binary, encoding='latin1', errors='replace', newline='')
                reader = csv.DictReader(text, delimiter=';')
                fields = set(reader.fieldnames or [])
                if not {'CNPJ_CIA', 'DT_REFER', 'VERSAO', 'CD_CONTA'}.issubset(fields):
                    continue
                for record in reader:
                    if ((record.get('CNPJ_CIA') or '').strip(), (record.get('DT_REFER') or '').strip(),
                        (record.get('VERSAO') or '').strip()) != (cnpj, refer, version):
                        continue
                    for value_field in CSV_VALUE_FIELDS:
                        if value_field not in fields:
                            continue
                        value = normalize_number(record.get(value_field))
                        if value is not None:
                            results.append({'csv_file': member, 'csv_account_code': record.get('CD_CONTA', ''),
                                            'csv_account_description': (record.get('DS_CONTA') or '')[:160],
                                            'csv_value_field': value_field, 'csv_value': str(value),
                                            'csv_group': record.get('GRUPO_DFP', record.get('GRUPO_ITR', '')),
                                            'csv_currency_scale': record.get('ESCALA_MOEDA', ''),
                                            'csv_period_start': record.get('DT_INI_EXERC', ''),
                                            'csv_period_end': record.get('DT_FIM_EXERC', '')})
                            if len(results) >= MAX_CVM_CANDIDATES_PER_DOC:
                                return results, 'CSV_LIMITE_AMOSTRA'
    return results, 'CSV_CONSULTADO'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gaps = read_csv('fase3_divergencias.csv')
    links = {key(r): r for r in read_csv('fase3_links_identificadores.csv')}
    if not gaps or len(gaps) > MAX_DOCS:
        raise RuntimeError(f'Número de divergências fora do escopo: {len(gaps)}')
    docs, xml_rows, comparisons, issues = [], [], [], []
    for gap in gaps:
        k = key(gap)
        base = {field: gap.get(field, '') for field in ('dataset', 'ticker', 'cnpj_cia', 'dt_refer', 'versao', 'dt_receb')}
        result = {**base, 'status': 'PENDENTE', 'numero_sequencial': '', 'zip_sha256': '',
                  'xml_files': 0, 'xml_values': 0, 'csv_values': 0, 'same_code_and_value_candidates': 0,
                  'same_value_only_candidates': 0, 'xml_status': '', 'csv_status': '', 'error': '',
                  'numeric_reconciliation_approved': False, 'publication_authenticated': False,
                  'pit_approved': False}
        try:
            source = links.get(k)
            if not source:
                raise ValueError('LINK_NAO_ENCONTRADO')
            url, seq = official_url(source.get('document_link', ''))
            result['numero_sequencial'] = seq
            payload = None
            last_error = None
            for attempt in (1, 2):
                try:
                    data, _, status, _ = fetch(url)
                    if status != 200 or not zipfile.is_zipfile(io.BytesIO(data)):
                        raise ValueError(f'RESPOSTA_INVALIDA_HTTP_{status}')
                    payload = data
                    break
                except (OSError, TimeoutError, ValueError, zipfile.BadZipFile) as exc:
                    last_error = exc
                    if attempt == 1:
                        time.sleep(2)
            if payload is None:
                raise RuntimeError(f'DOWNLOAD_FALHOU:{last_error}')
            result['zip_sha256'] = hashlib.sha256(payload).hexdigest()
            extracted = []
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                infos = archive.infolist()
                if len(infos) > MAX_MEMBERS or sum(i.file_size for i in infos) > MAX_TOTAL_UNCOMPRESSED:
                    raise ValueError('LIMITES_ZIP_EXCEDIDOS')
                if archive.testzip() is not None:
                    raise ValueError('CRC_INVALIDO')
                for info in infos:
                    if not info.filename.lower().endswith('.xml') or info.file_size > MAX_XML_BYTES or info.flag_bits & 1:
                        continue
                    if not re.search(r'(?:dfp|itr).*\.xml$', info.filename, re.IGNORECASE):
                        continue
                    raw = archive.read(info)
                    rows, status = xml_accounts(raw, info.filename)
                    result['xml_files'] += 1
                    result['xml_status'] = status
                    extracted.extend(rows)
                    if status != 'EXTRAIDO':
                        issues.append({**base, 'issue': status, 'source_file': info.filename})
            result['xml_values'] = len(extracted)
            csv_rows, csv_status = cvm_candidates(k[0], int(k[2][:4]), (k[1], k[2], k[3]))
            result['csv_values'] = len(csv_rows)
            result['csv_status'] = csv_status
            by_code_value = defaultdict(list)
            by_value = defaultdict(list)
            for c in csv_rows:
                by_code_value[(c['csv_account_code'].strip(), c['csv_value'])].append(c)
                by_value[c['csv_value']].append(c)
            for x in extracted:
                if len(xml_rows) < MAX_REPORT_ROWS:
                    xml_rows.append({**base, 'numero_sequencial': seq, **x})
                exact = by_code_value.get((x['xml_account_code'].strip(), x['xml_value']), [])
                numeric = by_value.get(x['xml_value'], [])
                if exact:
                    result['same_code_and_value_candidates'] += 1
                elif numeric:
                    result['same_value_only_candidates'] += 1
                if (exact or numeric) and len(comparisons) < MAX_REPORT_ROWS:
                    for c in (exact or numeric)[:3]:
                        comparisons.append({**base, 'numero_sequencial': seq, **x, **c,
                                            'comparison_class': 'MESMO_CODIGO_E_NUMERO_SEM_SEMANTICA' if exact else 'MESMO_NUMERO_SEM_CHAVE',
                                            'unit_period_scope_verified': False, 'reconciled': False})
            result['status'] = 'AMOSTRA_NUMERICA_EXTRAIDA_SEM_RECONCILIACAO'
            if not extracted:
                issues.append({**base, 'issue': 'SEM_VALORES_XML_EXTRAIDOS', 'source_file': ''})
            if not csv_rows:
                issues.append({**base, 'issue': 'SEM_VALORES_CSV_NA_VERSAO', 'source_file': ''})
        except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, ET.ParseError) as exc:
            result['status'] = 'FALHA_DIAGNOSTICA'
            result['error'] = f'{type(exc).__name__}: {str(exc)[:250]}'
            issues.append({**base, 'issue': result['error'], 'source_file': ''})
        docs.append(result)
        print(f"{base['ticker']} {base['dataset']} {base['dt_refer']} v{base['versao']}: "
              f"XML={result['xml_values']} CSV={result['csv_values']} {result['status']}", flush=True)
    write_csv('fase7_documentos.csv', ['dataset','ticker','cnpj_cia','dt_refer','versao','dt_receb',
              'numero_sequencial','status','zip_sha256','xml_files','xml_values','csv_values',
              'same_code_and_value_candidates','same_value_only_candidates','xml_status','csv_status',
              'error','numeric_reconciliation_approved','publication_authenticated','pit_approved'], docs)
    write_csv('fase7_valores_xml.csv', ['dataset','ticker','cnpj_cia','dt_refer','versao','dt_receb',
              'numero_sequencial','member','xml_tag','xml_account_code','xml_account_description',
              'xml_value','xml_value_sha256'], xml_rows)
    write_csv('fase7_comparacoes_candidatas.csv', ['dataset','ticker','cnpj_cia','dt_refer','versao','dt_receb',
              'numero_sequencial','member','xml_tag','xml_account_code','xml_account_description','xml_value',
              'xml_value_sha256','csv_file','csv_account_code','csv_account_description','csv_value_field',
              'csv_value','csv_group','csv_currency_scale','csv_period_start','csv_period_end',
              'comparison_class','unit_period_scope_verified','reconciled'], comparisons)
    write_csv('fase7_alertas.csv', ['dataset','ticker','cnpj_cia','dt_refer','versao','dt_receb',
              'issue','source_file'], issues)
    summary = {'tool': 'B3 STOCK SELECTOR - fase 7 - auditoria numerica exploratoria',
               'generated_at_utc': datetime.now(timezone.utc).isoformat(),
               'documents_requested': len(gaps), 'documents_processed': sum(d['status'].startswith('AMOSTRA') for d in docs),
               'xml_values_extracted': sum(d['xml_values'] for d in docs),
               'csv_values_same_document_version': sum(d['csv_values'] for d in docs),
               'same_code_and_number_candidates': sum(d['same_code_and_value_candidates'] for d in docs),
               'same_number_only_candidates': sum(d['same_value_only_candidates'] for d in docs),
               'issues': dict(Counter(x['issue'] for x in issues)),
               'numeric_reconciliation_approved': False, 'publication_authenticated': False,
               'pit_approved': False,
               'limitations': [
                   'Comparação de números é apenas diagnóstico: não há mapeamento homologado de unidade, escala, escopo ou períodos XML x CSV.',
                   'Ausência de linhas CSV para versão histórica é um resultado válido; não inventar equivalência com versões posteriores.',
                   'Arquivos XML e CSV podem usar hierarquias de contas distintas; igualdade numérica isolada não comprova reconciliação.',
                   'Não autentica data de divulgação nem identidade documental independente.',
                   'Não altera o robô nem gera publication_registry.csv; FAIL-SAFE preservado.'
               ]}
    (OUT / 'fase7_resumo.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
