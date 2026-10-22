"""B3 STOCK SELECTOR — Fase 6, auditoria read-only de conteúdo ENET.

Executar após as fases 1–3. Baixa APENAS documentos divergentes (máximo 15),
inspeciona membros internos e indícios textuais de identidade, sem extrair em disco.
Não autentica publicação, não valida números contábeis e não cria PIT registry.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from auditoria_cvm_pit_fase5 import official_url, fetch, MAX_UNCOMPRESSED

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'cvm_pit_audit'
MAX_DOCS = 15
MAX_ENTRIES = 100
MAX_MEMBER = 20 * 1024 * 1024
MAX_TOTAL = 80 * 1024 * 1024
MAX_TEXT_SCAN = 2 * 1024 * 1024
MAX_XML_ELEMENTS = 10000


def read_rows(filename):
    with (OUT / filename).open(newline='', encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


def write_rows(filename, rows, fields):
    with (OUT / filename).open('w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return tuple((row.get(field) or '').strip() for field in ('dataset', 'cnpj_cia', 'dt_refer', 'versao'))


def text_evidence(data, cnpj, refer, version):
    """Busca somente ocorrências textuais, sem assumir semântica dos campos."""
    head = data[:MAX_TEXT_SCAN]
    samples = []
    for enc in ('utf-8-sig', 'utf-16', 'latin1'):
        try:
            samples.append(head.decode(enc))
        except UnicodeError:
            continue
    compact_cnpj = re.sub(r'\D', '', cnpj)
    reference_formats = {refer, refer.replace('-', '/'), refer[8:10] + '/' + refer[5:7] + '/' + refer[:4]} if len(refer) == 10 else {refer}
    cnpj_found = any(compact_cnpj in re.sub(r'\D', '', s) for s in samples if compact_cnpj)
    reference_found = any(any(d in s for d in reference_formats if d) for s in samples)
    version_found = any(bool(re.search(r'(?i)vers[aã]o\W{0,20}' + re.escape(version) + r'\b', s)) for s in samples if version)
    return cnpj_found, reference_found, version_found


def inspect_zip(data, gap):
    results = []
    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise ValueError('RESPOSTA_NAO_E_ZIP')
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_ENTRIES:
            raise ValueError('NUMERO_MEMBROS_ACIMA_LIMITE')
        if sum(i.file_size for i in infos) > min(MAX_TOTAL, MAX_UNCOMPRESSED):
            raise ValueError('ZIP_DESCOMPRIMIDO_ACIMA_LIMITE')
        if zf.testzip() is not None:
            raise ValueError('ZIP_CRC_INVALIDO')
        for info in infos:
            if info.is_dir():
                continue
            if info.file_size > MAX_MEMBER or info.flag_bits & 1:
                results.append({'member': info.filename, 'bytes': info.file_size, 'status': 'IGNORADO_LIMITE_OU_CRIPTOGRAFIA'})
                continue
            with zf.open(info) as fh:
                raw = fh.read(MAX_MEMBER + 1)
            if len(raw) > MAX_MEMBER:
                raise ValueError('MEMBRO_EXCEDE_LIMITE_REAL')
            kind = ('PDF' if raw.startswith(b'%PDF-') else 'XML' if raw.lstrip()[:100].startswith((b'<?xml', b'<')) else 'OUTRO')
            xml_root, xml_tags = '', ''
            if kind == 'XML':
                try:
                    tags = Counter()
                    count = 0
                    for _, element in ET.iterparse(io.BytesIO(raw), events=('start',)):
                        count += 1
                        if count > MAX_XML_ELEMENTS:
                            break
                        tags[element.tag.rsplit('}', 1)[-1]] += 1
                    xml_root = next(iter(tags), '')
                    xml_tags = '|'.join(t for t, _ in tags.most_common(12))
                except ET.ParseError:
                    kind = 'TEXTO_OU_XML_INVALIDO'
            cnpj_hit, ref_hit, version_hit = text_evidence(raw, gap['cnpj_cia'], gap['dt_refer'], gap['versao'])
            results.append({'member': info.filename, 'bytes': len(raw), 'sha256_member': hashlib.sha256(raw).hexdigest(),
                            'kind': kind, 'xml_root': xml_root, 'xml_tags_sample': xml_tags,
                            'cnpj_text_hit': cnpj_hit, 'reference_text_hit': ref_hit, 'version_text_hit': version_hit,
                            'status': 'INDICIOS_NAO_AUTENTICADOS'})
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gaps = read_rows('fase3_divergencias.csv')
    links = {key(row): row for row in read_rows('fase3_links_identificadores.csv')}
    if not gaps or len(gaps) > MAX_DOCS:
        raise RuntimeError(f'Quantidade de divergências fora do escopo seguro: {len(gaps)}')
    docs, members = [], []
    for gap in gaps:
        base = {field: gap.get(field, '') for field in ('dataset', 'ticker', 'cnpj_cia', 'dt_refer', 'versao', 'dt_receb')}
        row = {**base, 'status': 'PENDENTE', 'sha256_zip': '', 'members': 0, 'textual_cnpj_hits': 0,
               'textual_reference_hits': 0, 'textual_version_hits': 0, 'error': '',
               'identity_authenticated': False, 'publication_authenticated': False, 'pit_approved': False}
        try:
            source = links.get(key(gap))
            if not source:
                raise ValueError('SEM_LINK_NA_FASE_3')
            url, seq = official_url(source.get('document_link', ''))
            row['numero_sequencial'] = seq
            last_error = None
            for attempt in (1, 2):
                try:
                    data, final, http_status, content_type = fetch(url)
                    if http_status != 200:
                        raise ValueError(f'HTTP_{http_status}')
                    row['sha256_zip'] = hashlib.sha256(data).hexdigest()
                    inspected = inspect_zip(data, gap)
                    for item in inspected:
                        members.append({**base, 'numero_sequencial': seq, **item})
                    row['members'] = len(inspected)
                    row['textual_cnpj_hits'] = sum(x.get('cnpj_text_hit') is True for x in inspected)
                    row['textual_reference_hits'] = sum(x.get('reference_text_hit') is True for x in inspected)
                    row['textual_version_hits'] = sum(x.get('version_text_hit') is True for x in inspected)
                    row['status'] = 'CONTEUDO_INSPECIONADO_SEM_AUTENTICACAO'
                    last_error = None
                    break
                except (OSError, ValueError, TimeoutError, zipfile.BadZipFile, ET.ParseError) as exc:
                    last_error = f'{type(exc).__name__}: {str(exc)[:180]}'
                    if attempt == 1:
                        time.sleep(2)
            if last_error:
                row['status'], row['error'] = 'FALHA_DOCUMENTAL', last_error
        except (ValueError, OSError) as exc:
            row['status'], row['error'] = 'FALHA_PREPARACAO', f'{type(exc).__name__}: {str(exc)[:180]}'
        docs.append(row)
        print(f"{row['ticker']} {row['dataset']} {row['dt_refer']} v{row['versao']}: {row['status']}", flush=True)
    doc_fields = ['dataset', 'ticker', 'cnpj_cia', 'dt_refer', 'versao', 'dt_receb', 'numero_sequencial',
                  'status', 'sha256_zip', 'members', 'textual_cnpj_hits', 'textual_reference_hits',
                  'textual_version_hits', 'error', 'identity_authenticated', 'publication_authenticated', 'pit_approved']
    member_fields = ['dataset', 'ticker', 'cnpj_cia', 'dt_refer', 'versao', 'dt_receb', 'numero_sequencial',
                     'member', 'bytes', 'sha256_member', 'kind', 'xml_root', 'xml_tags_sample',
                     'cnpj_text_hit', 'reference_text_hit', 'version_text_hit', 'status']
    write_rows('fase6_documentos.csv', docs, doc_fields)
    write_rows('fase6_membros.csv', members, member_fields)
    summary = {'tool': 'B3 STOCK SELECTOR - fase 6 - inspeção documental exploratória',
               'generated_at_utc': datetime.now(timezone.utc).isoformat(),
               'documents_requested': len(gaps), 'documents_inspected': sum(x['status'] == 'CONTEUDO_INSPECIONADO_SEM_AUTENTICACAO' for x in docs),
               'members_inspected': len(members), 'status_counts': dict(Counter(x['status'] for x in docs)),
               'content_identity_authenticated': False, 'historical_publication_authenticated': False,
               'accounting_values_reconciled': False, 'pit_approved': False,
               'limitations': ['Busca de CNPJ, data e versão é textual e pode produzir falsos positivos/negativos.',
                               'XML/PDF é inspecionado estruturalmente; não há reconciliação de valores contábeis.',
                               'SHA256 e resposta HTTP atual não comprovam divulgação histórica.',
                               'Nenhum documento ou dado é promovido ao registro PIT; FAIL-SAFE mantido.']}
    (OUT / 'fase6_resumo.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
