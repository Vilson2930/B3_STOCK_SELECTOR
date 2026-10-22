"""B3 STOCK SELECTOR | Fase 8: diagnostico XML e contraste entre versoes CVM.

Auditoria READ-ONLY. Reutiliza fase3_divergencias.csv e URLs da fase3.
Nao autentica publicacao, nao reconcilia unidade/escopo, nao aprova PIT.
Requer auditoria_cvm_pit_fase5.py e auditoria_cvm_pit_fase7.py.
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
from pathlib import Path
from xml.etree import ElementTree as ET
from auditoria_cvm_pit_fase5 import fetch, official_url
from auditoria_cvm_pit_fase7 import ROOT, OUT, RAW, read_csv, write_csv, key, tagname, xml_accounts, normalize_number

MAX_XML = 20 * 1024 * 1024
MAX_ARCHIVE_UNCOMPRESSED = 120 * 1024 * 1024
MAX_FILES = 120
MAX_ROWS = 100000
CSV_FIELDS = ('VL_CONTA', 'VL_CONTA1', 'VL_CONTA2', 'VL_CONTA3')


def csv_other_versions(dataset, year, cnpj, refer, requested_version):
    """Consulta versoes DISTINTAS sem tratá-las como equivalentes."""
    path = RAW / f'{dataset.lower()}_cia_aberta_{year}.zip'
    if not path.is_file():
        return [], 'ZIP_CVM_AUSENTE'
    rows = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            low = name.lower()
            if not low.endswith('.csv') or not any(f'_{t}_' in low for t in ('bpa','bpp','dre','dfc','dva','dmpl','drai')):
                continue
            with archive.open(name) as binary:
                reader = csv.DictReader(io.TextIOWrapper(binary, encoding='latin1', errors='replace', newline=''), delimiter=';')
                fields = set(reader.fieldnames or [])
                if not {'CNPJ_CIA','DT_REFER','VERSAO','CD_CONTA'}.issubset(fields):
                    continue
                for r in reader:
                    if (r.get('CNPJ_CIA','').strip() != cnpj or r.get('DT_REFER','').strip() != refer or
                        r.get('VERSAO','').strip() == requested_version):
                        continue
                    for field in CSV_FIELDS:
                        if field in fields:
                            value = normalize_number(r.get(field))
                            if value is not None:
                                rows.append({'csv_file':name, 'csv_version':r.get('VERSAO',''),
                                  'csv_account_code':r.get('CD_CONTA','').strip(), 'csv_account_description':r.get('DS_CONTA','')[:160],
                                  'csv_value_field':field,'csv_value':str(value), 'csv_group':r.get('GRUPO_DFP',r.get('GRUPO_ITR','')),
                                  'csv_scale':r.get('ESCALA_MOEDA',''), 'csv_start':r.get('DT_INI_EXERC',''),
                                  'csv_end':r.get('DT_FIM_EXERC','')})
                            if len(rows) >= MAX_ROWS:
                                return rows, 'CSV_LIMITE'
    return rows, 'CSV_CONSULTADO'


def xml_diagnostics(raw, member):
    if len(raw) > MAX_XML:
        return [], [], 'XML_MUITO_GRANDE'
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        return [], [], f'XML_INVALIDO:{str(exc)[:100]}'
    tagcounts = Counter()
    examples = defaultdict(list)
    for node in root.iter():
        t = tagname(node.tag)
        tagcounts[t] += 1
        s = (node.text or '').strip()
        if s and len(examples[t]) < 2:
            examples[t].append(s[:100])
    tags = [{'member':member, 'tag':t,'count':n,'examples':' | '.join(examples[t])} for t,n in tagcounts.most_common(300)]
    rows, status = xml_accounts(raw, member)
    return tags, rows, status


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gaps = read_csv('fase3_divergencias.csv')
    links = {key(r):r for r in read_csv('fase3_links_identificadores.csv')}
    if len(gaps) != 9:
        raise RuntimeError(f'Esperados 9 documentos divergentes; encontrados {len(gaps)}. Nao prosseguir silenciosamente.')
    docs, tags, comparisons, issues = [], [], [], []
    for gap in gaps:
        k = key(gap)
        base = {f:gap.get(f,'') for f in ('dataset','ticker','cnpj_cia','dt_refer','versao','dt_receb')}
        doc = {**base,'numero_sequencial':'','zip_sha256':'','xml_files':0,'xml_values':0,
               'other_version_csv_values':0,'same_code_number_cross_version':0,
               'same_number_cross_version':0,'xml_statuses':'','csv_status':'','status':'PENDENTE',
               'error':'','numeric_reconciliation_approved':False,'publication_authenticated':False,'pit_approved':False}
        try:
            source = links.get(k)
            if source is None:
                raise ValueError('LINK_AUSENTE')
            url, seq = official_url(source.get('document_link',''))
            doc['numero_sequencial'] = seq
            data = None
            for attempt in range(2):
                try:
                    payload, _, status, _ = fetch(url)
                    if status != 200 or not zipfile.is_zipfile(io.BytesIO(payload)):
                        raise ValueError(f'HTTP_OU_ZIP_INVALIDO:{status}')
                    data = payload
                    break
                except (OSError,TimeoutError,ValueError) as exc:
                    if attempt:
                        raise
                    time.sleep(2)
            if data is None:
                raise RuntimeError('DOWNLOAD_SEM_DADOS')
            doc['zip_sha256'] = hashlib.sha256(data).hexdigest()
            xml_values, states = [], []
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                infos = archive.infolist()
                if len(infos) > MAX_FILES or sum(x.file_size for x in infos) > MAX_ARCHIVE_UNCOMPRESSED:
                    raise ValueError('ZIP_EXCEDE_LIMITES')
                if archive.testzip() is not None:
                    raise ValueError('ZIP_CRC_INVALIDO')
                for info in infos:
                    if not info.filename.lower().endswith('.xml') or info.flag_bits & 1:
                        continue
                    doc['xml_files'] += 1
                    raw = archive.read(info)
                    tag_rows, vals, state = xml_diagnostics(raw, info.filename)
                    states.append(f'{info.filename}:{state}')
                    for tr in tag_rows:
                        if len(tags) < MAX_ROWS:
                            tags.append({**base,'numero_sequencial':seq,**tr})
                    xml_values.extend(vals)
            doc['xml_values'] = len(xml_values)
            doc['xml_statuses'] = '; '.join(states)[:500]
            other, csv_state = csv_other_versions(k[0], int(k[2][:4]), k[1],k[2],k[3])
            doc['csv_status'] = csv_state
            doc['other_version_csv_values'] = len(other)
            by_code_number = defaultdict(list)
            by_number = defaultdict(list)
            for c in other:
                by_code_number[(c['csv_account_code'],c['csv_value'])].append(c)
                by_number[c['csv_value']].append(c)
            for x in xml_values:
                exact = by_code_number.get((x['xml_account_code'].strip(),x['xml_value']),[])
                numeric = by_number.get(x['xml_value'],[])
                if exact:
                    doc['same_code_number_cross_version'] += 1
                elif numeric:
                    doc['same_number_cross_version'] += 1
                if exact or numeric:
                    for c in (exact or numeric)[:2]:
                        if len(comparisons) < MAX_ROWS:
                            comparisons.append({**base,'numero_sequencial':seq,**x,**c,
                                'comparison_class':'CODIGO_NUMERO_VERSAO_DIFERENTE' if exact else 'NUMERO_VERSAO_DIFERENTE',
                                'same_version':False,'semantics_verified':False,'reconciled':False})
            if not xml_values:
                issues.append({**base,'issue':'SEM_VALORES_XML','detail':doc['xml_statuses']})
            if not other:
                issues.append({**base,'issue':'SEM_CSV_EM_OUTRAS_VERSOES','detail':csv_state})
            doc['status'] = 'DIAGNOSTICO_CONCLUIDO_SEM_HOMOLOGACAO'
        except (OSError,ValueError,RuntimeError,zipfile.BadZipFile,ET.ParseError) as exc:
            doc['status'] = 'FALHA_DIAGNOSTICA'
            doc['error'] = f'{type(exc).__name__}: {str(exc)[:250]}'
            issues.append({**base,'issue':'FALHA_DIAGNOSTICA','detail':doc['error']})
        docs.append(doc)
        print(f"{base['ticker']} {base['dataset']} {base['dt_refer']} v{base['versao']}: XML={doc['xml_values']} CSV outras versoes={doc['other_version_csv_values']} {doc['status']}",flush=True)
    write_csv('fase8_documentos.csv', list(docs[0]), docs)
    write_csv('fase8_tags_xml.csv', ['dataset','ticker','cnpj_cia','dt_refer','versao','dt_receb','numero_sequencial','member','tag','count','examples'],tags)
    write_csv('fase8_comparacoes_entre_versoes.csv', ['dataset','ticker','cnpj_cia','dt_refer','versao','dt_receb','numero_sequencial',
        'member','xml_tag','xml_account_code','xml_account_description','xml_value','xml_value_sha256','csv_file','csv_version',
        'csv_account_code','csv_account_description','csv_value_field','csv_value','csv_group','csv_scale','csv_start','csv_end',
        'comparison_class','same_version','semantics_verified','reconciled'],comparisons)
    write_csv('fase8_alertas.csv',['dataset','ticker','cnpj_cia','dt_refer','versao','dt_receb','issue','detail'],issues)
    summary = {'phase':8,'generated_at_utc':datetime.now(timezone.utc).isoformat(),'documents':len(docs),
        'processed':sum(d['status'].startswith('DIAGNOSTICO') for d in docs),'xml_values':sum(d['xml_values'] for d in docs),
        'csv_values_from_other_versions':sum(d['other_version_csv_values'] for d in docs),
        'same_code_number_cross_version':sum(d['same_code_number_cross_version'] for d in docs),
        'same_number_cross_version':sum(d['same_number_cross_version'] for d in docs),
        'xml_tag_inventory_rows':len(tags),'comparison_sample_rows':len(comparisons),
        'issues':dict(Counter(i['issue'] for i in issues)),
        'numeric_reconciliation_approved':False,'publication_authenticated':False,'pit_approved':False,
        'limitations':['Comparacoes sao exploratorias e apenas ENTRE VERSOES DIFERENTES.',
         'Valores iguais nao demonstram mesma unidade, escopo, conta ou periodo.',
         'O inventario de tags auxilia a diagnosticar XMLs que nao geraram valores.',
         'Sem prova independente de publicacao historica; FAIL-SAFE mantido.',
         'Nenhum arquivo do robo principal foi alterado.']}
    (OUT/'fase8_resumo.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
