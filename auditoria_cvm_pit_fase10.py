"""B3 STOCK SELECTOR — Fase 10: integridade documental e diagnostico XML.

Somente leitura das fontes CVM e dos relatorios anteriores. Preserva os ZIPs
obtidos e seus manifestos, sem homologar identidade, contabilidade ou PIT.
Executar apos fases 3, 7, 8 e 9.
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
from auditoria_cvm_pit_fase5 import fetch, official_url
from auditoria_cvm_pit_fase7 import OUT, read_csv, key, tagname, normalize_number

EVIDENCE = OUT / 'fase10_evidencias_zip'
MAX_FILES = 120
MAX_UNCOMPRESSED = 120 * 1024 * 1024
MAX_XML = 20 * 1024 * 1024
DOC_FIELDS = ['dataset','ticker','cnpj_cia','dt_refer','versao','dt_receb','numero_sequencial']


def write_csv(name, fields, rows):
    p = OUT / name
    with p.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def xml_diagnostic(data, filename, base):
    result = []
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        return [{'member': filename, 'tag': '(XML_INVALIDO)', 'occurrences': 0,
                 'numeric_leaves': 0, 'example': str(exc)[:120], **base}]
    counts = Counter()
    nums = Counter()
    examples = {}
    for node in root.iter():
        tag = tagname(node.tag)
        counts[tag] += 1
        value = (node.text or '').strip()
        if value and len(node) == 0:
            if tag not in examples:
                examples[tag] = value[:90] if not re.search(r'(?i)cpf|email|senha|telefone', tag) else '[OCULTO]'
            if normalize_number(value) is not None:
                nums[tag] += 1
    for tag, n in counts.most_common():
        result.append({**base, 'member': filename, 'tag': tag,
                       'occurrences': n, 'numeric_leaves': nums[tag],
                       'example': examples.get(tag, '')})
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    phase8 = read_csv('fase8_documentos.csv')
    phase9 = read_csv('fase9_documentos_itr.csv')
    links = {key(r): r for r in read_csv('fase3_links_identificadores.csv')}
    old9 = {key(r): r for r in phase9}
    docs, members, tags, alerts = [], [], [], []
    if len(phase8) != 9:
        alerts.append({'issue': 'QUANTIDADE_DOCUMENTOS_ALTERADA', 'detail': f'Fase8={len(phase8)}, esperado historico=9'})
    for row in phase8:
        base = {k: row.get(k, '') for k in DOC_FIELDS}
        result = {**base, 'status': 'FALHA', 'hash_fase8': row.get('zip_sha256',''),
                  'hash_fase9': old9.get(key(row), {}).get('zip_sha256',''),
                  'hash_fase10': '', 'hash_igual_fase8': '', 'hash_igual_fase9': '',
                  'zip_saved': '', 'members': 0, 'xml_members': 0,
                  'numeric_leaf_nodes': 0, 'error': '', 'identity_authenticated': False,
                  'publication_authenticated': False, 'pit_approved': False}
        try:
            source = links.get(key(row))
            if not source:
                raise ValueError('LINK_FASE3_AUSENTE')
            url, seq = official_url(source.get('document_link', ''))
            if str(seq) != str(row.get('numero_sequencial', '')):
                raise ValueError('SEQUENCIAL_DIVERGENTE')
            payload = None
            for attempt in range(3):
                try:
                    data, _, status, _ = fetch(url)
                    if status != 200 or not zipfile.is_zipfile(io.BytesIO(data)):
                        raise ValueError(f'RESPOSTA_NAO_ZIP_{status}')
                    payload = data
                    break
                except (OSError, TimeoutError, ValueError, zipfile.BadZipFile):
                    if attempt == 2:
                        raise
                    time.sleep(3 * (attempt + 1))
            digest = sha(payload)
            result['hash_fase10'] = digest
            result['hash_igual_fase8'] = bool(result['hash_fase8']) and digest == result['hash_fase8']
            result['hash_igual_fase9'] = bool(result['hash_fase9']) and digest == result['hash_fase9']
            with zipfile.ZipFile(io.BytesIO(payload)) as z:
                infos = z.infolist()
                if len(infos) > MAX_FILES or sum(i.file_size for i in infos) > MAX_UNCOMPRESSED:
                    raise ValueError('ZIP_EXCEDE_LIMITES')
                if z.testzip() is not None:
                    raise ValueError('ZIP_CRC_INVALIDO')
                for info in infos:
                    if info.flag_bits & 1:
                        raise ValueError('MEMBRO_CRIPTOGRAFADO')
                    raw = z.read(info)
                    is_xml = info.filename.lower().endswith('.xml')
                    record = {**base, 'member': info.filename, 'bytes': len(raw),
                              'sha256': sha(raw), 'crc32': f'{info.CRC:08x}',
                              'is_xml': is_xml, 'xml_status': ''}
                    if is_xml:
                        result['xml_members'] += 1
                        if len(raw) > MAX_XML:
                            record['xml_status'] = 'XML_EXCEDE_LIMITE'
                        else:
                            diagnostic = xml_diagnostic(raw, info.filename, base)
                            tags.extend(diagnostic)
                            result['numeric_leaf_nodes'] += sum(int(t['numeric_leaves']) for t in diagnostic)
                            record['xml_status'] = 'DIAGNOSTICADO_SEM_HOMOLOGACAO'
                    members.append(record)
            filename = f"{base['dataset']}_{base['ticker']}_{base['dt_refer']}_v{base['versao']}_seq{seq}_{digest[:12]}.zip"
            path = EVIDENCE / filename
            path.write_bytes(payload)
            if sha(path.read_bytes()) != digest:
                raise IOError('EVIDENCIA_GRAVADA_COM_HASH_DIVERGENTE')
            result['zip_saved'] = str(path.relative_to(OUT))
            result['members'] = len(infos)
            result['status'] = 'CAPTURADO_NAO_HOMOLOGADO'
            if result['hash_fase8'] and not result['hash_igual_fase8']:
                alerts.append({**base, 'issue': 'HASH_DIFERENTE_FASE8', 'detail': f"fase8={result['hash_fase8']};fase10={digest}"})
            if result['hash_fase9'] and not result['hash_igual_fase9']:
                alerts.append({**base, 'issue': 'HASH_DIFERENTE_FASE9', 'detail': f"fase9={result['hash_fase9']};fase10={digest}"})
            if result['numeric_leaf_nodes'] == 0:
                alerts.append({**base, 'issue': 'SEM_FOLHAS_NUMERICAS_XML', 'detail': 'Inspecionar fase10_tags_xml.csv'})
        except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, ET.ParseError) as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'[:300]
            alerts.append({**base, 'issue': 'FALHA_CAPTURA_OU_INSPECAO', 'detail': result['error']})
        docs.append(result)
        print(base['ticker'], base['dt_refer'], base['versao'], result['status'], result['hash_igual_fase8'], flush=True)
    write_csv('fase10_documentos.csv', list(docs[0]) if docs else DOC_FIELDS + ['status'], docs)
    write_csv('fase10_membros.csv', DOC_FIELDS + ['member','bytes','sha256','crc32','is_xml','xml_status'], members)
    write_csv('fase10_tags_xml.csv', DOC_FIELDS + ['member','tag','occurrences','numeric_leaves','example'], tags)
    write_csv('fase10_alertas.csv', DOC_FIELDS + ['issue','detail'], alerts)
    summary = {'phase': 10, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
               'documents_expected_from_phase8': len(phase8), 'documents_captured': sum(x['status']=='CAPTURADO_NAO_HOMOLOGADO' for x in docs),
               'hash_matches_phase8': sum(x['hash_igual_fase8'] is True for x in docs),
               'hash_differences_phase8': sum(x['hash_igual_fase8'] is False and bool(x['hash_fase10']) for x in docs),
               'xml_numeric_leaf_nodes': sum(x['numeric_leaf_nodes'] for x in docs),
               'alerts': dict(Counter(x['issue'] for x in alerts)),
               'identity_authenticated': False, 'publication_authenticated': False, 'pit_approved': False,
               'limitations': ['Hashes divergentes nao provam alteracao historica sem comparar duas copias preservadas.',
                               'Diagnostico de folhas numericas nao certifica conta, periodo, escala ou demonstracao.',
                               'ZIPs da fase8 nao estavam preservados neste workflow; comparacao interna entre versoes depende de copias anteriores.',
                               'Arquivos capturados podem conter dados financeiros e cadastrais: revisar antes de compartilhar publicamente.',
                               'Nenhum arquivo do robo principal foi modificado.']}
    (OUT / 'fase10_resumo.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
