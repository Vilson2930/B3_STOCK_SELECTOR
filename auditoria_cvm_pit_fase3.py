"""B3 STOCK SELECTOR — fase 3: auditoria PIT exploratoria, estritamente read-only.

Consome outputs/cvm_pit_audit/{documentos,correspondencias}.csv produzidos pelo
auditoria_cvm_pit.py. Nao consulta rede, nao autentica documento, nao cria
publication_registry.csv e NUNCA promove PIT_VALID. Execucao independente:
    python auditoria_cvm_pit_fase3.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'cvm_pit_audit'
EXPECTED = {'DFP', 'ITR'}
OFFICIAL_HOSTS = {'rad.cvm.gov.br', 'www.rad.cvm.gov.br'}


def read_csv(filename: str) -> list[dict]:
    path = OUT / filename
    if not path.is_file():
        raise FileNotFoundError(f'Relatorio da fase 2 ausente: {path}')
    with path.open(newline='', encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))


def write_csv(filename: str, fields: list[str], records: list[dict]) -> None:
    with (OUT / filename).open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def parse_date(value: str) -> date | None:
    try:
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value or ''):
            return None
        return date.fromisoformat(value)
    except ValueError:
        return None


def key(record: dict) -> tuple[str, str, str, str]:
    return tuple(record.get(x, '').strip() for x in ('dataset', 'cnpj_cia', 'dt_refer', 'versao'))


def inspect_link(value: str) -> dict:
    result = {'scheme': '', 'host': '', 'host_cvm': False, 'https': False,
              'numero_sequencial': '', 'codigo_instituicao': '', 'identifier_status': ''}
    try:
        url = urlsplit(value.strip())
        host = (url.hostname or '').lower()
        query = parse_qs(url.query, keep_blank_values=True)
        numbers = query.get('NumeroSequencialDocumento', [])
        institutions = query.get('CodigoInstituicao', [])
        seq = numbers[0] if len(numbers) == 1 else ''
        institution = institutions[0] if len(institutions) == 1 else ''
        valid_seq = bool(re.fullmatch(r'[0-9]+', seq))
        result.update(scheme=url.scheme.lower(), host=host,
                      host_cvm=host in OFFICIAL_HOSTS, https=url.scheme.lower() == 'https',
                      numero_sequencial=seq, codigo_instituicao=institution,
                      identifier_status='FORMATO_PRESENTE_NAO_AUTENTICADO' if valid_seq else 'AUSENTE_OU_INVALIDO')
    except ValueError:
        result['identifier_status'] = 'URL_MALFORMADA'
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    documents = read_csv('documentos.csv')
    matches = read_csv('correspondencias.csv')
    if not documents or not matches:
        raise RuntimeError('Fase 2 sem documentos/correspondencias: auditoria fase 3 nao pode prosseguir.')
    match_index = defaultdict(list)
    for match in matches:
        match_index[key(match)].append(match)
    versions = defaultdict(list)
    link_records = []
    gaps = []
    scenarios = []
    issues = []
    identifiers = defaultdict(set)

    for doc in documents:
        k = key(doc)
        receipt = parse_date(doc.get('dt_receb', ''))
        reference = parse_date(doc.get('dt_refer', ''))
        info = inspect_link(doc.get('link_doc', ''))
        associated = match_index.get(k, [])
        count = sum(int(m.get('accounting_rows', 0) or 0) for m in associated)
        label = doc.get('ticker_candidate', '')
        base = {'dataset': k[0], 'ticker': label, 'cnpj_cia': k[1],
                'dt_refer': k[2], 'versao': k[3], 'dt_receb': doc.get('dt_receb', '')}
        versions[(k[0], k[1], k[2])].append((k[3], receipt, count, info['numero_sequencial']))
        link_records.append({**base, **info, 'accounting_rows': count,
                             'document_link': doc.get('link_doc', ''),
                             'publication_authenticated': False})
        if info['numero_sequencial']:
            identifiers[info['numero_sequencial']].add(k)
        if count == 0 or len(associated) != 1:
            gaps.append({**base, 'accounting_rows': count,
                         'match_records': len(associated),
                         'issue': 'SEM_LINHAS_CONTABEIS' if count == 0 else 'CHAVE_AMBIGUA'})
        if k[0] not in EXPECTED or not receipt or not reference or receipt < reference:
            issues.append({**base, 'issue': 'METADADOS_INCONSISTENTES'})
        if not info['host_cvm']:
            issues.append({**base, 'issue': 'DOMINIO_FORA_DA_LISTA_CVM'})
        if not info['https']:
            issues.append({**base, 'issue': 'LINK_HTTP_SEM_AUTENTICACAO_TRANSPORTE'})
        if info['identifier_status'] != 'FORMATO_PRESENTE_NAO_AUTENTICADO':
            issues.append({**base, 'issue': 'IDENTIFICADOR_AUSENTE_OU_INVALIDO'})
        # Simulacoes mecanicas: NAO provam data de divulgacao publica.
        if receipt:
            for cutoff in (receipt - timedelta(days=1), receipt, receipt + timedelta(days=1)):
                scenarios.append({**base, 'cutoff': cutoff.isoformat(),
                                  'receipt_based_eligible_candidate': cutoff >= receipt,
                                  'public_availability_verified': False,
                                  'pit_approved': False})

    version_rows = []
    for (dataset, cnpj, reference), items in sorted(versions.items()):
        numeric = sorted({int(v) for v, _, _, _ in items if v.isdigit()})
        observed = {int(v) for v, _, _, _ in items if v.isdigit()}
        missing = [v for v in range(1, max(numeric) + 1) if v not in observed] if numeric else []
        chronology = sorted((int(v), d) for v, d, _, _ in items if v.isdigit() and d)
        nonmonotonic = any(b_date < a_date for (_, a_date), (_, b_date) in zip(chronology, chronology[1:]))
        versions_without_accounting = sorted(v for v, _, count, _ in items if count == 0)
        version_rows.append({'dataset': dataset, 'cnpj_cia': cnpj, 'dt_refer': reference,
                             'versions_in_main': '|'.join(str(v) for v in numeric),
                             'missing_version_numbers': '|'.join(map(str, missing)),
                             'versions_without_accounting': '|'.join(versions_without_accounting),
                             'receipt_nonmonotonic_by_version': nonmonotonic,
                             'status': 'INVESTIGAR' if missing or nonmonotonic or versions_without_accounting else 'CONSISTENCIA_ESTRUTURAL_SEM_AUTENTICACAO'})
        if nonmonotonic:
            issues.append({'dataset': dataset, 'cnpj_cia': cnpj, 'dt_refer': reference,
                           'issue': 'RECEBIMENTO_NAO_MONOTONICO_POR_VERSAO'})

    collisions = [{'numero_sequencial': seq, 'keys': '|'.join('/'.join(k) for k in sorted(keys)),
                   'key_count': len(keys)} for seq, keys in identifiers.items() if len(keys) > 1]
    for item in collisions:
        issues.append({'issue': 'IDENTIFICADOR_COMPARTILHADO_ENTRE_CHAVES', **item})

    write_csv('fase3_links_identificadores.csv', ['dataset', 'ticker', 'cnpj_cia', 'dt_refer',
              'versao', 'dt_receb', 'scheme', 'host', 'host_cvm', 'https', 'numero_sequencial',
              'codigo_instituicao', 'identifier_status', 'accounting_rows', 'document_link',
              'publication_authenticated'], link_records)
    write_csv('fase3_divergencias.csv', ['dataset', 'ticker', 'cnpj_cia', 'dt_refer', 'versao',
              'dt_receb', 'accounting_rows', 'match_records', 'issue'], gaps)
    write_csv('fase3_versoes.csv', ['dataset', 'cnpj_cia', 'dt_refer', 'versions_in_main',
              'missing_version_numbers', 'versions_without_accounting',
              'receipt_nonmonotonic_by_version', 'status'], version_rows)
    write_csv('fase3_cortes_historicos.csv', ['dataset', 'ticker', 'cnpj_cia', 'dt_refer',
              'versao', 'dt_receb', 'cutoff', 'receipt_based_eligible_candidate',
              'public_availability_verified', 'pit_approved'], scenarios)
    write_csv('fase3_alertas.csv', ['dataset', 'ticker', 'cnpj_cia', 'dt_refer', 'versao',
              'dt_receb', 'issue', 'numero_sequencial', 'keys', 'key_count'], issues)
    summary = {
        'tool': 'B3 STOCK SELECTOR - auditoria fase 3 de evidencias PIT',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'documents': len(documents), 'missing_accounting_matches': len(gaps),
        'version_groups': len(version_rows),
        'version_groups_requiring_investigation': sum(x['status'] == 'INVESTIGAR' for x in version_rows),
        'links_on_known_cvm_host': sum(x['host_cvm'] for x in link_records),
        'links_https': sum(x['https'] for x in link_records),
        'sequential_identifiers_present': sum(bool(x['numero_sequencial']) for x in link_records),
        'sequential_identifier_collisions': len(collisions),
        'simulated_cutoffs': len(scenarios), 'alerts_by_type': dict(Counter(x['issue'] for x in issues)),
        'source_authenticated': False, 'public_availability_verified': False,
        'pit_approved': False,
        'limitations': [
            'Os links e NumeroSequencialDocumento sao apenas metadados extraidos dos CSVs; nao houve consulta ao ENET.',
            'HTTP em dominio da CVM nao comprova HTTPS, identidade documental nem acesso historico.',
            'DT_RECEB nao foi homologado como instante de disponibilidade publica.',
            'Cortes historicos testam somente uma regra hipotetica baseada em DT_RECEB; nao autorizam backtest.',
            'A ausencia de uma versao nos CSVs atuais nao prova ausencia na divulgacao historica.',
            'Nao existe verificacao de conteudo documental por hash/assinatura/protocolo externo.',
            'Nao gera publication_registry.csv nem modifica o robo; FAIL-SAFE permanece ativo.'
        ],
    }
    (OUT / 'fase3_resumo.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('documents', 'missing_accounting_matches',
          'version_groups_requiring_investigation', 'simulated_cutoffs', 'pit_approved')},
          ensure_ascii=False, indent=2))
    print('PIT PENDENTE: auditoria estrutural, sem autenticacao documental externa.')


if __name__ == '__main__':
    main()
