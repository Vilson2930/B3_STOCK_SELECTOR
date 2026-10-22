"""B3 STOCK SELECTOR - fase 4: sondagem documental CVM/ENET (read-only).

Executar APOS fases 1-3: python auditoria_cvm_pit_fase4.py
Faz requisicoes GET limitadas a URLs oficiais HTTPS para avaliar acessibilidade.
Nao autentica datas historicas, nao baixa/guarda documentos, nao escreve registro PIT.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'cvm_pit_audit'
HOSTS = {'rad.cvm.gov.br', 'www.rad.cvm.gov.br'}
MAX_BYTES = 65536
TIMEOUT = 18
FIELDS = ['priority', 'dataset', 'ticker', 'cnpj_cia', 'dt_refer', 'versao', 'dt_receb',
          'numero_sequencial', 'source_url', 'requested_https_url', 'final_url',
          'http_status', 'content_type', 'sample_bytes', 'sample_sha256', 'response_kind',
          'probe_status', 'error', 'independent_metadata_verified', 'historical_pit_verified']


def load_csv(filename: str) -> list[dict]:
    path = OUT / filename
    if not path.is_file():
        raise FileNotFoundError(f'Falta relatorio anterior: {path}')
    with path.open('r', newline='', encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))


def safe_url(source: str) -> tuple[str, str]:
    """Nao aceita hosts arbitrarios, credenciais, portas ou caminhos inesperados."""
    try:
        parsed = urllib.parse.urlsplit(source.strip())
        host = (parsed.hostname or '').lower()
        if (host not in HOSTS or parsed.scheme.lower() not in ('http', 'https')
                or parsed.username or parsed.password or parsed.port not in (None, 443, 80)
                or parsed.path.lower() != '/enetconsulta/frmdownloaddocumento.aspx'):
            return '', 'URL_FORA_ESCOPO'
        qs = urllib.parse.parse_qs(parsed.query)
        seqs = qs.get('NumeroSequencialDocumento', [])
        inst = qs.get('CodigoInstituicao', [])
        if len(seqs) != 1 or not re.fullmatch(r'\d{1,16}', seqs[0]):
            return '', 'IDENTIFICADOR_INVALIDO'
        if len(inst) != 1 or not re.fullmatch(r'\d{1,8}', inst[0]):
            return '', 'INSTITUICAO_INVALIDA'
        # Reconstroi a query apenas com parametros documentais reconhecidos.
        query = urllib.parse.urlencode({'CodigoInstituicao': inst[0],
                                        'NumeroSequencialDocumento': seqs[0]})
        return urllib.parse.urlunsplit(('https', host, parsed.path, query, '')), ''
    except (ValueError, TypeError):
        return '', 'URL_MALFORMADA'


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme.lower() != 'https' or (parsed.hostname or '').lower() not in HOSTS:
            raise urllib.error.URLError('REDIRECIONAMENTO_FORA_HOST_HTTPS_OFICIAL')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def probe(url: str) -> dict:
    result = dict(final_url='', http_status='', content_type='', sample_bytes=0,
                  sample_sha256='', response_kind='', probe_status='', error='')
    opener = urllib.request.build_opener(SafeRedirect())
    request = urllib.request.Request(url, headers={
        'User-Agent': 'B3-Stock-Selector-Research-Audit/1.0',
        'Accept': 'application/pdf,application/zip,text/html,*/*',
        'Range': f'bytes=0-{MAX_BYTES-1}',
    })
    try:
        with opener.open(request, timeout=TIMEOUT) as response:
            sample = response.read(MAX_BYTES)
            result['final_url'] = response.geturl()
            result['http_status'] = response.status
            result['content_type'] = response.headers.get('Content-Type', '')[:150]
            result['sample_bytes'] = len(sample)
            result['sample_sha256'] = hashlib.sha256(sample).hexdigest()
            prefix = sample[:4096].lower()
            ctype = result['content_type'].lower()
            if sample.startswith(b'%PDF-'):
                result['response_kind'] = 'PDF_PREFIX'
            elif sample.startswith(b'PK\x03\x04'):
                result['response_kind'] = 'ZIP_PREFIX'
            elif b'<html' in prefix or b'<!doctype html' in prefix or 'text/html' in ctype:
                result['response_kind'] = 'HTML_NAO_DOCUMENTO_VERIFICADO'
            else:
                result['response_kind'] = 'CONTEUDO_NAO_CLASSIFICADO'
            result['probe_status'] = 'ACESSIVEL_SEM_AUTENTICACAO' if result['response_kind'] in ('PDF_PREFIX', 'ZIP_PREFIX') else 'INCONCLUSIVO'
    except urllib.error.HTTPError as exc:
        result['http_status'] = exc.code
        result['probe_status'] = 'HTTP_ERRO'
        result['error'] = f'HTTP {exc.code}: {exc.reason}'[:240]
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        result['probe_status'] = 'ERRO_REDE_OU_REDIRECIONAMENTO'
        result['error'] = f'{type(exc).__name__}: {exc}'[:240]
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    documents = load_csv('documentos.csv')
    divergences = load_csv('fase3_divergencias.csv')
    gap_keys = {(r.get('dataset',''), r.get('cnpj_cia',''), r.get('dt_refer',''), r.get('versao','')) for r in divergences}
    ordered = sorted(documents, key=lambda d: (0 if (d.get('dataset',''), d.get('cnpj_cia',''), d.get('dt_refer',''), d.get('versao','')) in gap_keys else 1,
                                               d.get('dataset',''), d.get('cnpj_cia',''), d.get('dt_refer',''), d.get('versao','')))
    results = []
    for index, doc in enumerate(ordered, 1):
        k = (doc.get('dataset',''), doc.get('cnpj_cia',''), doc.get('dt_refer',''), doc.get('versao',''))
        source = doc.get('link_doc','')
        url, invalid = safe_url(source)
        parsed = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query) if url else {}
        row = {'priority': 'DIVERGENCIA' if k in gap_keys else 'AMOSTRA',
               'dataset': k[0], 'ticker': doc.get('ticker_candidate',''), 'cnpj_cia': k[1],
               'dt_refer': k[2], 'versao': k[3], 'dt_receb': doc.get('dt_receb',''),
               'numero_sequencial': (parsed.get('NumeroSequencialDocumento') or [''])[0],
               'source_url': source, 'requested_https_url': url,
               'independent_metadata_verified': False, 'historical_pit_verified': False}
        if url:
            row.update(probe(url))
        else:
            row.update(dict(final_url='', http_status='', content_type='', sample_bytes=0,
                            sample_sha256='', response_kind='', probe_status='URL_INVALIDA', error=invalid))
        results.append(row)
        print(f'[{index}/{len(ordered)}] {row["priority"]} {row["ticker"]} {k[0]} {k[2]} v{k[3]} -> {row["probe_status"]}', flush=True)
        if index < len(ordered):
            time.sleep(0.2)
    report = OUT / 'fase4_sondagens_enet.csv'
    with report.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    summary = {'tool': 'B3 STOCK SELECTOR - fase 4, sondagem read-only de ENET',
               'generated_at_utc': datetime.now(timezone.utc).isoformat(),
               'documents_expected': len(documents), 'probes_completed': len(results),
               'priority_divergence_count': sum(r['priority'] == 'DIVERGENCIA' for r in results),
               'probe_status_counts': dict(Counter(r['probe_status'] for r in results)),
               'response_kind_counts': dict(Counter(r['response_kind'] for r in results)),
               'independent_metadata_verified': False, 'historical_pit_verified': False,
               'source_authenticated': False, 'pit_approved': False,
               'limitations': [
                   'GET de documento nao valida data historica de divulgacao.',
                   'HTML, PDF e ZIP podem representar erro, login, redirecionamento ou arquivo diferente.',
                   'Nao houve comparacao independente de protocolo, versao e data com metadados da consulta publica.',
                   'SHA256 e somente da amostra de ate 64 KiB da resposta atual, NAO do documento integral.',
                   'URLs HTTP de origem foram convertidas para HTTPS apenas para sondagem; falha HTTPS nao prova ausencia documental.',
                   'Nao gera registro de publicacao nem altera os 34 estagios do robo.'
               ]}
    (OUT / 'fase4_resumo.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary['probe_status_counts'], ensure_ascii=False, indent=2))
    print('PIT PENDENTE - nenhuma evidencia homologada.')


if __name__ == '__main__':
    main()
