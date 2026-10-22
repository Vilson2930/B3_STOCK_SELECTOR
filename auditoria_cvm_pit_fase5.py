"""B3 STOCK SELECTOR — Fase 5: coleta diagnóstica dos 9 documentos divergentes.

Somente leitura da CVM. Lê relatórios das fases 3 e 4, consulta URLs HTTPS
permitidas, baixa respostas integralmente até limite rígido, valida formato ZIP/PDF,
inspeciona o índice ZIP sem extrair e descarta o conteúdo ao fim da execução.
Não autentica data de publicação, não altera CVM/accounting, não cria registry.
Executar: python auditoria_cvm_pit_fase5.py
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'cvm_pit_audit'
HOSTS = {'rad.cvm.gov.br', 'www.rad.cvm.gov.br'}
PATH = '/enetconsulta/frmdownloaddocumento.aspx'
MAX_BYTES = 25 * 1024 * 1024
MAX_ZIP_ENTRIES = 500
MAX_UNCOMPRESSED = 150 * 1024 * 1024
TIMEOUT = 35
ATTEMPTS = 2
FIELDS = ['dataset','ticker','cnpj_cia','dt_refer','versao','dt_receb','numero_sequencial',
          'phase4_status','url_https','final_url','attempts','http_status','content_type',
          'bytes_downloaded','sha256_complete','format_detected','zip_entries','zip_uncompressed_bytes',
          'zip_crc_ok','pdf_header','status','error','identity_authenticated',
          'publication_date_authenticated','pit_approved']


def read_csv(name):
    path = OUT / name
    if not path.is_file():
        raise FileNotFoundError(f'Relatório anterior ausente: {path}')
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def official_url(source):
    try:
        p = urllib.parse.urlsplit(source.strip())
        if (p.scheme.lower() not in ('http','https') or p.hostname not in HOSTS or
            p.username or p.password or p.port not in (None,80,443) or
            p.path.lower() != PATH):
            raise ValueError('HOST_CAMINHO_OU_ESQUEMA_INVALIDO')
        q = urllib.parse.parse_qs(p.query, keep_blank_values=True)
        seq, inst = q.get('NumeroSequencialDocumento', []), q.get('CodigoInstituicao', [])
        if (len(seq) != 1 or not re.fullmatch(r'\d{1,16}',seq[0]) or
            len(inst) != 1 or not re.fullmatch(r'\d{1,8}',inst[0])):
            raise ValueError('PARAMETROS_INVALIDOS')
        query = urllib.parse.urlencode({'CodigoInstituicao':inst[0], 'NumeroSequencialDocumento':seq[0]})
        return urllib.parse.urlunsplit(('https',p.hostname,p.path,query,'')),seq[0]
    except (ValueError, TypeError) as exc:
        raise ValueError(f'URL_NAO_PERMITIDA: {exc}') from exc


class RedirectOnlyCVM(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        p = urllib.parse.urlsplit(newurl)
        if p.scheme.lower() != 'https' or p.hostname not in HOSTS or p.port not in (None,443):
            raise urllib.error.URLError('REDIRECIONAMENTO_FORA_HTTPS_CVM')
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def fetch(url):
    opener = urllib.request.build_opener(RedirectOnlyCVM())
    request = urllib.request.Request(url,headers={
        'User-Agent':'B3-Stock-Selector-Documentary-Audit/1.0',
        'Accept':'application/zip,application/pdf,application/octet-stream,*/*',
    })
    with opener.open(request,timeout=TIMEOUT) as response:
        final = urllib.parse.urlsplit(response.geturl())
        if final.scheme.lower() != 'https' or final.hostname not in HOSTS:
            raise ValueError('DESTINO_FINAL_FORA_HTTPS_CVM')
        length = response.headers.get('Content-Length','')
        if length.isdigit() and int(length)>MAX_BYTES:
            raise ValueError('TAMANHO_DECLARADO_ACIMA_LIMITE')
        data = bytearray()
        while True:
            chunk = response.read(min(1024*1024,MAX_BYTES+1-len(data)))
            if not chunk:
                break
            data.extend(chunk)
            if len(data)>MAX_BYTES:
                raise ValueError('RESPOSTA_ACIMA_LIMITE')
        return bytes(data),str(response.geturl()),getattr(response,'status',None),response.headers.get('Content-Type','')


def classify(data):
    result = {'format_detected':'OUTRO','zip_entries':'','zip_uncompressed_bytes':'',
              'zip_crc_ok':'','pdf_header':False,'status':'FORMATO_NAO_RECONHECIDO'}
    if data.startswith(b'%PDF-'):
        result.update(format_detected='PDF',pdf_header=True,status='PDF_OBTIDO_NAO_AUTENTICADO')
    elif data.startswith(b'PK') and zipfile.is_zipfile(io.BytesIO(data)):
        result['format_detected']='ZIP'
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            info=z.infolist()
            result['zip_entries']=len(info)
            result['zip_uncompressed_bytes']=sum(i.file_size for i in info)
            if len(info)>MAX_ZIP_ENTRIES or result['zip_uncompressed_bytes']>MAX_UNCOMPRESSED:
                result['status']='ZIP_LIMITES_INTERNOS_EXCEDIDOS'
            elif any(i.flag_bits & 1 for i in info):
                result['status']='ZIP_COM_ENTRADAS_CRIPTOGRAFADAS'
            else:
                result['zip_crc_ok']=(z.testzip() is None)
                result['status']='ZIP_INTEGRO_NAO_AUTENTICADO' if result['zip_crc_ok'] else 'ZIP_CRC_INVALIDO'
    elif data.lstrip().lower().startswith((b'<!doctype html',b'<html')):
        result.update(format_detected='HTML',status='HTML_NAO_DOCUMENTO')
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    gaps=read_csv('fase3_divergencias.csv')
    links=read_csv('fase3_links_identificadores.csv')
    probes=read_csv('fase4_sondagens_enet.csv')
    by_key=lambda r:tuple((r.get(k) or '').strip() for k in ('dataset','cnpj_cia','dt_refer','versao'))
    link_map={by_key(r):r for r in links}
    probe_map={by_key(r):r for r in probes}
    rows=[]
    for gap in gaps:
        key=by_key(gap)
        source=link_map.get(key,{})
        prior=probe_map.get(key,{})
        row={k:'' for k in FIELDS}
        row.update({k:gap.get(k,'') for k in ('dataset','ticker','cnpj_cia','dt_refer','versao','dt_receb')})
        row.update(phase4_status=prior.get('probe_status',''),identity_authenticated=False,
                   publication_date_authenticated=False,pit_approved=False)
        try:
            url,seq=official_url(source.get('document_link',''))
            row.update(url_https=url,numero_sequencial=seq)
            for attempt in range(1,ATTEMPTS+1):
                row['attempts']=attempt
                try:
                    data,final,status,content_type=fetch(url)
                    row.update(final_url=final,http_status=status,content_type=content_type,
                               bytes_downloaded=len(data),sha256_complete=hashlib.sha256(data).hexdigest())
                    row.update(classify(data))
                    del data
                    break
                except (OSError,urllib.error.URLError,ValueError,TimeoutError,zipfile.BadZipFile) as exc:
                    row.update(status='FALHA_CONSULTA',error=f'{type(exc).__name__}: {str(exc)[:220]}')
                    if attempt<ATTEMPTS:
                        time.sleep(2)
        except ValueError as exc:
            row.update(status='URL_REJEITADA',error=str(exc))
        rows.append(row)
        print(f"{row['dataset']} {row['ticker']} {row['dt_refer']} v{row['versao']}: {row['status']}",flush=True)
    with (OUT/'fase5_documentos_prioritarios.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore');w.writeheader();w.writerows(rows)
    summary={'tool':'B3 STOCK SELECTOR — fase 5, coleta documental diagnóstica',
             'generated_at_utc':datetime.now(timezone.utc).isoformat(),
             'priority_records_expected':len(gaps),'priority_records_processed':len(rows),
             'status_counts':dict(Counter(r['status'] for r in rows)),
             'whole_response_hashes':sum(bool(r['sha256_complete']) for r in rows),
             'identity_authenticated':False,'publication_date_authenticated':False,
             'historical_pit_verified':False,'pit_approved':False,
             'limitations':['Verificação de CRC/assinatura ZIP não comprova identidade do emissor ou versão.',
                            'SHA-256 da resposta atual não prova disponibilidade histórica.',
                            'Não compara valores contábeis com documentos internos.',
                            'Não consulta metadados independentes de entrega/publicação.',
                            'Conteúdo integral é descartado após a inspeção; só metadados e hashes são exportados.',
                            'Nenhum dado é promovido a publication_registry.csv; FAIL-SAFE mantido.']}
    (OUT/'fase5_resumo.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
