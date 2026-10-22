"""B3 STOCK SELECTOR — Fase 9: diagnostico ITR e filtros semanticos.

READ-ONLY. Requer relatorios das fases 3, 7 e 8, ZIPs CVM e modulo fase5.
Nao autentica publicacao, nao aprova PIT e nao altera dados do robo.
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
from decimal import Decimal
from pathlib import Path
from xml.etree import ElementTree as ET
from auditoria_cvm_pit_fase5 import fetch, official_url
from auditoria_cvm_pit_fase7 import OUT, read_csv, write_csv, key, tagname, normalize_number

NEW_TAGS = {'TRIMESTREATUAL','TRIMESTREANTERIOR','EXERCICIOANTERIOR',
            'ACUMULADOEXERCICIOATUAL','ACUMULADOEXERCICIOANTERIOR',
            'ACUMULADOATUALEXERCICIO'}
MAX_FILES=120
MAX_UNCOMPRESSED=120*1024*1024
MAX_XML=20*1024*1024
MAX_EXTRACT=120000


def itr_extract(data, member):
    if len(data)>MAX_XML: return [], 'XML_EXCEDE_LIMITE'
    try: root=ET.fromstring(data)
    except ET.ParseError as exc: return [], 'XML_INVALIDO:'+str(exc)[:90]
    rows=[]
    # Visita somente descendentes de CONTA/CONTAFIXA; nao infere escopo/escala.
    for account in root.iter():
        if tagname(account.tag) not in ('CONTA','CONTAFIXA'): continue
        direct={tagname(c.tag):(c.text or '').strip() for c in account}
        code=direct.get('CODIGOCONTA','')
        desc=direct.get('DESCRICAOCONTA','')
        if not code: continue
        for child in account.iter():
            t=tagname(child.tag)
            if t not in NEW_TAGS: continue
            v=normalize_number(child.text)
            if v is None: continue
            rows.append({'member':member,'xml_tag':t,'xml_account_code':code,
                         'xml_account_description':desc[:160], 'xml_value':str(v),
                         'xml_value_sha256':hashlib.sha256((child.text or '').encode()).hexdigest()})
            if len(rows)>=MAX_EXTRACT: return rows,'LIMITE_EXTRAÇÃO'
    return rows,'EXTRAIDO'


def semantic_check(r):
    """Apenas rejeita incompatibilidades verificaveis; nao certifica equivalencia."""
    reasons=[]
    if r.get('comparison_class')!='CODIGO_NUMERO_VERSAO_DIFERENTE': reasons.append('CODIGO_NAO_COMPROVADO')
    if r.get('xml_account_code','').strip()!=r.get('csv_account_code','').strip(): reasons.append('CODIGO_DIFERENTE')
    a=normalize_number(r.get('xml_value')); b=normalize_number(r.get('csv_value'))
    if a is None or b is None or a!=b: reasons.append('VALOR_DIFERENTE')
    if r.get('versao','').strip()==r.get('csv_version','').strip(): reasons.append('VERSAO_INESPERADAMENTE_IGUAL')
    if r.get('csv_end') and r.get('csv_end')!=r.get('dt_refer'): reasons.append('FIM_PERIODO_CSV_DIFERENTE')
    # Nao existe no relatorio fase8 identificacao XML comprovada de demonstracao,
    # escopo, moeda/escala, nem periodo de cada valor. Permanecem pendentes.
    reasons += ['DEMONSTRACAO_XML_NAO_COMPROVADA','ESCOPO_XML_NAO_COMPROVADO',
                'ESCALA_XML_NAO_COMPROVADA','PERIODO_VALOR_XML_NAO_COMPROVADO',
                'VERSOES_DIFERENTES','PUBLICACAO_PIT_NAO_AUTENTICADA']
    return reasons


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    phase8=read_csv('fase8_documentos.csv')
    matches=read_csv('fase8_comparacoes_entre_versoes.csv')
    links={key(r):r for r in read_csv('fase3_links_identificadores.csv')}
    target=[r for r in phase8 if int(r['xml_values'])==0]
    if len(phase8)!=9 or len(target)!=5:
        raise RuntimeError(f'Esperados 9 documentos e 5 com zero XML; obtidos {len(phase8)} e {len(target)}. Revisar antes de prosseguir.')
    docs=[]; values=[]; issues=[]
    for r in target:
        base={k:r.get(k,'') for k in ('dataset','ticker','cnpj_cia','dt_refer','versao','dt_receb','numero_sequencial')}
        result={**base,'status':'PENDENTE','xml_members':0,'new_tag_values':0,'zip_sha256':'','xml_statuses':'','error':'',
                'semantics_verified':False,'publication_authenticated':False,'pit_approved':False}
        try:
            source=links.get(key(r))
            if source is None: raise ValueError('LINK_FASE3_AUSENTE')
            url,seq=official_url(source.get('document_link',''))
            if seq!=r.get('numero_sequencial'): raise ValueError('SEQUENCIAL_DIVERGENTE_FASE8')
            payload=None
            for attempt in range(3):
                try:
                    data,_,status,_=fetch(url)
                    if status!=200 or not zipfile.is_zipfile(io.BytesIO(data)):
                        raise ValueError(f'RESPOSTA_INVALIDA_{status}')
                    payload=data;break
                except (OSError,TimeoutError,ValueError,zipfile.BadZipFile):
                    if attempt==2:raise
                    time.sleep(3*(attempt+1))
            digest=hashlib.sha256(payload).hexdigest()
            result['zip_sha256']=digest
            if r.get('zip_sha256') and digest!=r['zip_sha256']:
                raise ValueError('HASH_ZIP_DIFERENTE_FASE8')
            states=[]
            with zipfile.ZipFile(io.BytesIO(payload)) as z:
                infos=z.infolist()
                if len(infos)>MAX_FILES or sum(x.file_size for x in infos)>MAX_UNCOMPRESSED:raise ValueError('ZIP_EXCEDE_LIMITES')
                if z.testzip() is not None:raise ValueError('CRC_INVALIDO')
                for info in infos:
                    if not info.filename.lower().endswith('.xml') or info.flag_bits&1:continue
                    if info.file_size>MAX_XML:
                        states.append(info.filename+':XML_EXCEDE_LIMITE');continue
                    rows,state=itr_extract(z.read(info),info.filename)
                    result['xml_members']+=1
                    states.append(info.filename+':'+state)
                    for v in rows:
                        values.append({**base,'zip_sha256':digest,**v})
                        result['new_tag_values']+=1
            result['xml_statuses']='; '.join(states)[:800]
            result['status']='EXTRAIDO_SEM_HOMOLOGACAO' if result['new_tag_values'] else 'SEM_VALORES_EXTRAIDOS'
            if not result['new_tag_values']:issues.append({**base,'issue':'ZERO_VALORES_APOS_EXPANSAO','detail':result['xml_statuses']})
        except (OSError,ValueError,RuntimeError,zipfile.BadZipFile) as exc:
            result['status']='FALHA';result['error']=f'{type(exc).__name__}:{str(exc)[:200]}'
            issues.append({**base,'issue':'FALHA','detail':result['error']})
        docs.append(result)
        print(base['ticker'],base['dt_refer'],base['versao'],result['new_tag_values'],result['status'],flush=True)
    sem=[];counts=Counter()
    for r in matches:
        reasons=semantic_check(r)
        counts.update(reasons)
        sem.append({**r,'semantic_gate':'NAO_HOMOLOGADO','blocking_reasons':'|'.join(reasons),
                    'semantics_verified':False,'reconciled':False,'pit_approved':False})
    write_csv('fase9_documentos_itr.csv',list(docs[0]),docs)
    write_csv('fase9_valores_itr.csv',['dataset','ticker','cnpj_cia','dt_refer','versao','dt_receb','numero_sequencial',
              'zip_sha256','member','xml_tag','xml_account_code','xml_account_description','xml_value','xml_value_sha256'],values)
    fields=list(matches[0]) if matches else []
    write_csv('fase9_filtro_semantico.csv',fields+['semantic_gate','blocking_reasons','pit_approved'],sem)
    write_csv('fase9_alertas.csv',['dataset','ticker','cnpj_cia','dt_refer','versao','dt_receb','numero_sequencial','issue','detail'],issues)
    summary={'phase':9,'generated_at_utc':datetime.now(timezone.utc).isoformat(),
             'documents_reexamined':len(docs),'documents_with_values':sum(d['new_tag_values']>0 for d in docs),
             'new_tag_values':len(values),'cross_version_candidates_screened':len(sem),
             'semantic_reconciliations_approved':0,'publication_authenticated':False,'pit_approved':False,
             'blocking_reason_counts':dict(counts),'issues':dict(Counter(x['issue'] for x in issues)),
             'limitations':['Campos ITR adicionais identificados no inventario de tags da fase 8.',
              'Valores extraidos nao tem escopo, demonstracao, escala e periodo semanticamente certificados.',
              'Comparacoes fase 8 sao apenas amostra de candidatos entre versoes distintas.',
              'Data DT_RECEB nao equivale automaticamente a disponibilizacao publica historica.',
              'Nenhum dado do robo principal ou registro PIT foi modificado.']}
    (OUT/'fase9_resumo.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
