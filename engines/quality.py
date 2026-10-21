"""B3 STOCK SELECTOR — Quality Engine, estudo científico de 34 etapas.

Este módulo AUDITA indicadores; não inventa pesos, limiares, aprovação ou
recomendação. Dados sem proveniência documental comprovada ficam PENDENTES.
Os critérios definitivos dependem da matriz auditada das 34 etapas.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from config import OUTPUT_DIR, QUALITY_ENGINE
    from data.pit import PITContext
except ImportError as exc:
    raise RuntimeError('FAIL-SAFE: dependências do Quality Engine indisponíveis') from exc

QUALITY_DIR = Path(OUTPUT_DIR) / 'quality'
QUALITY_DIR.mkdir(parents=True, exist_ok=True)

# Catálogo de indicadores históricos: não constitui fórmula de ranking.
QUALITY_FACTORS = {
    'ROA': 'HIGH', 'GIRO_ATIVO': 'HIGH', 'FCO_ATIVO': 'HIGH',
    'FCO_RECEITA': 'HIGH', 'DIVIDA_BRUTA_ATIVO': 'LOW',
    'DIVIDA_LIQUIDA_ATIVO': 'LOW', 'CAPITAL_GIRO_ATIVO': 'HIGH',
}
QUALITY_FLAGS = (
    'LUCRO_POSITIVO', 'EBIT_POSITIVO', 'FCO_POSITIVO',
    'CAPITAL_GIRO_POSITIVO', 'PL_POSITIVO',
)
# Não há pesos de fatores aprovados no estudo.
FACTOR_WEIGHT = None
FLAG_WEIGHT = None


class QualityError(RuntimeError):
    pass


class QualityIntegrityError(QualityError):
    pass


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_dataframe(df: pd.DataFrame, name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise QualityIntegrityError(f'{name} não é DataFrame')


def _require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    _require_dataframe(df, name)
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise QualityIntegrityError(f'{name} sem colunas: {missing}')


def assert_no_future_return(df: pd.DataFrame) -> None:
    _require_dataframe(df, 'quality_data')
    forbidden = ('FUTURE_RETURN', 'RETORNO_FUTURO', 'RETURN_FUTURE',
                 'FUTURE_WINNER', 'WINNER_LABEL')
    allowed = {'FUTURE_RETURN_USED', 'FUTURE_RETURN_USED_QUALITY'}
    for col in df.columns:
        key = str(col).upper()
        if key in allowed:
            vals = df[col].dropna().astype(str).str.strip().str.lower()
            if not vals.isin(('false', '0')).all():
                raise QualityIntegrityError(f'Indicador de retorno futuro: {col}')
        elif any(term in key for term in forbidden):
            raise QualityIntegrityError(f'Coluna de retorno futuro proibida: {col}')


def _explicit_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in ('true', 'sim', '1'):
            return True
        if key in ('false', 'não', 'nao', '0'):
            return False
    return pd.NA


def prepare_quality_base(fundamentals: pd.DataFrame) -> pd.DataFrame:
    _require_columns(fundamentals, ('ISSUER_ID',), 'fundamentals')
    assert_no_future_return(fundamentals)
    result = fundamentals.copy()
    if result['ISSUER_ID'].isna().any() or result['ISSUER_ID'].astype(str).str.strip().eq('').any():
        raise QualityIntegrityError('Identidade de emissor ausente')
    if result['ISSUER_ID'].duplicated().any():
        raise QualityIntegrityError('ISSUER_ID duplicado')
    # Não eliminar empresas pendentes: preservar universo e motivos para auditoria.
    return result


def percentile_score(series: pd.Series, direction: str = 'HIGH') -> pd.Series:
    """Utilitário exploratório, NÃO utilizado para aprovar ou classificar ações."""
    if direction not in ('HIGH', 'LOW'):
        raise QualityIntegrityError('Direção inválida')
    values = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)
    score = values.rank(pct=True, method='average')
    return score if direction == 'HIGH' else 1 - score


def _evidence_status(df: pd.DataFrame, context: PITContext) -> pd.Series:
    date_col = next((x for x in ('DT_PUBLICACAO_CVM_VERIFICADA',
                                 'PUBLICATION_DATE_VERIFIED') if x in df), None)
    doc_col = next((x for x in ('PROTOCOLO_CVM_DOCUMENTO',
                                'CVM_DOCUMENT_ID') if x in df), None)
    if date_col is None or doc_col is None:
        return pd.Series(False, index=df.index, dtype=bool)
    publication = pd.to_datetime(df[date_col], errors='coerce', utc=True)
    cutoff = pd.Timestamp(context.accounting_cutoff)
    formation = pd.Timestamp(context.formation_date)
    cutoff = cutoff.tz_localize('UTC') if cutoff.tzinfo is None else cutoff.tz_convert('UTC')
    formation = formation.tz_localize('UTC') if formation.tzinfo is None else formation.tz_convert('UTC')
    docs = df[doc_col].astype('string')
    return (publication.notna() & publication.le(cutoff) &
            publication.le(formation) & docs.notna() &
            docs.str.strip().ne('').fillna(False)).fillna(False).astype(bool)


def calculate_factor_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Preserva valores brutos; percentis antigos não são seleção validada."""
    result = df.copy()
    present = [name for name in QUALITY_FACTORS if name in result]
    for name in present:
        values = pd.to_numeric(result[name], errors='coerce')
        result[f'QUALITY_OBS_{name}'] = values.replace([np.inf, -np.inf], np.nan)
    result['QUALITY_FACTOR_COVERAGE'] = (
        result[[f'QUALITY_OBS_{name}' for name in present]].notna().sum(axis=1)
        / len(QUALITY_FACTORS)
        if present else pd.Series(0.0, index=result.index)
    )
    result['QUALITY_CONTINUOUS_SCORE'] = np.nan
    return result


def calculate_flag_score(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    present = [name for name in QUALITY_FLAGS if name in result]
    for name in present:
        result[f'QUALITY_OBS_{name}'] = result[name].map(_explicit_bool).astype('boolean')
    result['QUALITY_FLAG_COVERAGE'] = (
        result[[f'QUALITY_OBS_{name}' for name in present]].notna().sum(axis=1)
        / len(QUALITY_FLAGS)
        if present else pd.Series(0.0, index=result.index)
    )
    result['QUALITY_FLAG_SCORE'] = np.nan
    return result


def calculate_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """Nunca transforma indicadores exploratórios em ranking de investimento."""
    result = df.copy()
    for field in ('QUALITY_FACTOR_COVERAGE', 'QUALITY_FLAG_COVERAGE'):
        if field not in result:
            result[field] = 0.0
    result['QUALITY_DATA_COVERAGE'] = (
        pd.to_numeric(result['QUALITY_FACTOR_COVERAGE'], errors='coerce').fillna(0)
        + pd.to_numeric(result['QUALITY_FLAG_COVERAGE'], errors='coerce').fillna(0)
    ) / 2
    result['QUALITY_SCORE'] = np.nan
    result['QUALITY_SCORE_VALID'] = False
    result['QUALITY_RANK'] = np.nan
    result['QUALITY_STATUS'] = 'PENDENTE'
    result['QUALITY_REASON'] = (
        'Critérios, pesos e evidências das 34 etapas ainda não homologados.'
    )
    return result


def run_quality_engine(fundamentals: pd.DataFrame, context: PITContext) -> pd.DataFrame:
    if not isinstance(QUALITY_ENGINE, dict):
        raise QualityIntegrityError('QUALITY_ENGINE inválido em config.py')
    if not QUALITY_ENGINE.get('enabled', True):
        raise QualityError('Quality Engine desativado')
    result = calculate_quality_score(calculate_flag_score(
        calculate_factor_scores(prepare_quality_base(fundamentals))
    ))
    result['QUALITY_PUBLICATION_EVIDENCE_PRESENT'] = _evidence_status(result, context)
    result.loc[~result['QUALITY_PUBLICATION_EVIDENCE_PRESENT'], 'QUALITY_REASON'] = (
        'Publicação/protocolo CVM sem comprovação PIT; demais critérios pendentes.'
    )
    result['QUALITY_ENGINE_VERSION'] = 'study34-evidence-only-1.0'
    result['QUALITY_ENGINE_STATUS'] = 'RESEARCH_ONLY_NOT_APPROVED'
    result['FORMATION_DATE_QUALITY'] = context.formation_date
    result['ACCOUNTING_CUTOFF_QUALITY'] = context.accounting_cutoff
    result['FUTURE_RETURN_USED_QUALITY'] = False
    assert_no_future_return(result)
    return result.sort_values('ISSUER_ID', kind='mergesort').reset_index(drop=True)


def audit_quality(result: pd.DataFrame) -> dict:
    _require_columns(result, ('ISSUER_ID', 'QUALITY_SCORE', 'QUALITY_STATUS',
                              'QUALITY_DATA_COVERAGE'), 'quality_result')
    return {
        'status': 'PENDENTE', 'issuers': len(result),
        'approved_issuers': 0, 'valid_scores': int(result['QUALITY_SCORE'].notna().sum()),
        'mean_score': None, 'median_score': None,
        'mean_data_coverage': float(result['QUALITY_DATA_COVERAGE'].mean()) if len(result) else None,
        'factor_weight': None, 'flag_weight': None,
        'future_return_used': False, 'turnaround_signal_used': False,
        'reason': 'Matriz das 34 etapas e evidência documental incompletas',
    }


def save_quality(result: pd.DataFrame, context: PITContext, *, prefix: str = 'quality') -> dict:
    """Salva resultados de auditoria, jamais carteira aprovada."""
    audit = audit_quality(result)
    tag = pd.Timestamp(context.formation_date).strftime('%Y%m%d')
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    result_path = QUALITY_DIR / f'{prefix}_{tag}.csv'
    manifest_path = QUALITY_DIR / f'{prefix}_manifest_{tag}.json'
    result.to_csv(result_path, index=False, encoding='utf-8-sig')
    manifest = {
        'engine': 'QUALITY_ENGINE', 'created_at_utc': _utc_now_iso(),
        'formation_date': str(pd.Timestamp(context.formation_date).date()),
        'accounting_cutoff': str(pd.Timestamp(context.accounting_cutoff).date()),
        'factors_observed_not_weighted': QUALITY_FACTORS,
        'flags_observed_not_weighted': list(QUALITY_FLAGS),
        **audit, 'files': {'quality': str(result_path)},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'quality': result_path, 'manifest': manifest_path}


def _self_test() -> bool:
    class Context:
        formation_date = pd.Timestamp('2025-12-31')
        accounting_cutoff = pd.Timestamp('2025-09-30')
    sample = pd.DataFrame({
        'ISSUER_ID': ['1', '2'], 'INVESTABLE': [False, 'False'],
        'ROA': [0.2, 0.1], 'FCO_POSITIVO': [True, 'False'],
        'DT_PUBLICACAO_CVM_VERIFICADA': ['2025-08-01', '2025-10-01'],
        'PROTOCOLO_CVM_DOCUMENTO': ['doc1', 'doc2'],
    })
    result = run_quality_engine(sample, Context())
    assert len(result) == 2
    assert result['QUALITY_SCORE'].isna().all()
    assert result['QUALITY_STATUS'].eq('PENDENTE').all()
    assert result['QUALITY_PUBLICATION_EVIDENCE_PRESENT'].tolist() == [True, False]
    assert result.loc[1, 'QUALITY_OBS_FCO_POSITIVO'] == False
    assert audit_quality(result)['approved_issuers'] == 0
    try:
        run_quality_engine(sample.assign(FUTURE_RETURN=[0.1, 0.2]), Context())
    except QualityIntegrityError:
        pass
    else:
        raise AssertionError('Look-ahead contaminado não foi bloqueado')
    return True


if __name__ == '__main__':
    print('B3 STOCK SELECTOR — QUALITY ENGINE — 34 ETAPAS')
    print('Autoteste:', 'OK' if _self_test() else 'FALHOU')
    print('Status: PENDENTE; nenhum ranking ou aprovação autorizado')
