# =============================================================================
# B3 STOCK SELECTOR
# Arquivo: engines/turnaround.py
#
# TURNAROUND / ASYMMETRY ENGINE
#
# SCORE FUNDAMENTAL DO ESTUDO:
#
#   MARGEM_BRUTA    = 100%
#   MARGEM_LIQUIDA  = 0% (auditoria)
#   ROE             = 0% (auditoria)
#
# Todos com direção LOW.
#
# IMPORTANTE:
# - pesos baseados na força relativa da evidência encontrada;
# - não são pesos otimizados por retorno futuro;
# - retorno futuro jamais entra no cálculo;
# - o score pode ser usado pelo Ranking quando explicitamente autorizado
#   em config.py;
# - a origem científica e as limitações do estudo permanecem auditáveis.
# =============================================================================

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from config import OUTPUT_DIR, TURNAROUND_ENGINE
except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar config.py."
    ) from exc

try:
    from data.pit import PITContext
except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar PITContext."
    ) from exc


TURNAROUND_DIR = OUTPUT_DIR / "turnaround"
TURNAROUND_DIR.mkdir(parents=True, exist_ok=True)


class TurnaroundError(RuntimeError):
    pass


class TurnaroundIntegrityError(TurnaroundError):
    pass


class TurnaroundProductionError(TurnaroundError):
    pass


# =============================================================================
# FATORES E PESOS
# =============================================================================

RESEARCH_FACTORS = {
    "MARGEM_BRUTA": {
        "direction": "LOW",
        "role": "PRIMARY",
        "status": "CANDIDATE",
        "weight": 0.00,
    },
    "MARGEM_LIQUIDA": {
        "direction": "LOW",
        "role": "SECONDARY",
        "status": "RECURRENT_SIGNAL",
        "weight": 0.00,
    },
    "ROE": {
        "direction": "LOW",
        "role": "SECONDARY",
        "status": "WEAK_RECURRENT_SIGNAL",
        "weight": 0.00,
    },
}


RESEARCH_WEIGHTS = {
    "MARGEM_BRUTA": 0.00,
    "MARGEM_LIQUIDA": 0.00,
    "ROE": 0.00,
}


BLOCKED_RESEARCH_FACTORS = (
    "ESTOQUES",
    "DIVIDA_BRUTA_ATIVO_AS_OPPORTUNITY",
    "INTANGIVEL_ATIVO_AS_OPPORTUNITY",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_dataframe(
    df: pd.DataFrame,
    name: str,
) -> None:

    if not isinstance(df, pd.DataFrame):
        raise TurnaroundIntegrityError(
            f"{name} não é DataFrame."
        )

    if df.empty:
        raise TurnaroundIntegrityError(
            f"FAIL-SAFE: {name} está vazio."
        )


def _require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:

    _require_dataframe(df, name)

    missing = set(columns) - set(df.columns)

    if missing:
        raise TurnaroundIntegrityError(
            f"FAIL-SAFE: {name} sem colunas obrigatórias: "
            f"{sorted(missing)}"
        )


# =============================================================================
# RETORNO FUTURO
# =============================================================================

def assert_no_future_return(
    df: pd.DataFrame,
) -> None:

    forbidden_terms = (
        "FUTURE_RETURN",
        "RETORNO_FUTURO",
        "RETURN_FUTURE",
        "WINNER_LABEL",
        "FUTURE_WINNER",
    )

    allowed_audit_columns = {
        "FUTURE_RETURN_USED",
        "FUTURE_RETURN_USED_QUALITY",
        "FUTURE_RETURN_USED_TURNAROUND",
        "FUTURE_RETURN_USED_VALUATION",
    }

    forbidden = []

    for column in df.columns:

        normalized = str(column).upper()

        if normalized in allowed_audit_columns:

            values = df[column].dropna()

            if not values.empty:

                normalized_values = (
                    values.astype(str)
                    .str.strip()
                    .str.upper()
                )

                invalid = ~normalized_values.isin(
                    {"FALSE", "0"}
                )

                if invalid.any():
                    raise TurnaroundIntegrityError(
                        "FAIL-SAFE: coluna de auditoria "
                        f"{column} indica possível uso de retorno futuro."
                    )

            continue

        if any(
            term in normalized
            for term in forbidden_terms
        ):
            forbidden.append(column)

    if forbidden:
        raise TurnaroundIntegrityError(
            "FAIL-SAFE: retorno futuro ou label detectado "
            f"no Turnaround Engine: {forbidden}"
        )


# =============================================================================
# BASE
# =============================================================================

def prepare_turnaround_base(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Mantém emissores pendentes para auditoria; não os aprova."""
    _require_dataframe(fundamentals, "fundamentals")
    _require_columns(fundamentals, {"ISSUER_ID"}, "fundamentals")
    assert_no_future_return(fundamentals)
    result = fundamentals.copy()
    if result["ISSUER_ID"].isna().any() or result["ISSUER_ID"].duplicated().any():
        raise TurnaroundIntegrityError("ISSUER_ID ausente ou duplicado.")
    for col in ("MARGEM_BRUTA", "MARGEM_LIQUIDA", "ROE"):
        if col not in result:
            result[col] = np.nan
    return result


# =============================================================================
# PERCENTIS
# =============================================================================

def percentile_position(
    series: pd.Series,
) -> pd.Series:

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    valid = numeric.notna()

    if valid.sum() == 0:
        return result

    result.loc[valid] = (
        numeric.loc[valid]
        .rank(
            pct=True,
            method="average",
        )
    )

    return result.clip(0.0, 1.0)


def calculate_research_positions(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    for factor in RESEARCH_FACTORS:

        result[
            f"{factor}_PERCENTILE"
        ] = percentile_position(
            result[factor]
        )

    return result


# =============================================================================
# SINAIS LOW
# =============================================================================

def calculate_depressed_signals(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    for factor in RESEARCH_FACTORS:

        percentile = pd.to_numeric(
            result[
                f"{factor}_PERCENTILE"
            ],
            errors="coerce",
        )

        result[
            f"{factor}_DEPRESSED_SIGNAL"
        ] = (
            1.0 - percentile
        ).clip(
            lower=0.0,
            upper=1.0,
        )

    # Compatibilidade com relatórios anteriores.
    result[
        "GROSS_MARGIN_RESEARCH_SIGNAL"
    ] = result[
        "MARGEM_BRUTA_DEPRESSED_SIGNAL"
    ]

    return result


# =============================================================================
# ZONA DE MARGEM BRUTA
# =============================================================================

def classify_gross_margin_zone(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    percentile = pd.to_numeric(
        result["MARGEM_BRUTA_PERCENTILE"],
        errors="coerce",
    )

    zone = pd.Series(
        "UNKNOWN",
        index=result.index,
        dtype="object",
    )

    zone.loc[
        percentile.le(0.20)
    ] = "VERY_LOW"

    zone.loc[
        percentile.gt(0.20)
        & percentile.le(0.40)
    ] = "LOW"

    zone.loc[
        percentile.gt(0.40)
        & percentile.le(0.60)
    ] = "MID"

    zone.loc[
        percentile.gt(0.60)
        & percentile.le(0.80)
    ] = "HIGH"

    zone.loc[
        percentile.gt(0.80)
    ] = "VERY_HIGH"

    result[
        "GROSS_MARGIN_ZONE"
    ] = zone

    return result


# =============================================================================
# SAFETY CONTEXT
# =============================================================================

def _boolean_context(
    result: pd.DataFrame,
    source: str,
    target: str,
) -> None:

    if source in result.columns:

        result[target] = (
            result[source]
            .astype("boolean")
        )

    else:

        result[target] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="boolean",
        )


def calculate_safety_context(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    _boolean_context(
        result,
        "PL_POSITIVO",
        "TURNAROUND_PL_POSITIVE",
    )

    _boolean_context(
        result,
        "FCO_POSITIVO",
        "TURNAROUND_FCO_POSITIVE",
    )

    _boolean_context(
        result,
        "EBIT_POSITIVO",
        "TURNAROUND_EBIT_POSITIVE",
    )

    return result


# =============================================================================
# SCORE OPERACIONAL — 100% MARGEM BRUTA LOW
# =============================================================================

def calculate_experimental_score(df: pd.DataFrame) -> pd.DataFrame:
    """API preservada, mas pontuação de margem bruta baixa desativada."""
    result = df.copy()
    result["TURNAROUND_RESEARCH_WEIGHT_COVERAGE"] = 0.0
    result["TURNAROUND_RESEARCH_SCORE"] = np.nan
    result["TURNAROUND_RESEARCH_SCORE_VALID"] = False
    result["TURNAROUND_RESEARCH_SCORE_COMPLETE"] = np.nan
    return result


# =============================================================================
# CLASSIFICAÇÃO EXPERIMENTAL
# =============================================================================

def classify_research_candidate(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["TURNAROUND_RESEARCH_STATUS"] = "PENDENTE"
    return result


# =============================================================================
# PRODUÇÃO
# =============================================================================

def enforce_production_lock() -> None:
    """Não aceita habilitação por configuração do modelo MB_100."""
    if TURNAROUND_ENGINE.get("production_authorized", False):
        raise TurnaroundProductionError(
            "Produção bloqueada: critérios das 34 etapas não comprovados."
        )


# =============================================================================
# MOTOR PRINCIPAL
# =============================================================================

def run_turnaround_engine(
    fundamentals: pd.DataFrame,
    context: PITContext,
) -> pd.DataFrame:
    """Pipeline descritivo: sem ranking, aprovação ou sinal de compra."""
    enforce_production_lock()
    base = prepare_turnaround_base(fundamentals)
    result = calculate_research_positions(base)
    result = calculate_depressed_signals(result)
    result = classify_gross_margin_zone(result)
    result = calculate_safety_context(result)
    result = calculate_experimental_score(result)
    result = classify_research_candidate(result)
    result["TURNAROUND_PRODUCTION_AUTHORIZED"] = False
    result["TURNAROUND_OPERATIONAL_SIGNAL"] = "BLOCKED_STUDY34_PENDING"
    result["TURNAROUND_ENGINE_VERSION"] = "0.4.0-study34-pending"
    result["TURNAROUND_ENGINE_STATUS"] = "PENDENTE"
    result["TURNAROUND_SCORE_MODEL"] = "DISABLED_LEGACY_MB100"
    result["FORMATION_DATE_TURNAROUND"] = context.formation_date
    result["ACCOUNTING_CUTOFF_TURNAROUND"] = context.accounting_cutoff
    result["FUTURE_RETURN_USED_TURNAROUND"] = False
    result["TURNAROUND_RESEARCH_RANK"] = np.nan
    result["STUDY_34_STATUS"] = "PENDENTE"
    result["STUDY_34_EVIDENCE_VERIFIED"] = False
    result["STUDY_34_REASON"] = (
        "Aprovação depende de comprovação integral das 34 etapas."
    )
    assert_no_future_return(result)
    return result


# =============================================================================
# AUDITORIA
# =============================================================================

def audit_turnaround(result: pd.DataFrame) -> dict:
    _require_dataframe(result, "turnaround_result")
    return {
        "status": "PENDENTE",
        "mode": "AUDIT_ONLY",
        "score_model": "DISABLED_LEGACY_MB100",
        "issuers": int(len(result)),
        "valid_complete_scores": 0,
        "weights": {},
        "primary_factor": None,
        "primary_direction": None,
        "research_status_distribution": result[
            "TURNAROUND_RESEARCH_STATUS"
        ].value_counts(dropna=False).to_dict(),
        "gross_margin_zone_distribution": result[
            "GROSS_MARGIN_ZONE"
        ].value_counts(dropna=False).to_dict(),
        "production_authorized": False,
        "weights_optimized_on_future_return": False,
        "future_return_used": False,
        "buy_signal_generated": False,
        "operational_score_available": False,
    }


# =============================================================================
# SALVAR
# =============================================================================

def save_turnaround(
    result: pd.DataFrame,
    context: PITContext,
) -> dict:

    audit = audit_turnaround(
        result
    )

    date_tag = (
        context
        .formation_date
        .strftime("%Y%m%d")
    )

    result_path = (
        TURNAROUND_DIR
        / f"turnaround_research_{date_tag}.csv"
    )

    manifest_path = (
        TURNAROUND_DIR
        / f"turnaround_manifest_{date_tag}.json"
    )

    result.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "engine":
            "TURNAROUND_ENGINE",
        "version":
            "0.3.0",
        "created_at_utc":
            _utc_now_iso(),
        "formation_date":
            str(
                context
                .formation_date
                .date()
            ),
        "accounting_cutoff":
            str(
                context
                .accounting_cutoff
                .date()
            ),
        "research_factors":
            RESEARCH_FACTORS,
        "research_weights":
            RESEARCH_WEIGHTS,
        "blocked_research_factors":
            list(
                BLOCKED_RESEARCH_FACTORS
            ),
        **audit,
        "files": {
            "turnaround_research":
                str(result_path),
        },
    }

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "turnaround":
            result_path,
        "manifest":
            manifest_path,
    }


# =============================================================================
# SELF-TEST
# =============================================================================

def _self_test() -> bool:
    from datetime import datetime
    class FakeContext:
        formation_date = pd.Timestamp("2026-01-15")
        accounting_cutoff = pd.Timestamp("2025-12-31")
    sample = pd.DataFrame({
        "ISSUER_ID": ["1", "2"],
        "MARGEM_BRUTA": [0.1, 0.5],
        "MARGEM_LIQUIDA": [-0.1, 0.2],
        "ROE": [-0.2, 0.3],
        "FUNDAMENTAL_ENGINE_OK": [False, True],
        "INVESTABLE": [False, False],
    })
    result = run_turnaround_engine(sample, FakeContext())
    if len(result) != 2 or result["TURNAROUND_RESEARCH_SCORE_VALID"].any():
        raise TurnaroundError("SELF-TEST: score ou filtragem indevida.")
    if not result["STUDY_34_STATUS"].eq("PENDENTE").all():
        raise TurnaroundError("SELF-TEST: pendência ausente.")
    if result["TURNAROUND_RESEARCH_RANK"].notna().any():
        raise TurnaroundError("SELF-TEST: ranking não validado.")
    try:
        run_turnaround_engine(sample.assign(FUTURE_RETURN=[1.0, 2.0]), FakeContext())
    except TurnaroundIntegrityError:
        pass
    else:
        raise TurnaroundError("SELF-TEST: retorno futuro não bloqueado.")
    return True


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("B3 STOCK SELECTOR — TURNAROUND ENGINE")
    print("=" * 72)

    _self_test()

    print("Self-test: OK")
    production_authorized = bool(
        TURNAROUND_ENGINE.get(
            "production_authorized",
            False,
        )
    )
    print(
        "Modo:",
        "STUDY OPERATIONAL"
        if production_authorized
        else "RESEARCH ONLY",
    )
    print("Pontuação antiga: DESATIVADA")
    print("Direção dos três fatores: LOW")
    print("Retorno futuro usado no cálculo: NÃO")
    print("Pesos otimizados por retorno futuro: NÃO")
    print("ESTOQUES: BLOQUEADO")
    print("DÍVIDA/ATIVO como oportunidade: BLOQUEADO")
    print("INTANGÍVEL/ATIVO como oportunidade: BLOQUEADO")
    print("Sinal direto de compra: NÃO")
    print(
        "Score operacional no Ranking:",
        "AUTORIZADO"
        if production_authorized
        else "BLOQUEADO",
    )
    print("=" * 72)
