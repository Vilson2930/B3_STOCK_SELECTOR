# =============================================================================
# B3 STOCK SELECTOR
# Arquivo: engines/turnaround.py
#
# TURNAROUND / ASYMMETRY ENGINE
#
# SCORE FUNDAMENTAL DO ESTUDO:
#
#   MARGEM_BRUTA    = 50%
#   MARGEM_LIQUIDA  = 30%
#   ROE             = 20%
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
        "weight": 0.50,
    },
    "MARGEM_LIQUIDA": {
        "direction": "LOW",
        "role": "SECONDARY",
        "status": "RECURRENT_SIGNAL",
        "weight": 0.30,
    },
    "ROE": {
        "direction": "LOW",
        "role": "SECONDARY",
        "status": "WEAK_RECURRENT_SIGNAL",
        "weight": 0.20,
    },
}


RESEARCH_WEIGHTS = {
    "MARGEM_BRUTA": 0.50,
    "MARGEM_LIQUIDA": 0.30,
    "ROE": 0.20,
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

def prepare_turnaround_base(
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        "ISSUER_ID",
        "FUNDAMENTAL_ENGINE_OK",
        "MARGEM_BRUTA",
        "MARGEM_LIQUIDA",
        "ROE",
    }

    _require_columns(
        fundamentals,
        required,
        "fundamentals",
    )

    assert_no_future_return(fundamentals)

    result = fundamentals.copy()

    if "INVESTABLE" in result.columns:

        result = result.loc[
            result["INVESTABLE"]
            .fillna(False)
            .astype(bool)
        ].copy()

    result = result.loc[
        result["FUNDAMENTAL_ENGINE_OK"]
        .fillna(False)
        .astype(bool)
    ].copy()

    if result.empty:
        raise TurnaroundIntegrityError(
            "FAIL-SAFE: nenhuma empresa elegível para Turnaround."
        )

    if result["ISSUER_ID"].duplicated().any():
        raise TurnaroundIntegrityError(
            "FAIL-SAFE: ISSUER_ID duplicado."
        )

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
# SCORE EXPERIMENTAL 50 / 30 / 20
# =============================================================================

def calculate_experimental_score(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    signal_columns = {
        "MARGEM_BRUTA":
            "MARGEM_BRUTA_DEPRESSED_SIGNAL",

        "MARGEM_LIQUIDA":
            "MARGEM_LIQUIDA_DEPRESSED_SIGNAL",

        "ROE":
            "ROE_DEPRESSED_SIGNAL",
    }

    weighted_sum = pd.Series(
        0.0,
        index=result.index,
        dtype=float,
    )

    available_weight = pd.Series(
        0.0,
        index=result.index,
        dtype=float,
    )

    for factor, weight in RESEARCH_WEIGHTS.items():

        column = signal_columns[factor]

        signal = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        valid = signal.notna()

        weighted_sum.loc[valid] += (
            signal.loc[valid]
            * weight
        )

        available_weight.loc[valid] += weight

    # -------------------------------------------------------------------------
    # NÃO IMPUTAMOS FATOR AUSENTE.
    #
    # O peso é renormalizado somente entre fatores disponíveis.
    # Porém o score só é considerado válido quando todos os três fatores
    # estiverem disponíveis. Assim não alteramos silenciosamente a regra.
    # -------------------------------------------------------------------------

    score = pd.Series(
        np.nan,
        index=result.index,
        dtype=float,
    )

    has_any = available_weight.gt(0)

    score.loc[has_any] = (
        weighted_sum.loc[has_any]
        /
        available_weight.loc[has_any]
    )

    result[
        "TURNAROUND_RESEARCH_WEIGHT_COVERAGE"
    ] = available_weight

    result[
        "TURNAROUND_RESEARCH_SCORE"
    ] = score.clip(
        lower=0.0,
        upper=1.0,
    )

    result[
        "TURNAROUND_RESEARCH_SCORE_VALID"
    ] = (
        result[
            "TURNAROUND_RESEARCH_SCORE"
        ]
        .notna()
        &
        available_weight.ge(0.999999)
    )

    # Score incompleto permanece visível para auditoria,
    # mas não é elegível como score completo.
    result[
        "TURNAROUND_RESEARCH_SCORE_COMPLETE"
    ] = np.where(
        result[
            "TURNAROUND_RESEARCH_SCORE_VALID"
        ],
        result[
            "TURNAROUND_RESEARCH_SCORE"
        ],
        np.nan,
    )

    return result


# =============================================================================
# CLASSIFICAÇÃO EXPERIMENTAL
# =============================================================================

def classify_research_candidate(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    score = pd.to_numeric(
        result[
            "TURNAROUND_RESEARCH_SCORE_COMPLETE"
        ],
        errors="coerce",
    )

    status = pd.Series(
        "NO_SIGNAL",
        index=result.index,
        dtype="object",
    )

    status.loc[
        score.ge(0.50)
    ] = "RESEARCH_CANDIDATE"

    status.loc[
        score.ge(0.80)
    ] = "HIGH_RESEARCH_INTEREST"

    status.loc[
        score.isna()
    ] = "INSUFFICIENT_DATA"

    result[
        "TURNAROUND_RESEARCH_STATUS"
    ] = status

    return result


# =============================================================================
# PRODUÇÃO
# =============================================================================

def enforce_production_lock() -> None:

    production_authorized = bool(
        TURNAROUND_ENGINE.get(
            "production_authorized",
            False,
        )
    )

    validated_rule = TURNAROUND_ENGINE.get(
        "validated_rule"
    )

    if (
        production_authorized
        and validated_rule is None
    ):
        raise TurnaroundProductionError(
            "FAIL-SAFE: Turnaround marcado como produção "
            "sem regra temporalmente validada."
        )


# =============================================================================
# MOTOR PRINCIPAL
# =============================================================================

def run_turnaround_engine(
    fundamentals: pd.DataFrame,
    context: PITContext,
) -> pd.DataFrame:

    if not TURNAROUND_ENGINE.get(
        "enabled",
        True,
    ):
        raise TurnaroundError(
            "Turnaround Engine está desativado."
        )

    enforce_production_lock()

    base = prepare_turnaround_base(
        fundamentals
    )

    result = calculate_research_positions(
        base
    )

    result = calculate_depressed_signals(
        result
    )

    result = classify_gross_margin_zone(
        result
    )

    result = calculate_safety_context(
        result
    )

    result = calculate_experimental_score(
        result
    )

    result = classify_research_candidate(
        result
    )

    production_authorized = bool(
        TURNAROUND_ENGINE.get(
            "production_authorized",
            False,
        )
    )

    result[
        "TURNAROUND_PRODUCTION_AUTHORIZED"
    ] = production_authorized

    if not production_authorized:

        result[
            "TURNAROUND_OPERATIONAL_SIGNAL"
        ] = "BLOCKED_RESEARCH_ONLY"

    else:

        validated_rule = TURNAROUND_ENGINE.get(
            "validated_rule"
        )

        if validated_rule is None:
            raise TurnaroundProductionError(
                "FAIL-SAFE: regra validada ausente."
            )

        if validated_rule != "MB_50_ML_30_ROE_20":
            raise TurnaroundProductionError(
                "FAIL-SAFE: regra operacional divergente do modelo "
                "MB_50_ML_30_ROE_20."
            )

        configured_weights = TURNAROUND_ENGINE.get(
            "study_weights",
            RESEARCH_WEIGHTS,
        )

        if configured_weights != RESEARCH_WEIGHTS:
            raise TurnaroundProductionError(
                "FAIL-SAFE: pesos configurados divergem do modelo "
                "50/30/20 implementado."
            )

        result[
            "TURNAROUND_OPERATIONAL_SIGNAL"
        ] = np.where(
            result[
                "TURNAROUND_RESEARCH_SCORE_VALID"
            ]
            .fillna(False)
            .astype(bool),
            "STUDY_SCORE_AUTHORIZED",
            "INSUFFICIENT_DATA",
        )

    result[
        "TURNAROUND_ENGINE_VERSION"
    ] = "0.3.0"

    result[
        "TURNAROUND_ENGINE_STATUS"
    ] = TURNAROUND_ENGINE.get(
        "status",
        "RESEARCH_CANDIDATE",
    )

    result[
        "TURNAROUND_SCORE_MODEL"
    ] = (
        "MB_50_ML_30_ROE_20_STUDY"
    )

    result[
        "FORMATION_DATE_TURNAROUND"
    ] = context.formation_date

    result[
        "ACCOUNTING_CUTOFF_TURNAROUND"
    ] = context.accounting_cutoff

    result[
        "FUTURE_RETURN_USED_TURNAROUND"
    ] = False

    result[
        "TURNAROUND_RESEARCH_RANK"
    ] = (
        result[
            "TURNAROUND_RESEARCH_SCORE_COMPLETE"
        ]
        .rank(
            ascending=False,
            method="min",
            na_option="bottom",
        )
    )

    result = (
        result
        .sort_values(
            [
                "TURNAROUND_RESEARCH_SCORE_COMPLETE",
                "ISSUER_ID",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    assert_no_future_return(
        result
    )

    return result


# =============================================================================
# AUDITORIA
# =============================================================================

def audit_turnaround(
    result: pd.DataFrame,
) -> dict:

    _require_dataframe(
        result,
        "turnaround_result",
    )

    status_counts = (
        result[
            "TURNAROUND_RESEARCH_STATUS"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    zone_counts = (
        result[
            "GROSS_MARGIN_ZONE"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    valid = (
        result[
            "TURNAROUND_RESEARCH_SCORE_VALID"
        ]
        .fillna(False)
        .astype(bool)
    )

    return {
        "status": "OK",
        "mode": (
            "STUDY_OPERATIONAL"
            if bool(
                TURNAROUND_ENGINE.get(
                    "production_authorized",
                    False,
                )
            )
            else "RESEARCH_ONLY"
        ),
        "score_model":
            "MB_50_ML_30_ROE_20_STUDY",
        "issuers":
            int(len(result)),
        "valid_complete_scores":
            int(valid.sum()),
        "weights": RESEARCH_WEIGHTS,
        "primary_factor":
            "MARGEM_BRUTA",
        "primary_direction":
            "LOW",
        "research_status_distribution":
            status_counts,
        "gross_margin_zone_distribution":
            zone_counts,
        "production_authorized":
            bool(
                TURNAROUND_ENGINE.get(
                    "production_authorized",
                    False,
                )
            ),
        "weights_optimized_on_future_return":
            False,
        "future_return_used":
            False,
        "buy_signal_generated":
            False,
        "operational_score_available":
            bool(
                TURNAROUND_ENGINE.get(
                    "production_authorized",
                    False,
                )
            ),
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

def _self_test():

    sample = pd.DataFrame(
        {
            "ISSUER_ID": [
                "1",
                "2",
                "3",
                "4",
                "5",
            ],

            "FUNDAMENTAL_ENGINE_OK": [
                True,
                True,
                True,
                True,
                True,
            ],

            "INVESTABLE": [
                True,
                True,
                True,
                True,
                True,
            ],

            "MARGEM_BRUTA": [
                0.05,
                0.15,
                0.30,
                0.50,
                0.80,
            ],

            "MARGEM_LIQUIDA": [
                -0.10,
                -0.02,
                0.05,
                0.10,
                0.20,
            ],

            "ROE": [
                -0.10,
                0.01,
                0.08,
                0.15,
                0.25,
            ],

            "PL_POSITIVO": [
                True,
                True,
                True,
                True,
                True,
            ],

            "FCO_POSITIVO": [
                False,
                True,
                True,
                True,
                True,
            ],

            "EBIT_POSITIVO": [
                False,
                True,
                True,
                True,
                True,
            ],
        }
    )

    class FakeContext:
        formation_date = pd.Timestamp(
            "2025-12-31"
        )

        accounting_cutoff = pd.Timestamp(
            "2025-09-30"
        )

    result = run_turnaround_engine(
        sample,
        FakeContext(),
    )

    indexed = result.set_index(
        "ISSUER_ID"
    )

    if not (
        indexed.loc[
            "1",
            "TURNAROUND_RESEARCH_SCORE_COMPLETE",
        ]
        >
        indexed.loc[
            "5",
            "TURNAROUND_RESEARCH_SCORE_COMPLETE",
        ]
    ):
        raise TurnaroundError(
            "SELF-TEST: direção do score 50/30/20 incorreta."
        )

    # Verifica matematicamente a ponderação.
    expected_first = (
        0.50
        * indexed.loc[
            "1",
            "MARGEM_BRUTA_DEPRESSED_SIGNAL",
        ]
        +
        0.30
        * indexed.loc[
            "1",
            "MARGEM_LIQUIDA_DEPRESSED_SIGNAL",
        ]
        +
        0.20
        * indexed.loc[
            "1",
            "ROE_DEPRESSED_SIGNAL",
        ]
    )

    actual_first = indexed.loc[
        "1",
        "TURNAROUND_RESEARCH_SCORE_COMPLETE",
    ]

    if not np.isclose(
        expected_first,
        actual_first,
    ):
        raise TurnaroundError(
            "SELF-TEST: pesos 50/30/20 incorretos."
        )

    production_authorized = bool(
        TURNAROUND_ENGINE.get(
            "production_authorized",
            False,
        )
    )

    if production_authorized:
        valid_mask = (
            result[
                "TURNAROUND_RESEARCH_SCORE_VALID"
            ]
            .fillna(False)
            .astype(bool)
        )

        if not (
            result.loc[
                valid_mask,
                "TURNAROUND_OPERATIONAL_SIGNAL",
            ]
            == "STUDY_SCORE_AUTHORIZED"
        ).all():
            raise TurnaroundError(
                "SELF-TEST: score 50/30/20 autorizado não foi "
                "liberado corretamente."
            )

        if (
            result.loc[
                ~valid_mask,
                "TURNAROUND_OPERATIONAL_SIGNAL",
            ]
            == "STUDY_SCORE_AUTHORIZED"
        ).any():
            raise TurnaroundError(
                "SELF-TEST: score incompleto foi liberado."
            )

    else:
        if not (
            result[
                "TURNAROUND_OPERATIONAL_SIGNAL"
            ]
            == "BLOCKED_RESEARCH_ONLY"
        ).all():
            raise TurnaroundError(
                "SELF-TEST: sinal operacional liberado sem autorização."
            )

    if (
        result[
            "FUTURE_RETURN_USED_TURNAROUND"
        ]
        .any()
    ):
        raise TurnaroundError(
            "SELF-TEST: retorno futuro utilizado."
        )

    contaminated = sample.copy()

    contaminated[
        "FUTURE_RETURN"
    ] = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]

    future_blocked = False

    try:

        run_turnaround_engine(
            contaminated,
            FakeContext(),
        )

    except TurnaroundIntegrityError:

        future_blocked = True

    if not future_blocked:
        raise TurnaroundError(
            "SELF-TEST: retorno futuro real não foi bloqueado."
        )

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
    print("Score do estudo:")
    print("  MARGEM_BRUTA   = 50%")
    print("  MARGEM_LIQUIDA = 30%")
    print("  ROE            = 20%")
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
