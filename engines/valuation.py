# =============================================================================
# B3 STOCK SELECTOR
# Arquivo: engines/valuation.py
#
# VALUATION ENGINE
#
# OBJETIVO:
# Medir valuation de forma independente das camadas Quality e Turnaround.
#
# RESPONSABILIDADE:
# - Trabalhar somente com dados disponíveis na data de formação
# - Calcular múltiplos quando os dados necessários existirem
# - Transformar múltiplos válidos em percentis cross-sectional
# - Separar múltiplos economicamente válidos de múltiplos sem interpretação
# - Registrar cobertura
#
# NÃO FAZ:
# - Descoberta de fatores
# - Otimização por retorno futuro
# - Quality Score
# - Turnaround Score
# - Ranking final
# - Recomendação de compra
#
# PRINCÍPIO:
# "Barato" só é considerado quando o múltiplo possui interpretação econômica
# válida. Exemplo: P/L negativo NÃO significa ação extremamente barata.
# =============================================================================

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

try:
    from config import (
        OUTPUT_DIR,
        VALUATION_ENGINE,
    )

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


# =============================================================================
# DIRETÓRIO
# =============================================================================

VALUATION_DIR = (
    OUTPUT_DIR
    / "valuation"
)

VALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# EXCEÇÕES
# =============================================================================

class ValuationError(RuntimeError):
    """Erro geral do Valuation Engine."""


class ValuationIntegrityError(ValuationError):
    """Erro estrutural ou de integridade."""


# =============================================================================
# MÚLTIPLOS
#
# LOW = menor múltiplo válido recebe melhor percentil de valuation.
#
# Nenhum peso foi otimizado com retorno futuro.
# =============================================================================

VALUATION_FACTORS = {
    "PL_MULTIPLE": "LOW",
    "PVP_MULTIPLE": "LOW",
    "PS_MULTIPLE": "LOW",
    "EV_EBIT": "LOW",
    "EV_FCO": "LOW",
}


# =============================================================================
# UTILIDADES
# =============================================================================

def _utc_now_iso() -> str:

    return datetime.now(
        timezone.utc
    ).isoformat()


def _require_dataframe(
    df: pd.DataFrame,
    name: str,
) -> None:

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise ValuationIntegrityError(
            f"{name} não é DataFrame."
        )

    if df.empty:
        raise ValuationIntegrityError(
            f"FAIL-SAFE: {name} está vazio."
        )


def _require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:

    _require_dataframe(
        df,
        name,
    )

    missing = (
        set(columns)
        - set(df.columns)
    )

    if missing:

        raise ValuationIntegrityError(
            f"FAIL-SAFE: {name} sem colunas obrigatórias: "
            f"{sorted(missing)}"
        )


def _numeric(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:

    if column not in df.columns:

        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).astype(float)


# =============================================================================
# PROTEÇÃO CONTRA RETORNO FUTURO
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

    # -------------------------------------------------------------------------
    # COLUNAS DE AUDITORIA PERMITIDAS
    #
    # Estas colunas NÃO contêm retorno futuro.
    # Elas apenas registram explicitamente que retorno futuro não foi usado.
    # -------------------------------------------------------------------------

    allowed = {
        "FUTURE_RETURN_USED",
        "FUTURE_RETURN_USED_QUALITY",
        "FUTURE_RETURN_USED_TURNAROUND",
        "FUTURE_RETURN_USED_VALUATION",
    }

    forbidden = []

    for column in df.columns:

        normalized = (
            str(column)
            .upper()
        )

        if normalized in allowed:

            values = (
                df[column]
                .dropna()
            )

            if not values.empty:

                normalized_values = (
                    values
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )

                invalid_values = (
                    ~normalized_values.isin(
                        {
                            "FALSE",
                            "0",
                        }
                    )
                )

                if invalid_values.any():

                    raise ValuationIntegrityError(
                        "FAIL-SAFE: coluna de auditoria "
                        f"{column} indica possível uso de retorno futuro."
                    )

            continue

        if any(
            term in normalized
            for term in forbidden_terms
        ):

            forbidden.append(
                column
            )

    if forbidden:

        raise ValuationIntegrityError(
            "FAIL-SAFE: variável de retorno futuro "
            "detectada no Valuation Engine: "
            f"{forbidden}"
        )


# =============================================================================
# DIVISÃO SEGURA
# =============================================================================

def safe_positive_multiple(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Calcula múltiplo somente quando:
        numerador > 0
        denominador > 0

    Isso evita interpretar denominadores negativos como "barato".
    """

    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    result = pd.Series(
        np.nan,
        index=numerator.index,
        dtype=float,
    )

    valid = (
        numerator.notna()
        &
        denominator.notna()
        &
        numerator.gt(0)
        &
        denominator.gt(0)
    )

    result.loc[
        valid
    ] = (
        numerator.loc[
            valid
        ]
        /
        denominator.loc[
            valid
        ]
    )

    return result.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


# =============================================================================
# PREPARAÇÃO
# =============================================================================

def prepare_valuation_base(
    df: pd.DataFrame,
) -> pd.DataFrame:

    _require_columns(
        df,
        [
            "ISSUER_ID",
            "FUNDAMENTAL_ENGINE_OK",
        ],
        "valuation_base",
    )

    assert_no_future_return(
        df
    )

    result = (
        df
        .copy()
    )

    if "INVESTABLE" in result.columns:

        result = result.loc[
            result[
                "INVESTABLE"
            ]
            .fillna(False)
            .astype(bool)
        ].copy()

    result = result.loc[
        result[
            "FUNDAMENTAL_ENGINE_OK"
        ]
        .fillna(False)
        .astype(bool)
    ].copy()

    if result.empty:

        raise ValuationIntegrityError(
            "FAIL-SAFE: nenhuma empresa elegível "
            "para Valuation."
        )

    if (
        result[
            "ISSUER_ID"
        ]
        .duplicated()
        .any()
    ):

        raise ValuationIntegrityError(
            "FAIL-SAFE: ISSUER_ID duplicado."
        )

    return result


# =============================================================================
# VALOR DE MERCADO
# =============================================================================

def calculate_market_value(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Prioridade:

    1. MARKET_CAP já calculado corretamente pelo pipeline;
    2. FORMATION_PRICE * SHARES_OUTSTANDING.

    Não usa preço posterior à formação.
    """

    market_cap = _numeric(
        df,
        "MARKET_CAP",
    )

    valid_market_cap = (
        market_cap.notna()
        &
        market_cap.gt(0)
    )

    result = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    result.loc[
        valid_market_cap
    ] = market_cap.loc[
        valid_market_cap
    ]

    if (
        "FORMATION_PRICE"
        in df.columns
        and
        "SHARES_OUTSTANDING"
        in df.columns
    ):

        price = _numeric(
            df,
            "FORMATION_PRICE",
        )

        shares = _numeric(
            df,
            "SHARES_OUTSTANDING",
        )

        calculated = (
            price
            *
            shares
        )

        valid_calculated = (
            result.isna()
            &
            price.gt(0)
            &
            shares.gt(0)
        )

        result.loc[
            valid_calculated
        ] = calculated.loc[
            valid_calculated
        ]

    return result


# =============================================================================
# ENTERPRISE VALUE
# =============================================================================

def calculate_enterprise_value(
    df: pd.DataFrame,
    market_value: pd.Series,
) -> pd.Series:
    """
    EV = Market Cap + Dívida Bruta - Caixa
    """

    debt = _numeric(
        df,
        "DIVIDA_BRUTA",
    )

    cash = _numeric(
        df,
        "CAIXA",
    )

    result = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    valid = (
        market_value.notna()
        &
        debt.notna()
        &
        cash.notna()
    )

    result.loc[
        valid
    ] = (
        market_value.loc[
            valid
        ]
        +
        debt.loc[
            valid
        ]
        -
        cash.loc[
            valid
        ]
    )

    result.loc[
        result.le(0)
    ] = np.nan

    return result


# =============================================================================
# CÁLCULO DOS MÚLTIPLOS
# =============================================================================

def calculate_multiples(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        df
        .copy()
    )

    market_value = (
        calculate_market_value(
            result
        )
    )

    result[
        "MARKET_VALUE"
    ] = market_value

    enterprise_value = (
        calculate_enterprise_value(
            result,
            market_value,
        )
    )

    result[
        "ENTERPRISE_VALUE"
    ] = enterprise_value

    profit = _numeric(
        result,
        "LUCRO_LIQUIDO",
    )

    equity = _numeric(
        result,
        "PL",
    )

    revenue = _numeric(
        result,
        "RECEITA",
    )

    ebit = _numeric(
        result,
        "EBIT",
    )

    fco = _numeric(
        result,
        "FCO",
    )

    # -------------------------------------------------------------------------
    # P/L
    # -------------------------------------------------------------------------

    result[
        "PL_MULTIPLE"
    ] = safe_positive_multiple(
        market_value,
        profit,
    )

    # -------------------------------------------------------------------------
    # P/VP
    # -------------------------------------------------------------------------

    result[
        "PVP_MULTIPLE"
    ] = safe_positive_multiple(
        market_value,
        equity,
    )

    # -------------------------------------------------------------------------
    # P/S
    # -------------------------------------------------------------------------

    result[
        "PS_MULTIPLE"
    ] = safe_positive_multiple(
        market_value,
        revenue,
    )

    # -------------------------------------------------------------------------
    # EV / EBIT
    # -------------------------------------------------------------------------

    result[
        "EV_EBIT"
    ] = safe_positive_multiple(
        enterprise_value,
        ebit,
    )

    # -------------------------------------------------------------------------
    # EV / FCO
    # -------------------------------------------------------------------------

    result[
        "EV_FCO"
    ] = safe_positive_multiple(
        enterprise_value,
        fco,
    )

    return result


# =============================================================================
# PERCENTIL
# =============================================================================

def low_is_better_percentile(
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

    valid = (
        numeric.notna()
        &
        numeric.gt(0)
    )

    n = int(
        valid.sum()
    )

    if n == 0:
        return result

    rank = (
        numeric.loc[
            valid
        ]
        .rank(
            pct=True,
            method="average",
        )
    )

    result.loc[
        valid
    ] = (
        1.0
        -
        rank
        +
        (
            1.0 / n
        )
    )

    return result.clip(
        lower=0.0,
        upper=1.0,
    )


# =============================================================================
# SCORES INDIVIDUAIS
# =============================================================================

def calculate_valuation_factor_scores(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        df
        .copy()
    )

    score_columns = []

    for factor in VALUATION_FACTORS:

        score_column = (
            f"VALUATION_FACTOR_{factor}"
        )

        result[
            score_column
        ] = low_is_better_percentile(
            result[
                factor
            ]
        )

        score_columns.append(
            score_column
        )

    result[
        "VALUATION_FACTORS_AVAILABLE"
    ] = (
        result[
            score_columns
        ]
        .notna()
        .sum(
            axis=1
        )
        .astype(int)
    )

    result[
        "VALUATION_FACTORS_TOTAL"
    ] = len(
        score_columns
    )

    result[
        "VALUATION_DATA_COVERAGE"
    ] = (
        result[
            "VALUATION_FACTORS_AVAILABLE"
        ]
        /
        result[
            "VALUATION_FACTORS_TOTAL"
        ]
    )

    return result


# =============================================================================
# SCORE DE VALUATION
# =============================================================================

def calculate_valuation_score(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pesos iguais.

    Não há otimização baseada em retorno futuro.

    Exigimos ao menos dois múltiplos válidos para gerar um score utilizável.
    """

    result = (
        df
        .copy()
    )

    score_columns = [
        f"VALUATION_FACTOR_{factor}"
        for factor in VALUATION_FACTORS
    ]

    result[
        "VALUATION_SCORE"
    ] = (
        result[
            score_columns
        ]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    insufficient = (
        result[
            "VALUATION_FACTORS_AVAILABLE"
        ]
        < 2
    )

    result.loc[
        insufficient,
        "VALUATION_SCORE",
    ] = np.nan

    result[
        "VALUATION_SCORE_VALID"
    ] = (
        result[
            "VALUATION_SCORE"
        ]
        .notna()
    )

    return result


# =============================================================================
# MOTOR PRINCIPAL
# =============================================================================

def run_valuation_engine(
    fundamentals: pd.DataFrame,
    context: PITContext,
) -> pd.DataFrame:

    if not VALUATION_ENGINE.get(
        "enabled",
        True,
    ):

        raise ValuationError(
            "Valuation Engine está desativado."
        )

    base = prepare_valuation_base(
        fundamentals
    )

    result = calculate_multiples(
        base
    )

    result = (
        calculate_valuation_factor_scores(
            result
        )
    )

    result = calculate_valuation_score(
        result
    )

    # -------------------------------------------------------------------------
    # METADADOS
    # -------------------------------------------------------------------------

    result[
        "VALUATION_ENGINE_VERSION"
    ] = "0.1.1"

    result[
        "FORMATION_DATE_VALUATION"
    ] = context.formation_date

    result[
        "ACCOUNTING_CUTOFF_VALUATION"
    ] = context.accounting_cutoff

    result[
        "FUTURE_RETURN_USED_VALUATION"
    ] = False

    # -------------------------------------------------------------------------
    # RANK INTERNO
    #
    # Não é ranking final de compra.
    # -------------------------------------------------------------------------

    result[
        "VALUATION_RANK"
    ] = (
        result[
            "VALUATION_SCORE"
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
                "VALUATION_SCORE",
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

def audit_valuation(
    result: pd.DataFrame,
) -> dict:

    _require_dataframe(
        result,
        "valuation_result",
    )

    valid = (
        result[
            "VALUATION_SCORE_VALID"
        ]
        .fillna(False)
        .astype(bool)
    )

    coverage = {}

    for factor in VALUATION_FACTORS:

        if factor not in result.columns:
            continue

        coverage[
            factor
        ] = float(
            result[
                factor
            ]
            .notna()
            .mean()
        )

    return {
        "status":
            "OK",

        "issuers":
            int(
                len(
                    result
                )
            ),

        "valid_scores":
            int(
                valid.sum()
            ),

        "valid_score_rate":
            float(
                valid.mean()
            ),

        "mean_data_coverage":
            float(
                result[
                    "VALUATION_DATA_COVERAGE"
                ]
                .mean()
            ),

        "multiple_coverage":
            coverage,

        "weights_optimized_on_future_return":
            False,

        "future_return_used":
            False,

        "ranking_final_generated":
            False,
    }


# =============================================================================
# SALVAR
# =============================================================================

def save_valuation(
    result: pd.DataFrame,
    context: PITContext,
) -> dict:

    audit = audit_valuation(
        result
    )

    date_tag = (
        context
        .formation_date
        .strftime(
            "%Y%m%d"
        )
    )

    result_path = (
        VALUATION_DIR
        / f"valuation_{date_tag}.csv"
    )

    manifest_path = (
        VALUATION_DIR
        / f"valuation_manifest_{date_tag}.json"
    )

    result.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "engine":
            "VALUATION_ENGINE",

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

        "valuation_factors":
            VALUATION_FACTORS,

        **audit,

        "files": {
            "valuation":
                str(
                    result_path
                ),
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
        "valuation":
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
            ],

            "FUNDAMENTAL_ENGINE_OK": [
                True,
                True,
                True,
            ],

            "INVESTABLE": [
                True,
                True,
                True,
            ],

            "MARKET_CAP": [
                1000.0,
                2000.0,
                3000.0,
            ],

            "LUCRO_LIQUIDO": [
                200.0,
                200.0,
                -100.0,
            ],

            "PL": [
                800.0,
                800.0,
                1000.0,
            ],

            "RECEITA": [
                1500.0,
                1500.0,
                1800.0,
            ],

            "EBIT": [
                250.0,
                250.0,
                -50.0,
            ],

            "FCO": [
                220.0,
                220.0,
                -20.0,
            ],

            "DIVIDA_BRUTA": [
                100.0,
                100.0,
                500.0,
            ],

            "CAIXA": [
                50.0,
                50.0,
                100.0,
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

    result = run_valuation_engine(
        sample,
        FakeContext(),
    )

    indexed = (
        result
        .set_index(
            "ISSUER_ID"
        )
    )

    # Empresa 1 possui mesmos fundamentos da 2,
    # porém menor preço/market cap.
    if not (
        indexed.loc[
            "1",
            "VALUATION_SCORE",
        ]
        >
        indexed.loc[
            "2",
            "VALUATION_SCORE",
        ]
    ):

        raise ValuationError(
            "SELF-TEST: empresa mais barata "
            "não recebeu melhor valuation."
        )

    # Prejuízo não pode gerar P/L artificialmente barato.
    if not pd.isna(
        indexed.loc[
            "3",
            "PL_MULTIPLE",
        ]
    ):

        raise ValuationError(
            "SELF-TEST: P/L negativo foi "
            "interpretado como múltiplo válido."
        )

    if (
        result[
            "FUTURE_RETURN_USED_VALUATION"
        ]
        .any()
    ):

        raise ValuationError(
            "SELF-TEST: retorno futuro utilizado."
        )

    # -------------------------------------------------------------------------
    # TESTE FAIL-SAFE:
    # retorno futuro REAL deve continuar bloqueado.
    # -------------------------------------------------------------------------

    contaminated = sample.copy()

    contaminated[
        "FUTURE_RETURN"
    ] = [
        1.0,
        2.0,
        3.0,
    ]

    future_blocked = False

    try:

        run_valuation_engine(
            contaminated,
            FakeContext(),
        )

    except ValuationIntegrityError:

        future_blocked = True

    if not future_blocked:

        raise ValuationError(
            "SELF-TEST: retorno futuro real não foi bloqueado."
        )

    return True


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("B3 STOCK SELECTOR — VALUATION ENGINE")
    print("=" * 72)

    _self_test()

    print("Self-test: OK")
    print("P/L: CALCULADO")
    print("P/VP: CALCULADO")
    print("P/S: CALCULADO")
    print("EV/EBIT: CALCULADO")
    print("EV/FCO: CALCULADO")
    print("Múltiplos negativos como barato: BLOQUEADO")
    print("Pesos otimizados em retorno futuro: NÃO")
    print("Quality: INDEPENDENTE")
    print("Turnaround: INDEPENDENTE")
    print("Retorno futuro: NÃO UTILIZADO")
    print("Fail-safe de retorno futuro: OK")
    print("Ranking final: NÃO EXECUTADO")
    print("=" * 72)
