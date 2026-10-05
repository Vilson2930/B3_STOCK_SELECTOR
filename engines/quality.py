# =============================================================================
# B3 STOCK SELECTOR
# Arquivo: engines/quality.py
#
# QUALITY ENGINE
#
# OBJETIVO:
# Medir qualidade econômica/financeira das empresas que já passaram pelo
# Investability Gate.
#
# IMPORTANTE:
# - Quality NÃO é Turnaround.
# - Este motor NÃO utiliza a descoberta de "margem baixa" como vantagem.
# - Não utiliza retorno futuro.
# - Não otimiza pesos olhando desempenho futuro.
# - Produz uma camada independente que posteriormente poderá ser combinada
#   com Valuation e/ou Turnaround.
#
# FILOSOFIA:
# Empresas de qualidade apresentam, em termos gerais:
# - rentabilidade saudável;
# - operação lucrativa;
# - geração de caixa;
# - estrutura financeira sustentável;
# - eficiência operacional.
#
# O motor utiliza percentis cross-sectional para reduzir dependência
# de escalas absolutas.
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
        QUALITY_ENGINE,
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

QUALITY_DIR = (
    OUTPUT_DIR
    / "quality"
)

QUALITY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# EXCEÇÕES
# =============================================================================

class QualityError(RuntimeError):
    """Erro geral do Quality Engine."""


class QualityIntegrityError(QualityError):
    """Erro estrutural."""


# =============================================================================
# FATORES
#
# Estes fatores representam qualidade tradicional.
#
# HIGH:
#     maior valor econômico tende a representar maior qualidade.
#
# LOW:
#     menor valor tende a representar melhor estrutura financeira.
#
# ATENÇÃO:
# Estes fatores NÃO representam a camada Turnaround.
# =============================================================================

QUALITY_FACTORS = {

    "ROA": "HIGH",

    "GIRO_ATIVO": "HIGH",

    "FCO_ATIVO": "HIGH",

    "FCO_RECEITA": "HIGH",

    "DIVIDA_BRUTA_ATIVO": "LOW",

    "DIVIDA_LIQUIDA_ATIVO": "LOW",

    "CAPITAL_GIRO_ATIVO": "HIGH",
}


# =============================================================================
# FLAGS DE QUALIDADE
# =============================================================================

QUALITY_FLAGS = (
    "LUCRO_POSITIVO",
    "EBIT_POSITIVO",
    "FCO_POSITIVO",
    "CAPITAL_GIRO_POSITIVO",
    "PL_POSITIVO",
)


# =============================================================================
# PESOS
#
# Não são pesos "otimizados" em retorno futuro.
#
# Mantemos inicialmente pesos iguais para evitar introduzir uma falsa
# precisão antes de validação científica específica.
# =============================================================================

FACTOR_WEIGHT = 0.80
FLAG_WEIGHT = 0.20


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

        raise QualityIntegrityError(
            f"{name} não é DataFrame."
        )

    if df.empty:

        raise QualityIntegrityError(
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

        raise QualityIntegrityError(
            f"FAIL-SAFE: {name} sem colunas: "
            f"{sorted(missing)}"
        )


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
        "FUTURE_WINNER",
        "WINNER_LABEL",
    )

    # -------------------------------------------------------------------------
    # COLUNAS DE AUDITORIA PERMITIDAS
    #
    # Elas NÃO contêm retorno futuro.
    # Apenas registram explicitamente que retorno futuro NÃO foi utilizado.
    # -------------------------------------------------------------------------

    allowed_audit_columns = {
        "FUTURE_RETURN_USED",
        "FUTURE_RETURN_USED_QUALITY",
    }

    forbidden = []

    for column in df.columns:

        normalized = (
            str(column)
            .upper()
        )

        if normalized in allowed_audit_columns:

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

                    raise QualityIntegrityError(
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

        raise QualityIntegrityError(
            "FAIL-SAFE: variável relacionada a "
            "retorno futuro detectada no Quality Engine: "
            f"{forbidden}"
        )


# =============================================================================
# PREPARAÇÃO
# =============================================================================

def prepare_quality_base(
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        "ISSUER_ID",
        "FUNDAMENTAL_ENGINE_OK",
    }

    _require_columns(
        fundamentals,
        required,
        "fundamentals",
    )

    assert_no_future_return(
        fundamentals
    )

    result = (
        fundamentals
        .copy()
    )

    # -------------------------------------------------------------------------
    # SE INVESTABILITY ESTIVER PRESENTE, SOMENTE PASSADOS AVANÇAM
    # -------------------------------------------------------------------------

    if "INVESTABLE" in result.columns:

        result = result.loc[
            result[
                "INVESTABLE"
            ]
            .fillna(False)
            .astype(bool)
        ].copy()

    # -------------------------------------------------------------------------
    # FUNDAMENTAL ENGINE
    # -------------------------------------------------------------------------

    result = result.loc[
        result[
            "FUNDAMENTAL_ENGINE_OK"
        ]
        .fillna(False)
        .astype(bool)
    ].copy()

    if result.empty:

        raise QualityIntegrityError(
            "FAIL-SAFE: nenhuma empresa elegível "
            "para o Quality Engine."
        )

    if (
        result[
            "ISSUER_ID"
        ]
        .duplicated()
        .any()
    ):

        raise QualityIntegrityError(
            "FAIL-SAFE: ISSUER_ID duplicado."
        )

    return result


# =============================================================================
# PERCENTIL
# =============================================================================

def percentile_score(
    series: pd.Series,
    direction: str,
) -> pd.Series:
    """
    Converte fator em percentil [0,1].

    HIGH:
        maior valor = maior score.

    LOW:
        menor valor = maior score.

    Missing:
        permanece NaN.

    Nenhuma imputação é realizada neste estágio.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    score = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    valid = numeric.notna()

    if valid.sum() == 0:
        return score

    ranked = (
        numeric.loc[
            valid
        ]
        .rank(
            pct=True,
            method="average",
        )
    )

    if direction == "HIGH":

        score.loc[
            valid
        ] = ranked

    elif direction == "LOW":

        score.loc[
            valid
        ] = (
            1.0
            -
            ranked
            +
            (
                1.0
                /
                valid.sum()
            )
        )

    else:

        raise QualityIntegrityError(
            f"Direção inválida: {direction}"
        )

    return score.clip(
        lower=0.0,
        upper=1.0,
    )


# =============================================================================
# SCORE DOS FATORES
# =============================================================================

def calculate_factor_scores(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        df
        .copy()
    )

    score_columns = []

    for (
        factor,
        direction,
    ) in QUALITY_FACTORS.items():

        score_column = (
            f"QUALITY_FACTOR_{factor}"
        )

        if factor not in result.columns:

            result[
                score_column
            ] = np.nan

        else:

            result[
                score_column
            ] = percentile_score(
                result[
                    factor
                ],
                direction,
            )

        score_columns.append(
            score_column
        )

    result[
        "QUALITY_FACTOR_AVAILABLE"
    ] = (
        result[
            score_columns
        ]
        .notna()
        .sum(
            axis=1
        )
    )

    result[
        "QUALITY_FACTOR_REQUIRED"
    ] = len(
        score_columns
    )

    result[
        "QUALITY_FACTOR_COVERAGE"
    ] = (
        result[
            "QUALITY_FACTOR_AVAILABLE"
        ]
        /
        result[
            "QUALITY_FACTOR_REQUIRED"
        ]
    )

    result[
        "QUALITY_CONTINUOUS_SCORE"
    ] = (
        result[
            score_columns
        ]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    return result


# =============================================================================
# SCORE DOS FLAGS
# =============================================================================

def _boolean_to_float(
    series: pd.Series,
) -> pd.Series:

    output = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    if pd.api.types.is_bool_dtype(
        series
    ):

        valid = series.notna()

        output.loc[
            valid
        ] = (
            series.loc[
                valid
            ]
            .astype(float)
        )

        return output

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )

    true_values = {
        "TRUE",
        "1",
        "YES",
        "SIM",
    }

    false_values = {
        "FALSE",
        "0",
        "NO",
        "NAO",
        "NÃO",
    }

    true_mask = (
        normalized.isin(
            true_values
        )
    )

    false_mask = (
        normalized.isin(
            false_values
        )
    )

    output.loc[
        true_mask
    ] = 1.0

    output.loc[
        false_mask
    ] = 0.0

    return output


def calculate_flag_score(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        df
        .copy()
    )

    flag_columns = []

    for flag in QUALITY_FLAGS:

        score_column = (
            f"QUALITY_FLAG_{flag}"
        )

        if flag not in result.columns:

            result[
                score_column
            ] = np.nan

        else:

            result[
                score_column
            ] = _boolean_to_float(
                result[
                    flag
                ]
            )

        flag_columns.append(
            score_column
        )

    result[
        "QUALITY_FLAGS_AVAILABLE"
    ] = (
        result[
            flag_columns
        ]
        .notna()
        .sum(
            axis=1
        )
    )

    result[
        "QUALITY_FLAGS_REQUIRED"
    ] = len(
        flag_columns
    )

    result[
        "QUALITY_FLAG_COVERAGE"
    ] = (
        result[
            "QUALITY_FLAGS_AVAILABLE"
        ]
        /
        result[
            "QUALITY_FLAGS_REQUIRED"
        ]
    )

    result[
        "QUALITY_FLAG_SCORE"
    ] = (
        result[
            flag_columns
        ]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    return result


# =============================================================================
# SCORE FINAL QUALITY
# =============================================================================

def calculate_quality_score(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        df
        .copy()
    )

    continuous = pd.to_numeric(
        result[
            "QUALITY_CONTINUOUS_SCORE"
        ],
        errors="coerce",
    )

    flags = pd.to_numeric(
        result[
            "QUALITY_FLAG_SCORE"
        ],
        errors="coerce",
    )

    valid = (
        continuous.notna()
        &
        flags.notna()
    )

    result[
        "QUALITY_SCORE"
    ] = np.nan

    result.loc[
        valid,
        "QUALITY_SCORE",
    ] = (
        FACTOR_WEIGHT
        *
        continuous.loc[
            valid
        ]
        +
        FLAG_WEIGHT
        *
        flags.loc[
            valid
        ]
    )

    result[
        "QUALITY_SCORE"
    ] = (
        result[
            "QUALITY_SCORE"
        ]
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    result[
        "QUALITY_DATA_COVERAGE"
    ] = (
        (
            result[
                "QUALITY_FACTOR_COVERAGE"
            ]
            +
            result[
                "QUALITY_FLAG_COVERAGE"
            ]
        )
        / 2.0
    )

    result[
        "QUALITY_SCORE_VALID"
    ] = (
        result[
            "QUALITY_SCORE"
        ]
        .notna()
        &
        (
            result[
                "QUALITY_DATA_COVERAGE"
            ]
            >= 0.50
        )
    )

    return result


# =============================================================================
# MOTOR PRINCIPAL
# =============================================================================

def run_quality_engine(
    fundamentals: pd.DataFrame,
    context: PITContext,
) -> pd.DataFrame:

    if not QUALITY_ENGINE.get(
        "enabled",
        True,
    ):

        raise QualityError(
            "Quality Engine está desativado."
        )

    base = prepare_quality_base(
        fundamentals
    )

    result = calculate_factor_scores(
        base
    )

    result = calculate_flag_score(
        result
    )

    result = calculate_quality_score(
        result
    )

    result[
        "QUALITY_ENGINE_VERSION"
    ] = "0.1.1"

    result[
        "QUALITY_ENGINE_STATUS"
    ] = QUALITY_ENGINE.get(
        "status",
        "REFERENCE_ENGINE",
    )

    result[
        "FORMATION_DATE_QUALITY"
    ] = context.formation_date

    result[
        "ACCOUNTING_CUTOFF_QUALITY"
    ] = context.accounting_cutoff

    result[
        "FUTURE_RETURN_USED_QUALITY"
    ] = False

    result[
        "QUALITY_RANK"
    ] = (
        result[
            "QUALITY_SCORE"
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
                "QUALITY_SCORE",
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

def audit_quality(
    result: pd.DataFrame,
) -> dict:

    _require_dataframe(
        result,
        "quality_result",
    )

    valid_scores = (
        result[
            "QUALITY_SCORE"
        ]
        .notna()
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
                valid_scores.sum()
            ),

        "mean_score":
            (
                float(
                    result.loc[
                        valid_scores,
                        "QUALITY_SCORE",
                    ]
                    .mean()
                )
                if valid_scores.any()
                else None
            ),

        "median_score":
            (
                float(
                    result.loc[
                        valid_scores,
                        "QUALITY_SCORE",
                    ]
                    .median()
                )
                if valid_scores.any()
                else None
            ),

        "mean_data_coverage":
            float(
                result[
                    "QUALITY_DATA_COVERAGE"
                ]
                .mean()
            ),

        "factor_weight":
            FACTOR_WEIGHT,

        "flag_weight":
            FLAG_WEIGHT,

        "future_return_used":
            False,

        "turnaround_signal_used":
            False,
    }


# =============================================================================
# SALVAR
# =============================================================================

def save_quality(
    result: pd.DataFrame,
    context: PITContext,
) -> dict:

    audit = audit_quality(
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
        QUALITY_DIR
        / f"quality_{date_tag}.csv"
    )

    manifest_path = (
        QUALITY_DIR
        / f"quality_manifest_{date_tag}.json"
    )

    result.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "engine":
            "QUALITY_ENGINE",

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

        "factors":
            QUALITY_FACTORS,

        "flags":
            list(
                QUALITY_FLAGS
            ),

        **audit,

        "files": {
            "quality":
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
        "quality":
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

            "ROA": [
                0.15,
                0.05,
                -0.05,
            ],

            "GIRO_ATIVO": [
                1.2,
                0.8,
                0.4,
            ],

            "FCO_ATIVO": [
                0.18,
                0.07,
                -0.02,
            ],

            "FCO_RECEITA": [
                0.15,
                0.08,
                -0.04,
            ],

            "DIVIDA_BRUTA_ATIVO": [
                0.10,
                0.30,
                0.70,
            ],

            "DIVIDA_LIQUIDA_ATIVO": [
                0.05,
                0.20,
                0.60,
            ],

            "CAPITAL_GIRO_ATIVO": [
                0.30,
                0.10,
                -0.20,
            ],

            "LUCRO_POSITIVO": [
                True,
                True,
                False,
            ],

            "EBIT_POSITIVO": [
                True,
                True,
                False,
            ],

            "FCO_POSITIVO": [
                True,
                True,
                False,
            ],

            "CAPITAL_GIRO_POSITIVO": [
                True,
                True,
                False,
            ],

            "PL_POSITIVO": [
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

    result = run_quality_engine(
        sample,
        FakeContext(),
    )

    scores = (
        result
        .set_index(
            "ISSUER_ID"
        )[
            "QUALITY_SCORE"
        ]
    )

    if not (
        scores["1"]
        >
        scores["2"]
        >
        scores["3"]
    ):

        raise QualityError(
            "SELF-TEST: ordenação de qualidade incorreta."
        )

    if (
        result[
            "QUALITY_SCORE"
        ]
        .min()
        < 0
    ):

        raise QualityError(
            "SELF-TEST: score abaixo de zero."
        )

    if (
        result[
            "QUALITY_SCORE"
        ]
        .max()
        > 1
    ):

        raise QualityError(
            "SELF-TEST: score acima de 1."
        )

    # -------------------------------------------------------------------------
    # TESTE DO FAIL-SAFE:
    # coluna de auditoria False deve ser permitida.
    # -------------------------------------------------------------------------

    if not (
        result[
            "FUTURE_RETURN_USED_QUALITY"
        ]
        .eq(False)
        .all()
    ):

        raise QualityError(
            "SELF-TEST: auditoria de retorno futuro inválida."
        )

    # -------------------------------------------------------------------------
    # TESTE DO FAIL-SAFE:
    # variável futura real deve continuar proibida.
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

        run_quality_engine(
            contaminated,
            FakeContext(),
        )

    except QualityIntegrityError:

        future_blocked = True

    if not future_blocked:

        raise QualityError(
            "SELF-TEST: retorno futuro real não foi bloqueado."
        )

    return True


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("B3 STOCK SELECTOR — QUALITY ENGINE")
    print("=" * 72)

    _self_test()

    print("Self-test: OK")
    print("Quality e Turnaround: SEPARADOS")
    print("Pesos ajustados por retorno futuro: NÃO")
    print("Retorno futuro: NÃO UTILIZADO")
    print("Fail-safe de retorno futuro: OK")
    print("Margem bruta baixa como vantagem: NÃO")
    print("Score final de compra: NÃO")
    print("Quality Score: 0–1")
    print("Ranking interno: SOMENTE AUDITORIA")

    print("=" * 72)
