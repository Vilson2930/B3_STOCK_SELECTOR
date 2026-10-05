# =============================================================================
# B3 STOCK SELECTOR
# Arquivo: engines/ranking.py
#
# RANKING ENGINE
#
# OBJETIVO:
# Consolidar somente motores AUTORIZADOS em um ranking determinístico.
#
# PRINCÍPIOS:
# - retorno futuro jamais entra como variável;
# - fatores de pesquisa não entram silenciosamente;
# - Turnaround fica bloqueado enquanto não houver validação temporal;
# - scores inválidos não recebem imputação favorável;
# - pesos devem estar explicitamente definidos;
# - ranking deve ser reproduzível e auditável.
#
# IMPORTANTE:
# O ranking atual é uma arquitetura operacional de referência.
# Ele NÃO declara que os pesos atuais são cientificamente ótimos.
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
        RANKING,
        TURNAROUND_ENGINE,
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

RANKING_DIR = (
    OUTPUT_DIR
    / "ranking"
)

RANKING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# EXCEÇÕES
# =============================================================================

class RankingError(RuntimeError):
    """Erro geral do Ranking Engine."""


class RankingIntegrityError(RankingError):
    """Erro estrutural ou de integridade."""


class RankingAuthorizationError(RankingError):
    """Tentativa de utilizar motor não autorizado."""


# =============================================================================
# PESOS DE REFERÊNCIA
#
# O estudo histórico anterior encontrou 70% Quality / 30% Valuation.
#
# IMPORTANTE:
# Neste novo estudo esse resultado é REFERÊNCIA, não conclusão científica.
#
# Portanto:
# - pode existir como baseline operacional;
# - deve ficar explicitamente identificado como REFERENCE;
# - não pode ser confundido com nova descoberta validada.
# =============================================================================

DEFAULT_ENGINE_WEIGHTS = {
    "QUALITY": 0.70,
    "VALUATION": 0.30,
    "TURNAROUND": 0.00,
}


# =============================================================================
# COLUNAS PROIBIDAS
# =============================================================================

FORBIDDEN_TERMS = (
    "FUTURE_RETURN",
    "RETORNO_FUTURO",
    "RETURN_FUTURE",
    "FUTURE_WINNER",
    "WINNER_LABEL",
    "TARGET_RETURN",
)


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

        raise RankingIntegrityError(
            f"{name} não é DataFrame."
        )

    if df.empty:

        raise RankingIntegrityError(
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

        raise RankingIntegrityError(
            f"FAIL-SAFE: {name} sem colunas obrigatórias: "
            f"{sorted(missing)}"
        )


def _normalize_issuer_id(
    value,
):

    if pd.isna(value):
        return None

    value = str(
        value
    ).strip()

    if not value:
        return None

    try:

        numeric = float(
            value
        )

        if numeric.is_integer():

            return str(
                int(numeric)
            )

    except Exception:
        pass

    return value


# =============================================================================
# PROTEÇÃO CONTRA LOOK-AHEAD
# =============================================================================

def assert_no_future_information(
    df: pd.DataFrame,
) -> None:

    forbidden = []

    explicitly_allowed_metadata = {
        "FUTURE_RETURN_USED",
        "FUTURE_RETURN_USED_QUALITY",
        "FUTURE_RETURN_USED_TURNAROUND",
        "FUTURE_RETURN_USED_VALUATION",
    }

    for column in df.columns:

        normalized = (
            str(column)
            .upper()
        )

        if normalized in explicitly_allowed_metadata:
            continue

        if any(
            term in normalized
            for term in FORBIDDEN_TERMS
        ):

            forbidden.append(
                column
            )

    if forbidden:

        raise RankingIntegrityError(
            "FAIL-SAFE: informação futura detectada "
            "no Ranking Engine: "
            f"{forbidden}"
        )


# =============================================================================
# VALIDAÇÃO DOS SCORES
# =============================================================================

def validate_score(
    series: pd.Series,
    name: str,
) -> pd.Series:

    score = pd.to_numeric(
        series,
        errors="coerce",
    )

    invalid = (
        score.notna()
        &
        (
            score.lt(0)
            |
            score.gt(1)
        )
    )

    if invalid.any():

        examples = (
            score.loc[
                invalid
            ]
            .head(10)
            .tolist()
        )

        raise RankingIntegrityError(
            f"FAIL-SAFE: {name} fora do intervalo [0,1]. "
            f"Exemplos={examples}"
        )

    return score


# =============================================================================
# PREPARAÇÃO DE CADA MOTOR
# =============================================================================

def prepare_quality(
    quality: pd.DataFrame,
) -> pd.DataFrame:

    _require_columns(
        quality,
        [
            "ISSUER_ID",
            "QUALITY_SCORE",
            "QUALITY_SCORE_VALID",
        ],
        "quality",
    )

    assert_no_future_information(
        quality
    )

    result = quality[
        [
            "ISSUER_ID",
            "QUALITY_SCORE",
            "QUALITY_SCORE_VALID",
        ]
    ].copy()

    result[
        "ISSUER_ID"
    ] = (
        result[
            "ISSUER_ID"
        ]
        .map(
            _normalize_issuer_id
        )
    )

    if (
        result[
            "ISSUER_ID"
        ]
        .duplicated()
        .any()
    ):

        raise RankingIntegrityError(
            "FAIL-SAFE: emissor duplicado em Quality."
        )

    result[
        "QUALITY_SCORE"
    ] = validate_score(
        result[
            "QUALITY_SCORE"
        ],
        "QUALITY_SCORE",
    )

    return result


def prepare_valuation(
    valuation: pd.DataFrame,
) -> pd.DataFrame:

    _require_columns(
        valuation,
        [
            "ISSUER_ID",
            "VALUATION_SCORE",
            "VALUATION_SCORE_VALID",
        ],
        "valuation",
    )

    assert_no_future_information(
        valuation
    )

    result = valuation[
        [
            "ISSUER_ID",
            "VALUATION_SCORE",
            "VALUATION_SCORE_VALID",
        ]
    ].copy()

    result[
        "ISSUER_ID"
    ] = (
        result[
            "ISSUER_ID"
        ]
        .map(
            _normalize_issuer_id
        )
    )

    if (
        result[
            "ISSUER_ID"
        ]
        .duplicated()
        .any()
    ):

        raise RankingIntegrityError(
            "FAIL-SAFE: emissor duplicado em Valuation."
        )

    result[
        "VALUATION_SCORE"
    ] = validate_score(
        result[
            "VALUATION_SCORE"
        ],
        "VALUATION_SCORE",
    )

    return result


def prepare_turnaround(
    turnaround: pd.DataFrame | None,
) -> pd.DataFrame | None:

    if turnaround is None:
        return None

    _require_columns(
        turnaround,
        [
            "ISSUER_ID",
            "TURNAROUND_RESEARCH_SCORE",
            "TURNAROUND_PRODUCTION_AUTHORIZED",
        ],
        "turnaround",
    )

    assert_no_future_information(
        turnaround
    )

    result = turnaround[
        [
            "ISSUER_ID",
            "TURNAROUND_RESEARCH_SCORE",
            "TURNAROUND_PRODUCTION_AUTHORIZED",
        ]
    ].copy()

    result[
        "ISSUER_ID"
    ] = (
        result[
            "ISSUER_ID"
        ]
        .map(
            _normalize_issuer_id
        )
    )

    if (
        result[
            "ISSUER_ID"
        ]
        .duplicated()
        .any()
    ):

        raise RankingIntegrityError(
            "FAIL-SAFE: emissor duplicado em Turnaround."
        )

    result[
        "TURNAROUND_RESEARCH_SCORE"
    ] = validate_score(
        result[
            "TURNAROUND_RESEARCH_SCORE"
        ],
        "TURNAROUND_RESEARCH_SCORE",
    )

    return result


# =============================================================================
# PESOS
# =============================================================================

def resolve_engine_weights() -> dict:

    configured = RANKING.get(
        "engine_weights",
        DEFAULT_ENGINE_WEIGHTS,
    )

    weights = {
        "QUALITY":
            float(
                configured.get(
                    "QUALITY",
                    DEFAULT_ENGINE_WEIGHTS[
                        "QUALITY"
                    ],
                )
            ),

        "VALUATION":
            float(
                configured.get(
                    "VALUATION",
                    DEFAULT_ENGINE_WEIGHTS[
                        "VALUATION"
                    ],
                )
            ),

        "TURNAROUND":
            float(
                configured.get(
                    "TURNAROUND",
                    DEFAULT_ENGINE_WEIGHTS[
                        "TURNAROUND"
                    ],
                )
            ),
    }

    for engine, weight in weights.items():

        if (
            not np.isfinite(
                weight
            )
            or
            weight < 0
        ):

            raise RankingIntegrityError(
                f"Peso inválido para {engine}: {weight}"
            )

    # -------------------------------------------------------------------------
    # TRAVA TURNAROUND
    # -------------------------------------------------------------------------

    turnaround_authorized = bool(
        TURNAROUND_ENGINE.get(
            "production_authorized",
            False,
        )
    )

    if (
        weights[
            "TURNAROUND"
        ]
        > 0
        and
        not turnaround_authorized
    ):

        raise RankingAuthorizationError(
            "FAIL-SAFE: tentativa de dar peso ao "
            "Turnaround sem autorização científica "
            "para produção."
        )

    total = sum(
        weights.values()
    )

    if total <= 0:

        raise RankingIntegrityError(
            "FAIL-SAFE: soma dos pesos é zero."
        )

    # -------------------------------------------------------------------------
    # NORMALIZAÇÃO
    # -------------------------------------------------------------------------

    weights = {
        engine:
            weight / total

        for engine, weight
        in weights.items()
    }

    return weights


# =============================================================================
# JUNÇÃO DOS MOTORES
# =============================================================================

def merge_engines(
    quality: pd.DataFrame,
    valuation: pd.DataFrame,
    turnaround: pd.DataFrame | None = None,
) -> pd.DataFrame:

    q = prepare_quality(
        quality
    )

    v = prepare_valuation(
        valuation
    )

    result = q.merge(
        v,
        on="ISSUER_ID",
        how="outer",
        validate="one_to_one",
    )

    t = prepare_turnaround(
        turnaround
    )

    if t is not None:

        result = result.merge(
            t,
            on="ISSUER_ID",
            how="left",
            validate="one_to_one",
        )

    return result


# =============================================================================
# ELEGIBILIDADE
# =============================================================================

def calculate_ranking_eligibility(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        df
        .copy()
    )

    quality_valid = (
        result[
            "QUALITY_SCORE_VALID"
        ]
        .fillna(False)
        .astype(bool)
    )

    valuation_valid = (
        result[
            "VALUATION_SCORE_VALID"
        ]
        .fillna(False)
        .astype(bool)
    )

    # -------------------------------------------------------------------------
    # Para o baseline atual exigimos as duas camadas válidas.
    #
    # Não imputamos score ausente como 0.50 porque isso poderia promover
    # artificialmente uma empresa com informação incompleta.
    # -------------------------------------------------------------------------

    result[
        "RANKING_ELIGIBLE"
    ] = (
        quality_valid
        &
        valuation_valid
    )

    reason = pd.Series(
        "OK",
        index=result.index,
        dtype="object",
    )

    reason.loc[
        ~quality_valid
        &
        valuation_valid
    ] = "QUALITY_INVALID"

    reason.loc[
        quality_valid
        &
        ~valuation_valid
    ] = "VALUATION_INVALID"

    reason.loc[
        ~quality_valid
        &
        ~valuation_valid
    ] = "QUALITY_AND_VALUATION_INVALID"

    result[
        "RANKING_ELIGIBILITY_REASON"
    ] = reason

    return result


# =============================================================================
# SCORE FINAL
# =============================================================================

def calculate_final_score(
    df: pd.DataFrame,
    weights: dict,
) -> pd.DataFrame:

    result = (
        df
        .copy()
    )

    result[
        "FINAL_SCORE"
    ] = np.nan

    eligible = (
        result[
            "RANKING_ELIGIBLE"
        ]
        .fillna(False)
        .astype(bool)
    )

    if not eligible.any():
        return result

    score = (
        weights[
            "QUALITY"
        ]
        *
        result.loc[
            eligible,
            "QUALITY_SCORE",
        ]
        +
        weights[
            "VALUATION"
        ]
        *
        result.loc[
            eligible,
            "VALUATION_SCORE",
        ]
    )

    # -------------------------------------------------------------------------
    # TURNAROUND
    #
    # Somente entra se:
    # 1. peso > 0;
    # 2. produção autorizada;
    # 3. score operacional validado existir.
    #
    # O score de PESQUISA jamais entra diretamente.
    # -------------------------------------------------------------------------

    if (
        weights[
            "TURNAROUND"
        ]
        > 0
    ):

        raise RankingAuthorizationError(
            "FAIL-SAFE: Turnaround possui peso positivo, "
            "mas ainda não existe TURNAROUND_PRODUCTION_SCORE "
            "implementado e validado. "
            "TURNAROUND_RESEARCH_SCORE não pode entrar no ranking."
        )

    result.loc[
        eligible,
        "FINAL_SCORE",
    ] = score

    result[
        "FINAL_SCORE"
    ] = (
        result[
            "FINAL_SCORE"
        ]
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    return result


# =============================================================================
# RANKING
# =============================================================================

def assign_final_rank(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        df
        .copy()
    )

    result[
        "FINAL_RANK"
    ] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    eligible = (
        result[
            "RANKING_ELIGIBLE"
        ]
        .fillna(False)
        .astype(bool)
        &
        result[
            "FINAL_SCORE"
        ]
        .notna()
    )

    ranks = (
        result.loc[
            eligible,
            "FINAL_SCORE",
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype("Int64")
    )

    result.loc[
        eligible,
        "FINAL_RANK",
    ] = ranks

    return result


# =============================================================================
# MOTOR PRINCIPAL
# =============================================================================

def run_ranking_engine(
    quality: pd.DataFrame,
    valuation: pd.DataFrame,
    context: PITContext,
    turnaround: pd.DataFrame | None = None,
) -> pd.DataFrame:

    weights = resolve_engine_weights()

    merged = merge_engines(
        quality=quality,
        valuation=valuation,
        turnaround=turnaround,
    )

    merged = (
        calculate_ranking_eligibility(
            merged
        )
    )

    result = calculate_final_score(
        merged,
        weights,
    )

    result = assign_final_rank(
        result
    )

    # -------------------------------------------------------------------------
    # METADADOS
    # -------------------------------------------------------------------------

    result[
        "RANKING_ENGINE_VERSION"
    ] = "0.1.0"

    result[
        "RANKING_MODEL"
    ] = "QUALITY_VALUATION_REFERENCE"

    result[
        "QUALITY_WEIGHT"
    ] = weights[
        "QUALITY"
    ]

    result[
        "VALUATION_WEIGHT"
    ] = weights[
        "VALUATION"
    ]

    result[
        "TURNAROUND_WEIGHT"
    ] = weights[
        "TURNAROUND"
    ]

    result[
        "FORMATION_DATE_RANKING"
    ] = context.formation_date

    result[
        "ACCOUNTING_CUTOFF_RANKING"
    ] = context.accounting_cutoff

    result[
        "FUTURE_RETURN_USED_RANKING"
    ] = False

    result[
        "TURNAROUND_RESEARCH_USED_IN_RANKING"
    ] = False

    # -------------------------------------------------------------------------
    # ORDENAÇÃO DETERMINÍSTICA
    # -------------------------------------------------------------------------

    result = (
        result
        .sort_values(
            [
                "RANKING_ELIGIBLE",
                "FINAL_SCORE",
                "ISSUER_ID",
            ],
            ascending=[
                False,
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    assert_no_future_information(
        result
    )

    # -------------------------------------------------------------------------
    # FAIL-SAFE FINAL
    # -------------------------------------------------------------------------

    if (
        result[
            "ISSUER_ID"
        ]
        .duplicated()
        .any()
    ):

        raise RankingIntegrityError(
            "FAIL-SAFE: emissor duplicado "
            "no ranking final."
        )

    if (
        result[
            "TURNAROUND_RESEARCH_USED_IN_RANKING"
        ]
        .any()
    ):

        raise RankingAuthorizationError(
            "FAIL-SAFE: sinal de pesquisa Turnaround "
            "entrou no ranking."
        )

    return result


# =============================================================================
# SELEÇÃO TOP N
# =============================================================================

def select_top_n(
    ranking: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:

    if n <= 0:

        raise RankingIntegrityError(
            "n deve ser maior que zero."
        )

    _require_columns(
        ranking,
        [
            "ISSUER_ID",
            "FINAL_SCORE",
            "FINAL_RANK",
            "RANKING_ELIGIBLE",
        ],
        "ranking",
    )

    selected = (
        ranking.loc[
            ranking[
                "RANKING_ELIGIBLE"
            ]
            .fillna(False)
            .astype(bool)
            &
            ranking[
                "FINAL_SCORE"
            ]
            .notna()
        ]
        .sort_values(
            [
                "FINAL_RANK",
                "ISSUER_ID",
            ],
            ascending=[
                True,
                True,
            ],
            kind="mergesort",
        )
        .head(
            n
        )
        .copy()
    )

    selected[
        "SELECTED"
    ] = True

    selected[
        "SELECTION_SIZE_REQUESTED"
    ] = int(
        n
    )

    return selected


# =============================================================================
# AUDITORIA
# =============================================================================

def audit_ranking(
    result: pd.DataFrame,
) -> dict:

    _require_dataframe(
        result,
        "ranking_result",
    )

    eligible = (
        result[
            "RANKING_ELIGIBLE"
        ]
        .fillna(False)
        .astype(bool)
    )

    weights = resolve_engine_weights()

    return {
        "status":
            "OK",

        "model":
            "QUALITY_VALUATION_REFERENCE",

        "issuers_total":
            int(
                len(
                    result
                )
            ),

        "issuers_eligible":
            int(
                eligible.sum()
            ),

        "eligibility_rate":
            float(
                eligible.mean()
            ),

        "weights":
            weights,

        "turnaround_production_authorized":
            bool(
                TURNAROUND_ENGINE.get(
                    "production_authorized",
                    False,
                )
            ),

        "turnaround_research_used":
            False,

        "future_return_used":
            False,

        "weight_optimization_performed":
            False,
    }


# =============================================================================
# SALVAR
# =============================================================================

def save_ranking(
    result: pd.DataFrame,
    context: PITContext,
    *,
    top_n: int = 10,
) -> dict:

    audit = audit_ranking(
        result
    )

    selected = select_top_n(
        result,
        n=top_n,
    )

    date_tag = (
        context
        .formation_date
        .strftime(
            "%Y%m%d"
        )
    )

    ranking_path = (
        RANKING_DIR
        / f"ranking_{date_tag}.csv"
    )

    selected_path = (
        RANKING_DIR
        / f"selected_top{top_n}_{date_tag}.csv"
    )

    manifest_path = (
        RANKING_DIR
        / f"ranking_manifest_{date_tag}.json"
    )

    result.to_csv(
        ranking_path,
        index=False,
        encoding="utf-8-sig",
    )

    selected.to_csv(
        selected_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "engine":
            "RANKING_ENGINE",

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

        "selection_size":
            int(
                top_n
            ),

        **audit,

        "files": {
            "ranking":
                str(
                    ranking_path
                ),

            "selected":
                str(
                    selected_path
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
        "ranking":
            ranking_path,

        "selected":
            selected_path,

        "manifest":
            manifest_path,
    }


# =============================================================================
# SELF-TEST
# =============================================================================

def _self_test():

    quality = pd.DataFrame(
        {
            "ISSUER_ID": [
                "1",
                "2",
                "3",
            ],

            "QUALITY_SCORE": [
                0.90,
                0.70,
                0.40,
            ],

            "QUALITY_SCORE_VALID": [
                True,
                True,
                True,
            ],
        }
    )

    valuation = pd.DataFrame(
        {
            "ISSUER_ID": [
                "1",
                "2",
                "3",
            ],

            "VALUATION_SCORE": [
                0.50,
                0.90,
                0.30,
            ],

            "VALUATION_SCORE_VALID": [
                True,
                True,
                True,
            ],
        }
    )

    turnaround = pd.DataFrame(
        {
            "ISSUER_ID": [
                "1",
                "2",
                "3",
            ],

            "TURNAROUND_RESEARCH_SCORE": [
                0.10,
                0.20,
                1.00,
            ],

            "TURNAROUND_PRODUCTION_AUTHORIZED": [
                False,
                False,
                False,
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

    result = run_ranking_engine(
        quality=quality,
        valuation=valuation,
        turnaround=turnaround,
        context=FakeContext(),
    )

    indexed = (
        result
        .set_index(
            "ISSUER_ID"
        )
    )

    # 70% Quality + 30% Valuation
    expected_1 = (
        0.70 * 0.90
        +
        0.30 * 0.50
    )

    if not np.isclose(
        indexed.loc[
            "1",
            "FINAL_SCORE",
        ],
        expected_1,
    ):

        raise RankingError(
            "SELF-TEST: score final incorreto."
        )

    # Empresa 3 possui Turnaround Research Score = 1.00.
    # Isso NÃO pode ajudá-la no ranking.
    if not (
        indexed.loc[
            "3",
            "TURNAROUND_RESEARCH_USED_IN_RANKING"
        ]
        == False
    ):

        raise RankingError(
            "SELF-TEST: Turnaround experimental "
            "entrou no ranking."
        )

    if (
        result[
            "FUTURE_RETURN_USED_RANKING"
        ]
        .any()
    ):

        raise RankingError(
            "SELF-TEST: retorno futuro utilizado."
        )

    return True


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("B3 STOCK SELECTOR — RANKING ENGINE")
    print("=" * 72)

    _self_test()

    print("Self-test: OK")
    print("Quality: AUTORIZADO")
    print("Valuation: AUTORIZADO COMO REFERÊNCIA")
    print("Turnaround Research: BLOQUEADO")
    print("Retorno futuro: PROIBIDO")
    print("Pesos atuais: 70% Quality / 30% Valuation")
    print("Status dos pesos: REFERÊNCIA, NÃO NOVA CONCLUSÃO")
    print("Missing score: NÃO RECEBE IMPUTAÇÃO FAVORÁVEL")
    print("Ranking: DETERMINÍSTICO")
    print("Auditoria: ATIVA")

    print("=" * 72)
