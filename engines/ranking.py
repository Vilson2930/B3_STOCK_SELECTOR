# =============================================================================
# B3 STOCK SELECTOR
# Arquivo: engines/ranking.py
#
# RANKING ENGINE
#
# OBJETIVO:
# Consolidar somente motores autorizados em ranking determinístico.
#
# IMPORTANTE:
# - retorno futuro jamais entra como variável;
# - Turnaround Research NÃO entra no ranking final;
# - pesos 70/30 permanecem apenas como baseline de referência;
# - não transforma pesquisa experimental em regra de produção;
# - execução é determinística e auditável.
# =============================================================================

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
#
# Importamos o módulo inteiro.
# Assim a ausência de RANKING não é confundida com ausência de config.py.
# =============================================================================

try:
    import config as project_config
except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar config.py."
    ) from exc


if not hasattr(project_config, "OUTPUT_DIR"):
    raise RuntimeError(
        "FAIL-SAFE: OUTPUT_DIR ausente em config.py."
    )


OUTPUT_DIR = project_config.OUTPUT_DIR


RANKING = getattr(
    project_config,
    "RANKING",
    getattr(
        project_config,
        "RANKING_ENGINE",
        {},
    ),
)


TURNAROUND_ENGINE = getattr(
    project_config,
    "TURNAROUND_ENGINE",
    {},
)


if RANKING is None:
    RANKING = {}

if TURNAROUND_ENGINE is None:
    TURNAROUND_ENGINE = {}

if not isinstance(RANKING, dict):
    raise RuntimeError(
        "FAIL-SAFE: configuração RANKING/RANKING_ENGINE "
        "deve ser um dicionário."
    )

if not isinstance(TURNAROUND_ENGINE, dict):
    raise RuntimeError(
        "FAIL-SAFE: TURNAROUND_ENGINE deve ser um dicionário."
    )


try:
    from data.pit import PITContext
except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar PITContext."
    ) from exc


# =============================================================================
# DIRETÓRIO
# =============================================================================

RANKING_DIR = OUTPUT_DIR / "ranking"
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
    """Tentativa de utilizar componente não autorizado."""


# =============================================================================
# BASELINE DE REFERÊNCIA
#
# Resultado histórico anterior:
#
# Quality   70%
# Valuation 30%
#
# IMPORTANTE:
# Isto permanece REFERÊNCIA.
# Não é conclusão científica nova do estudo atual.
#
# O Turnaround 50/30/20 desenvolvido no estudo atual continua sendo
# pesquisa e NÃO entra silenciosamente neste ranking.
# =============================================================================

DEFAULT_ENGINE_WEIGHTS = {
    "QUALITY": 0.70,
    "VALUATION": 0.30,
    "TURNAROUND": 0.00,
}


# =============================================================================
# PROTEÇÃO CONTRA INFORMAÇÃO FUTURA
# =============================================================================

FORBIDDEN_TERMS = (
    "FUTURE_RETURN",
    "RETORNO_FUTURO",
    "RETURN_FUTURE",
    "FUTURE_WINNER",
    "WINNER_LABEL",
    "TARGET_RETURN",
)


ALLOWED_FUTURE_AUDIT_COLUMNS = {
    "FUTURE_RETURN_USED",
    "FUTURE_RETURN_USED_QUALITY",
    "FUTURE_RETURN_USED_TURNAROUND",
    "FUTURE_RETURN_USED_VALUATION",
    "FUTURE_RETURN_USED_RANKING",
    "FUTURE_RETURN_USED_RISK",
    "FUTURE_RETURN_USED_REPORT",
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

    if not isinstance(df, pd.DataFrame):
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


def _normalize_issuer_id(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    try:
        numeric = float(value)

        if numeric.is_integer():
            return str(int(numeric))

    except Exception:
        pass

    return value


def _as_bool_series(
    series: pd.Series,
) -> pd.Series:

    if pd.api.types.is_bool_dtype(series):
        return (
            series
            .fillna(False)
            .astype(bool)
        )

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return normalized.isin(
        {
            "TRUE",
            "1",
            "YES",
            "SIM",
        }
    )


# =============================================================================
# PROTEÇÃO CONTRA LOOK-AHEAD
# =============================================================================

def assert_no_future_information(
    df: pd.DataFrame,
) -> None:

    forbidden = []

    for column in df.columns:

        normalized = str(column).upper()

        if normalized in ALLOWED_FUTURE_AUDIT_COLUMNS:

            values = df[column].dropna()

            if not values.empty:

                normalized_values = (
                    values
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )

                invalid = ~normalized_values.isin(
                    {
                        "FALSE",
                        "0",
                    }
                )

                if invalid.any():
                    raise RankingIntegrityError(
                        "FAIL-SAFE: metadado "
                        f"{column} indica uso de retorno futuro."
                    )

            continue

        if any(
            term in normalized
            for term in FORBIDDEN_TERMS
        ):
            forbidden.append(column)

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
            score.loc[invalid]
            .head(10)
            .tolist()
        )

        raise RankingIntegrityError(
            f"FAIL-SAFE: {name} fora do intervalo [0,1]. "
            f"Exemplos={examples}"
        )

    return score


# =============================================================================
# QUALITY
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

    if result["ISSUER_ID"].isna().any():
        raise RankingIntegrityError(
            "FAIL-SAFE: ISSUER_ID ausente em Quality."
        )

    if result["ISSUER_ID"].duplicated().any():
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

    result[
        "QUALITY_SCORE_VALID"
    ] = _as_bool_series(
        result[
            "QUALITY_SCORE_VALID"
        ]
    )

    return result


# =============================================================================
# VALUATION
# =============================================================================

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

    if result["ISSUER_ID"].isna().any():
        raise RankingIntegrityError(
            "FAIL-SAFE: ISSUER_ID ausente em Valuation."
        )

    if result["ISSUER_ID"].duplicated().any():
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

    result[
        "VALUATION_SCORE_VALID"
    ] = _as_bool_series(
        result[
            "VALUATION_SCORE_VALID"
        ]
    )

    return result


# =============================================================================
# TURNAROUND
#
# Somente auditoria.
# O score de pesquisa NÃO entra no ranking final.
# =============================================================================

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

    columns = [
        "ISSUER_ID",
        "TURNAROUND_RESEARCH_SCORE",
        "TURNAROUND_PRODUCTION_AUTHORIZED",
    ]

    if (
        "TURNAROUND_RESEARCH_SCORE_COMPLETE"
        in turnaround.columns
    ):
        columns.append(
            "TURNAROUND_RESEARCH_SCORE_COMPLETE"
        )

    if (
        "TURNAROUND_SCORE_MODEL"
        in turnaround.columns
    ):
        columns.append(
            "TURNAROUND_SCORE_MODEL"
        )

    result = turnaround[
        columns
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

    if result["ISSUER_ID"].isna().any():
        raise RankingIntegrityError(
            "FAIL-SAFE: ISSUER_ID ausente em Turnaround."
        )

    if result["ISSUER_ID"].duplicated().any():
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

    if (
        "TURNAROUND_RESEARCH_SCORE_COMPLETE"
        in result.columns
    ):

        result[
            "TURNAROUND_RESEARCH_SCORE_COMPLETE"
        ] = validate_score(
            result[
                "TURNAROUND_RESEARCH_SCORE_COMPLETE"
            ],
            "TURNAROUND_RESEARCH_SCORE_COMPLETE",
        )

    result[
        "TURNAROUND_PRODUCTION_AUTHORIZED"
    ] = _as_bool_series(
        result[
            "TURNAROUND_PRODUCTION_AUTHORIZED"
        ]
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

    if configured is None:
        configured = DEFAULT_ENGINE_WEIGHTS

    if not isinstance(configured, dict):
        raise RankingIntegrityError(
            "FAIL-SAFE: engine_weights deve ser dicionário."
        )

    weights = {
        "QUALITY": float(
            configured.get(
                "QUALITY",
                DEFAULT_ENGINE_WEIGHTS["QUALITY"],
            )
        ),
        "VALUATION": float(
            configured.get(
                "VALUATION",
                DEFAULT_ENGINE_WEIGHTS["VALUATION"],
            )
        ),
        "TURNAROUND": float(
            configured.get(
                "TURNAROUND",
                DEFAULT_ENGINE_WEIGHTS["TURNAROUND"],
            )
        ),
    }

    for engine, weight in weights.items():

        if (
            not np.isfinite(weight)
            or
            weight < 0
        ):
            raise RankingIntegrityError(
                f"Peso inválido para {engine}: {weight}"
            )

    # -------------------------------------------------------------------------
    # TRAVA CIENTÍFICA DO TURNAROUND
    # -------------------------------------------------------------------------

    turnaround_authorized = bool(
        TURNAROUND_ENGINE.get(
            "production_authorized",
            False,
        )
    )

    if (
        weights["TURNAROUND"] > 0
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

    return {
        engine: weight / total
        for engine, weight
        in weights.items()
    }


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

    result = df.copy()

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

    # Baseline exige as duas camadas.
    # Não existe imputação favorável para score ausente.

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

    result = df.copy()

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

    # -------------------------------------------------------------------------
    # TURNAROUND RESEARCH NÃO ENTRA.
    # -------------------------------------------------------------------------

    if weights["TURNAROUND"] > 0:

        raise RankingAuthorizationError(
            "FAIL-SAFE: Turnaround possui peso positivo, "
            "mas ainda não existe TURNAROUND_PRODUCTION_SCORE "
            "implementado e temporalmente validado. "
            "TURNAROUND_RESEARCH_SCORE não pode entrar no ranking."
        )

    operational_weight = (
        weights["QUALITY"]
        +
        weights["VALUATION"]
    )

    if operational_weight <= 0:
        raise RankingIntegrityError(
            "FAIL-SAFE: Quality + Valuation possuem peso zero."
        )

    quality_weight = (
        weights["QUALITY"]
        /
        operational_weight
    )

    valuation_weight = (
        weights["VALUATION"]
        /
        operational_weight
    )

    score = (
        quality_weight
        *
        result.loc[
            eligible,
            "QUALITY_SCORE",
        ]
        +
        valuation_weight
        *
        result.loc[
            eligible,
            "VALUATION_SCORE",
        ]
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
# RANK FINAL
# =============================================================================

def assign_final_rank(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

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

    merged = calculate_ranking_eligibility(
        merged
    )

    result = calculate_final_score(
        merged,
        weights,
    )

    result = assign_final_rank(
        result
    )

    result[
        "RANKING_ENGINE_VERSION"
    ] = "0.1.1"

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

    if result["ISSUER_ID"].duplicated().any():
        raise RankingIntegrityError(
            "FAIL-SAFE: emissor duplicado no ranking final."
        )

    if (
        result[
            "TURNAROUND_RESEARCH_USED_IN_RANKING"
        ]
        .fillna(False)
        .astype(bool)
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
        .head(n)
        .copy()
    )

    selected[
        "SELECTED"
    ] = True

    selected[
        "SELECTION_SIZE_REQUESTED"
    ] = int(n)

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
        "status": "OK",

        "model":
            "QUALITY_VALUATION_REFERENCE",

        "issuers_total":
            int(len(result)),

        "issuers_eligible":
            int(eligible.sum()),

        "eligibility_rate":
            float(eligible.mean()),

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
        .strftime("%Y%m%d")
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

        "version":
            "0.1.1",

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
            int(top_n),

        **audit,

        "files": {
            "ranking":
                str(ranking_path),

            "selected":
                str(selected_path),
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

            "TURNAROUND_RESEARCH_SCORE_COMPLETE": [
                0.10,
                0.20,
                1.00,
            ],

            "TURNAROUND_PRODUCTION_AUTHORIZED": [
                False,
                False,
                False,
            ],

            "TURNAROUND_SCORE_MODEL": [
                "MB_50_ML_30_ROE_20_RESEARCH",
                "MB_50_ML_30_ROE_20_RESEARCH",
                "MB_50_ML_30_ROE_20_RESEARCH",
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

    weights = resolve_engine_weights()

    operational_total = (
        weights["QUALITY"]
        +
        weights["VALUATION"]
    )

    expected_1 = (
        (
            weights["QUALITY"]
            /
            operational_total
        )
        * 0.90
        +
        (
            weights["VALUATION"]
            /
            operational_total
        )
        * 0.50
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

    # O emissor 3 possui Turnaround Research = 1.
    # Isso não pode ajudá-lo no ranking de produção.

    if bool(
        indexed.loc[
            "3",
            "TURNAROUND_RESEARCH_USED_IN_RANKING",
        ]
    ):
        raise RankingError(
            "SELF-TEST: Turnaround experimental entrou no ranking."
        )

    if (
        result[
            "FUTURE_RETURN_USED_RANKING"
        ]
        .fillna(False)
        .astype(bool)
        .any()
    ):
        raise RankingError(
            "SELF-TEST: retorno futuro utilizado."
        )

    # -------------------------------------------------------------------------
    # TESTE DE CONTAMINAÇÃO FUTURA
    # -------------------------------------------------------------------------

    contaminated = quality.copy()

    contaminated[
        "FUTURE_RETURN"
    ] = [
        1.0,
        2.0,
        3.0,
    ]

    future_blocked = False

    try:
        run_ranking_engine(
            quality=contaminated,
            valuation=valuation,
            turnaround=turnaround,
            context=FakeContext(),
        )

    except RankingIntegrityError:
        future_blocked = True

    if not future_blocked:
        raise RankingError(
            "SELF-TEST: retorno futuro real não foi bloqueado."
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

    weights = resolve_engine_weights()

    print("Self-test: OK")
    print("Quality: BASELINE AUTORIZADO")
    print("Valuation: BASELINE DE REFERÊNCIA")
    print("Turnaround Research 50/30/20: BLOQUEADO NO RANKING")
    print("Retorno futuro: PROIBIDO")
    print(
        "Pesos atuais: "
        f"{weights['QUALITY']:.0%} Quality / "
        f"{weights['VALUATION']:.0%} Valuation"
    )
    print("Status dos pesos: REFERÊNCIA, NÃO NOVA CONCLUSÃO")
    print("Missing score: NÃO RECEBE IMPUTAÇÃO FAVORÁVEL")
    print("Ranking: DETERMINÍSTICO")
    print("Auditoria: ATIVA")

    print("=" * 72)
