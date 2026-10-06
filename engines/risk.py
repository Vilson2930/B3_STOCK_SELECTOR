# =============================================================================
# B3 STOCK SELECTOR
# Arquivo: engines/risk.py
#
# RISK ENGINE
#
# OBJETIVO:
# Transformar o ranking elegível em uma seleção controlada por risco.
#
# RESPONSABILIDADE:
# - preservar a ordem do Ranking Engine;
# - impedir concentração excessiva;
# - limitar exposição por setor quando a informação existir;
# - limitar quantidade de posições;
# - produzir pesos de carteira;
# - registrar exclusões por risco;
# - manter rastreabilidade completa.
#
# NÃO FAZ:
# - descobrir fatores;
# - alterar Quality Score;
# - alterar Valuation Score;
# - descobrir/recalcular o score fundamental 50/30/20;
# - utilizar retorno futuro;
# - otimizar carteira usando retorno futuro;
# - estimar retorno esperado.
# =============================================================================

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
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


try:
    from data.pit import PITContext
except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar PITContext."
    ) from exc


RISK_CONFIG = getattr(
    project_config,
    "RISK",
    {},
)

if RISK_CONFIG is None:
    RISK_CONFIG = {}

if not isinstance(RISK_CONFIG, dict):
    raise RuntimeError(
        "FAIL-SAFE: RISK deve ser dicionário."
    )


# =============================================================================
# DIRETÓRIO
# =============================================================================

RISK_DIR = OUTPUT_DIR / "risk"

RISK_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# EXCEÇÕES
# =============================================================================

class RiskError(RuntimeError):
    """Erro geral do Risk Engine."""


class RiskIntegrityError(RiskError):
    """Erro estrutural ou de integridade."""


class RiskConfigurationError(RiskError):
    """Configuração de risco inválida."""


# =============================================================================
# STATUS
# =============================================================================

STATUS_SELECTED = "SELECTED"
STATUS_REJECTED = "REJECTED"
STATUS_NOT_ELIGIBLE = "NOT_ELIGIBLE"

REASON_SELECTED = "SELECTED_BY_RANK"
REASON_NOT_ELIGIBLE = "RANKING_NOT_ELIGIBLE"
REASON_POSITION_LIMIT = "MAX_POSITIONS_REACHED"
REASON_SECTOR_LIMIT = "SECTOR_LIMIT"
REASON_INVALID_SCORE = "INVALID_FINAL_SCORE"


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


# Metadados de auditoria são permitidos somente se indicarem False.
ALLOWED_METADATA = {
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
        raise RiskIntegrityError(
            f"{name} não é DataFrame."
        )

    if df.empty:
        raise RiskIntegrityError(
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
        raise RiskIntegrityError(
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


def _metadata_is_false(
    series: pd.Series,
) -> bool:

    values = series.dropna()

    if values.empty:
        return True

    normalized = (
        values
        .astype(str)
        .str.strip()
        .str.upper()
    )

    allowed_false = {
        "FALSE",
        "0",
        "0.0",
        "NO",
        "NAO",
        "NÃO",
    }

    return bool(
        normalized.isin(
            allowed_false
        ).all()
    )


# =============================================================================
# PROTEÇÃO CONTRA LOOK-AHEAD
# =============================================================================

def assert_no_future_information(
    df: pd.DataFrame,
) -> None:

    forbidden = []

    for column in df.columns:

        normalized = str(
            column
        ).upper()

        if normalized in ALLOWED_METADATA:

            if not _metadata_is_false(
                df[column]
            ):
                raise RiskIntegrityError(
                    "FAIL-SAFE: metadado de auditoria "
                    f"{column} indica utilização de "
                    "informação futura."
                )

            continue

        if any(
            term in normalized
            for term in FORBIDDEN_TERMS
        ):
            forbidden.append(
                column
            )

    if forbidden:
        raise RiskIntegrityError(
            "FAIL-SAFE: informação futura detectada "
            "no Risk Engine: "
            f"{forbidden}"
        )


# =============================================================================
# PROTEÇÃO TURNAROUND RESEARCH
# =============================================================================

def assert_ranking_study_model_integrity(
    df: pd.DataFrame,
) -> None:
    """
    O Risk Engine não descobre nem recalcula fatores.

    Ele pode receber o score 50/30/20 já produzido e autorizado pelo
    Ranking Engine, desde que os metadados confirmem exatamente o modelo
    do estudo. Qualquer uso ambíguo/legado de Turnaround continua bloqueado.
    """

    used_column = "TURNAROUND_RESEARCH_USED_IN_RANKING"

    if used_column not in df.columns:
        return

    used = (
        df[used_column]
        .fillna(False)
        .astype(bool)
    )

    if not used.any():
        return

    required_metadata = {
        "RANKING_MODEL",
        "MARGEM_BRUTA_WEIGHT",
        "MARGEM_LIQUIDA_WEIGHT",
        "ROE_WEIGHT",
    }

    missing = required_metadata - set(df.columns)

    if missing:
        raise RiskIntegrityError(
            "FAIL-SAFE: ranking usa o score do estudo, mas faltam "
            f"metadados obrigatórios: {sorted(missing)}"
        )

    models = (
        df.loc[used, "RANKING_MODEL"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if models != ["MB_50_ML_30_ROE_20_STUDY"]:
        raise RiskIntegrityError(
            "FAIL-SAFE: modelo de ranking não autorizado no Risk Engine. "
            f"Modelos detectados={models}"
        )

    expected_weights = {
        "MARGEM_BRUTA_WEIGHT": 0.50,
        "MARGEM_LIQUIDA_WEIGHT": 0.30,
        "ROE_WEIGHT": 0.20,
    }

    for column, expected in expected_weights.items():

        values = pd.to_numeric(
            df.loc[used, column],
            errors="coerce",
        )

        if values.isna().any():
            raise RiskIntegrityError(
                "FAIL-SAFE: peso do estudo ausente/inválido "
                f"em {column}."
            )

        if not np.isclose(
            values.to_numpy(dtype=float),
            expected,
            atol=1e-12,
            rtol=0.0,
        ).all():
            raise RiskIntegrityError(
                "FAIL-SAFE: peso divergente do estudo em "
                f"{column}. Esperado={expected}."
            )

    # Se os metadados legados existirem, eles não podem reintroduzir
    # Quality ou Valuation no score principal.
    for column in (
        "QUALITY_WEIGHT",
        "VALUATION_WEIGHT",
    ):
        if column in df.columns:
            values = pd.to_numeric(
                df.loc[used, column],
                errors="coerce",
            )
            if values.isna().any() or not np.isclose(
                values.to_numpy(dtype=float),
                0.0,
                atol=1e-12,
                rtol=0.0,
            ).all():
                raise RiskIntegrityError(
                    "FAIL-SAFE: peso legado diferente de zero "
                    f"detectado em {column}."
                )


# =============================================================================
# CONFIG
# =============================================================================

def _get_positive_int(
    key: str,
    default: int,
) -> int:

    value = RISK_CONFIG.get(
        key,
        default,
    )

    try:
        value = int(
            value
        )

    except Exception as exc:
        raise RiskConfigurationError(
            f"Configuração {key} inválida: {value}"
        ) from exc

    if value <= 0:
        raise RiskConfigurationError(
            f"{key} deve ser > 0."
        )

    return value


def resolve_risk_config() -> dict:

    max_positions = _get_positive_int(
        "max_positions",
        10,
    )

    min_positions = _get_positive_int(
        "min_positions",
        1,
    )

    if min_positions > max_positions:
        raise RiskConfigurationError(
            "min_positions não pode ser maior "
            "que max_positions."
        )

    max_sector_weight = RISK_CONFIG.get(
        "max_sector_weight",
        None,
    )

    if max_sector_weight is not None:

        try:
            max_sector_weight = float(
                max_sector_weight
            )

        except Exception as exc:
            raise RiskConfigurationError(
                "max_sector_weight inválido."
            ) from exc

        if not (
            0 < max_sector_weight <= 1
        ):
            raise RiskConfigurationError(
                "max_sector_weight deve estar "
                "entre 0 e 1."
            )

    weighting_method = str(
        RISK_CONFIG.get(
            "weighting_method",
            "EQUAL_WEIGHT",
        )
    ).strip().upper()

    allowed_weighting = {
        "EQUAL_WEIGHT",
    }

    if weighting_method not in allowed_weighting:
        raise RiskConfigurationError(
            "Método de pesos não autorizado. "
            f"Permitidos={sorted(allowed_weighting)}"
        )

    return {
        "max_positions": max_positions,
        "min_positions": min_positions,
        "max_sector_weight": max_sector_weight,
        "weighting_method": weighting_method,
    }


# =============================================================================
# PREPARAÇÃO
# =============================================================================

def prepare_ranking(
    ranking: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        "ISSUER_ID",
        "FINAL_SCORE",
        "FINAL_RANK",
        "RANKING_ELIGIBLE",
    }

    _require_columns(
        ranking,
        required,
        "ranking",
    )

    assert_no_future_information(
        ranking
    )

    assert_ranking_study_model_integrity(
        ranking
    )

    result = ranking.copy()

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

    if result[
        "ISSUER_ID"
    ].isna().any():
        raise RiskIntegrityError(
            "FAIL-SAFE: ISSUER_ID ausente."
        )

    if result[
        "ISSUER_ID"
    ].duplicated().any():
        raise RiskIntegrityError(
            "FAIL-SAFE: emissor duplicado "
            "no ranking."
        )

    result[
        "FINAL_SCORE"
    ] = pd.to_numeric(
        result[
            "FINAL_SCORE"
        ],
        errors="coerce",
    )

    result[
        "FINAL_RANK"
    ] = pd.to_numeric(
        result[
            "FINAL_RANK"
        ],
        errors="coerce",
    )

    result[
        "RANKING_ELIGIBLE"
    ] = (
        result[
            "RANKING_ELIGIBLE"
        ]
        .fillna(False)
        .astype(bool)
    )

    invalid_score = (
        result[
            "FINAL_SCORE"
        ]
        .notna()
        &
        (
            result[
                "FINAL_SCORE"
            ].lt(0)
            |
            result[
                "FINAL_SCORE"
            ].gt(1)
        )
    )

    if invalid_score.any():
        raise RiskIntegrityError(
            "FAIL-SAFE: FINAL_SCORE fora "
            "do intervalo [0,1]."
        )

    return result


# =============================================================================
# SETOR
# =============================================================================

def resolve_sector_column(
    df: pd.DataFrame,
) -> Optional[str]:

    candidates = (
        "FCA_SETOR_ATIVIDADE",
        "SECTOR",
        "SETOR",
        "SECTOR_BUCKET",
    )

    for column in candidates:

        if column in df.columns:
            return column

    return None


# =============================================================================
# LIMITE SETORIAL
# =============================================================================

def _max_sector_positions(
    max_positions: int,
    max_sector_weight: Optional[float],
) -> Optional[int]:

    if max_sector_weight is None:
        return None

    limit = int(
        np.floor(
            max_sector_weight
            *
            max_positions
        )
    )

    return max(
        1,
        limit,
    )


# =============================================================================
# SELEÇÃO CONTROLADA
# =============================================================================

def apply_risk_selection(
    ranking: pd.DataFrame,
    risk_config: dict,
) -> pd.DataFrame:

    result = ranking.copy()

    max_positions = int(
        risk_config[
            "max_positions"
        ]
    )

    max_sector_weight = (
        risk_config[
            "max_sector_weight"
        ]
    )

    sector_column = resolve_sector_column(
        result
    )

    sector_position_limit = _max_sector_positions(
        max_positions,
        max_sector_weight,
    )

    result[
        "RISK_STATUS"
    ] = STATUS_NOT_ELIGIBLE

    result[
        "RISK_REASON"
    ] = REASON_NOT_ELIGIBLE

    result[
        "RISK_SELECTED"
    ] = False

    eligible_mask = (
        result[
            "RANKING_ELIGIBLE"
        ]
        &
        result[
            "FINAL_SCORE"
        ].notna()
        &
        result[
            "FINAL_RANK"
        ].notna()
    )

    invalid_score_mask = (
        result[
            "RANKING_ELIGIBLE"
        ]
        &
        (
            result[
                "FINAL_SCORE"
            ].isna()
            |
            result[
                "FINAL_RANK"
            ].isna()
        )
    )

    result.loc[
        invalid_score_mask,
        "RISK_STATUS",
    ] = STATUS_REJECTED

    result.loc[
        invalid_score_mask,
        "RISK_REASON",
    ] = REASON_INVALID_SCORE

    candidates = (
        result.loc[
            eligible_mask
        ]
        .sort_values(
            [
                "FINAL_RANK",
                "FINAL_SCORE",
                "ISSUER_ID",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="mergesort",
        )
    )

    selected_indices = []
    sector_counts = {}

    for index, row in candidates.iterrows():

        if len(
            selected_indices
        ) >= max_positions:

            result.loc[
                index,
                "RISK_STATUS",
            ] = STATUS_REJECTED

            result.loc[
                index,
                "RISK_REASON",
            ] = REASON_POSITION_LIMIT

            continue

        if (
            sector_column is not None
            and
            sector_position_limit is not None
        ):

            sector_value = row[
                sector_column
            ]

            if pd.isna(
                sector_value
            ):
                sector_key = "__UNKNOWN__"

            else:
                sector_key = str(
                    sector_value
                ).strip()

                if not sector_key:
                    sector_key = "__UNKNOWN__"

            current_count = (
                sector_counts.get(
                    sector_key,
                    0,
                )
            )

            if (
                current_count
                >= sector_position_limit
            ):

                result.loc[
                    index,
                    "RISK_STATUS",
                ] = STATUS_REJECTED

                result.loc[
                    index,
                    "RISK_REASON",
                ] = REASON_SECTOR_LIMIT

                continue

        else:
            sector_key = None

        selected_indices.append(
            index
        )

        result.loc[
            index,
            "RISK_SELECTED",
        ] = True

        result.loc[
            index,
            "RISK_STATUS",
        ] = STATUS_SELECTED

        result.loc[
            index,
            "RISK_REASON",
        ] = REASON_SELECTED

        if sector_key is not None:

            sector_counts[
                sector_key
            ] = (
                sector_counts.get(
                    sector_key,
                    0,
                )
                + 1
            )

    remaining_eligible = (
        eligible_mask
        &
        ~result[
            "RISK_SELECTED"
        ]
        &
        result[
            "RISK_STATUS"
        ].eq(
            STATUS_NOT_ELIGIBLE
        )
    )

    result.loc[
        remaining_eligible,
        "RISK_STATUS",
    ] = STATUS_REJECTED

    result.loc[
        remaining_eligible,
        "RISK_REASON",
    ] = REASON_POSITION_LIMIT

    return result


# =============================================================================
# PESOS
# =============================================================================

def assign_portfolio_weights(
    df: pd.DataFrame,
    risk_config: dict,
) -> pd.DataFrame:

    result = df.copy()

    result[
        "PORTFOLIO_WEIGHT"
    ] = 0.0

    selected = (
        result[
            "RISK_SELECTED"
        ]
        .fillna(False)
        .astype(bool)
    )

    n_selected = int(
        selected.sum()
    )

    if n_selected == 0:
        return result

    method = (
        risk_config[
            "weighting_method"
        ]
    )

    if method == "EQUAL_WEIGHT":

        weight = (
            1.0
            /
            n_selected
        )

        result.loc[
            selected,
            "PORTFOLIO_WEIGHT",
        ] = weight

    else:
        raise RiskConfigurationError(
            f"Método não implementado: {method}"
        )

    total_weight = float(
        result[
            "PORTFOLIO_WEIGHT"
        ].sum()
    )

    if not np.isclose(
        total_weight,
        1.0,
        atol=1e-10,
    ):
        raise RiskIntegrityError(
            "FAIL-SAFE: pesos da carteira "
            f"não somam 1. Soma={total_weight}"
        )

    return result


# =============================================================================
# AUDITORIA DE CONCENTRAÇÃO
# =============================================================================

def calculate_concentration_audit(
    df: pd.DataFrame,
) -> dict:

    selected = (
        df.loc[
            df[
                "RISK_SELECTED"
            ]
            .fillna(False)
            .astype(bool)
        ]
        .copy()
    )

    if selected.empty:

        return {
            "selected_positions": 0,
            "largest_position_weight": None,
            "sector_weights": {},
        }

    largest_position = float(
        selected[
            "PORTFOLIO_WEIGHT"
        ].max()
    )

    sector_column = resolve_sector_column(
        selected
    )

    sector_weights = {}

    if sector_column is not None:

        temp = (
            selected[
                [
                    sector_column,
                    "PORTFOLIO_WEIGHT",
                ]
            ]
            .copy()
        )

        temp[
            sector_column
        ] = (
            temp[
                sector_column
            ]
            .fillna(
                "__UNKNOWN__"
            )
            .astype(str)
        )

        sector_weights = (
            temp
            .groupby(
                sector_column
            )[
                "PORTFOLIO_WEIGHT"
            ]
            .sum()
            .sort_values(
                ascending=False
            )
            .to_dict()
        )

        sector_weights = {
            str(key): float(value)
            for key, value
            in sector_weights.items()
        }

    return {
        "selected_positions":
            int(
                len(
                    selected
                )
            ),
        "largest_position_weight":
            largest_position,
        "sector_weights":
            sector_weights,
    }


# =============================================================================
# MOTOR PRINCIPAL
# =============================================================================

def run_risk_engine(
    ranking: pd.DataFrame,
    context: PITContext,
) -> pd.DataFrame:

    risk_config = resolve_risk_config()

    base = prepare_ranking(
        ranking
    )

    result = apply_risk_selection(
        base,
        risk_config,
    )

    result = assign_portfolio_weights(
        result,
        risk_config,
    )

    selected_count = int(
        result[
            "RISK_SELECTED"
        ].sum()
    )

    if (
        selected_count
        >
        risk_config[
            "max_positions"
        ]
    ):
        raise RiskIntegrityError(
            "FAIL-SAFE: quantidade selecionada "
            "superou max_positions."
        )

    minimum_reached = (
        selected_count
        >=
        risk_config[
            "min_positions"
        ]
    )

    result[
        "MIN_POSITIONS_REACHED"
    ] = minimum_reached

    result[
        "RISK_ENGINE_VERSION"
    ] = "0.2.0"

    result[
        "FORMATION_DATE_RISK"
    ] = context.formation_date

    result[
        "ACCOUNTING_CUTOFF_RISK"
    ] = context.accounting_cutoff

    result[
        "MARKET_CUTOFF_RISK"
    ] = context.market_cutoff

    result[
        "FUTURE_RETURN_USED_RISK"
    ] = False

    result[
        "TURNAROUND_RESEARCH_USED_RISK"
    ] = False

    result[
        "STUDY_SCORE_FROM_RANKING_ACCEPTED_RISK"
    ] = (
        result[
            "TURNAROUND_RESEARCH_USED_IN_RANKING"
        ]
        .fillna(False)
        .astype(bool)
        if "TURNAROUND_RESEARCH_USED_IN_RANKING" in result.columns
        else False
    )

    result = (
        result
        .sort_values(
            [
                "RISK_SELECTED",
                "FINAL_RANK",
                "ISSUER_ID",
            ],
            ascending=[
                False,
                True,
                True,
            ],
            kind="mergesort",
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    assert_no_future_information(
        result
    )

    assert_ranking_study_model_integrity(
        result
    )

    return result


# =============================================================================
# CARTEIRA FINAL
# =============================================================================

def get_selected_portfolio(
    risk_result: pd.DataFrame,
) -> pd.DataFrame:

    _require_columns(
        risk_result,
        [
            "ISSUER_ID",
            "FINAL_SCORE",
            "FINAL_RANK",
            "RISK_SELECTED",
            "PORTFOLIO_WEIGHT",
        ],
        "risk_result",
    )

    portfolio = (
        risk_result.loc[
            risk_result[
                "RISK_SELECTED"
            ]
            .fillna(False)
            .astype(bool)
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
        .copy()
    )

    return portfolio


# =============================================================================
# AUDITORIA
# =============================================================================

def audit_risk(
    result: pd.DataFrame,
) -> dict:

    _require_dataframe(
        result,
        "risk_result",
    )

    risk_config = resolve_risk_config()

    selected = (
        result[
            "RISK_SELECTED"
        ]
        .fillna(False)
        .astype(bool)
    )

    concentration = (
        calculate_concentration_audit(
            result
        )
    )

    reason_counts = (
        result[
            "RISK_REASON"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    return {
        "status": "OK",
        "issuers_total":
            int(
                len(
                    result
                )
            ),
        "selected":
            int(
                selected.sum()
            ),
        "selection_rate":
            float(
                selected.mean()
            ),
        "max_positions":
            risk_config[
                "max_positions"
            ],
        "min_positions":
            risk_config[
                "min_positions"
            ],
        "min_positions_reached":
            bool(
                result[
                    "MIN_POSITIONS_REACHED"
                ].iloc[0]
            ),
        "max_sector_weight":
            risk_config[
                "max_sector_weight"
            ],
        "weighting_method":
            risk_config[
                "weighting_method"
            ],
        "reasons":
            reason_counts,
        "concentration":
            concentration,
        "future_return_used":
            False,
        "turnaround_research_used_by_risk":
            False,
        "study_score_from_ranking_accepted":
            bool(
                result.get(
                    "STUDY_SCORE_FROM_RANKING_ACCEPTED_RISK",
                    pd.Series(False, index=result.index),
                )
                .fillna(False)
                .astype(bool)
                .any()
            ),
        "return_optimization_performed":
            False,
    }


# =============================================================================
# SALVAR
# =============================================================================

def save_risk(
    result: pd.DataFrame,
    context: PITContext,
) -> dict:

    audit = audit_risk(
        result
    )

    portfolio = (
        get_selected_portfolio(
            result
        )
    )

    date_tag = (
        context
        .formation_date
        .strftime(
            "%Y%m%d"
        )
    )

    risk_path = (
        RISK_DIR
        / f"risk_{date_tag}.csv"
    )

    portfolio_path = (
        RISK_DIR
        / f"portfolio_{date_tag}.csv"
    )

    manifest_path = (
        RISK_DIR
        / f"risk_manifest_{date_tag}.json"
    )

    result.to_csv(
        risk_path,
        index=False,
        encoding="utf-8-sig",
    )

    portfolio.to_csv(
        portfolio_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "engine":
            "RISK_ENGINE",
        "version":
            "0.2.0",
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
        "market_cutoff":
            str(
                context
                .market_cutoff
                .date()
            ),
        **audit,
        "files": {
            "risk":
                str(
                    risk_path
                ),
            "portfolio":
                str(
                    portfolio_path
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
        "risk":
            risk_path,
        "portfolio":
            portfolio_path,
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
            ],
            "FINAL_SCORE": [
                0.90,
                0.80,
                0.70,
                0.60,
            ],
            "FINAL_RANK": [
                1,
                2,
                3,
                4,
            ],
            "RANKING_ELIGIBLE": [
                True,
                True,
                True,
                True,
            ],
            "FCA_SETOR_ATIVIDADE": [
                "INDUSTRIA",
                "INDUSTRIA",
                "ENERGIA",
                "LOGISTICA",
            ],
            "TURNAROUND_RESEARCH_USED_IN_RANKING": [
                True,
                True,
                True,
                True,
            ],
            "RANKING_MODEL": [
                "MB_50_ML_30_ROE_20_STUDY",
                "MB_50_ML_30_ROE_20_STUDY",
                "MB_50_ML_30_ROE_20_STUDY",
                "MB_50_ML_30_ROE_20_STUDY",
            ],
            "MARGEM_BRUTA_WEIGHT": [0.50, 0.50, 0.50, 0.50],
            "MARGEM_LIQUIDA_WEIGHT": [0.30, 0.30, 0.30, 0.30],
            "ROE_WEIGHT": [0.20, 0.20, 0.20, 0.20],
            "QUALITY_WEIGHT": [0.0, 0.0, 0.0, 0.0],
            "VALUATION_WEIGHT": [0.0, 0.0, 0.0, 0.0],
            "FUTURE_RETURN_USED_RANKING": [
                False,
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

        market_cutoff = pd.Timestamp(
            "2025-12-30"
        )

    result = run_risk_engine(
        sample,
        FakeContext(),
    )

    selected = (
        result.loc[
            result[
                "RISK_SELECTED"
            ]
        ]
    )

    if selected.empty:
        raise RiskError(
            "SELF-TEST: nenhuma empresa selecionada."
        )

    if (
        selected[
            "FINAL_RANK"
        ].min()
        != 1
    ):
        raise RiskError(
            "SELF-TEST: primeira empresa do ranking "
            "não foi preservada."
        )

    if not np.isclose(
        selected[
            "PORTFOLIO_WEIGHT"
        ].sum(),
        1.0,
    ):
        raise RiskError(
            "SELF-TEST: pesos não somam 100%."
        )

    if (
        result[
            "FUTURE_RETURN_USED_RISK"
        ]
        .fillna(False)
        .astype(bool)
        .any()
    ):
        raise RiskError(
            "SELF-TEST: retorno futuro utilizado."
        )

    if (
        result[
            "TURNAROUND_RESEARCH_USED_RISK"
        ]
        .fillna(False)
        .astype(bool)
        .any()
    ):
        raise RiskError(
            "SELF-TEST: Turnaround Research utilizado."
        )

    # Teste obrigatório:
    # metadado de auditoria True deve ser bloqueado.

    contaminated = sample.copy()

    contaminated[
        "FUTURE_RETURN_USED_RANKING"
    ] = True

    blocked = False

    try:
        run_risk_engine(
            contaminated,
            FakeContext(),
        )

    except RiskIntegrityError:
        blocked = True

    if not blocked:
        raise RiskError(
            "SELF-TEST: metadado de retorno futuro "
            "True não foi bloqueado."
        )

    # Modelo/pesos do estudo divergentes também devem ser bloqueados.

    contaminated = sample.copy()
    contaminated["MARGEM_BRUTA_WEIGHT"] = 0.40

    blocked = False

    try:
        run_risk_engine(
            contaminated,
            FakeContext(),
        )
    except RiskIntegrityError:
        blocked = True

    if not blocked:
        raise RiskError(
            "SELF-TEST: peso divergente do estudo não foi bloqueado."
        )

    # Variável real de retorno futuro também deve ser bloqueada.

    contaminated = sample.copy()

    contaminated[
        "FUTURE_RETURN"
    ] = 1.0

    blocked = False

    try:
        run_risk_engine(
            contaminated,
            FakeContext(),
        )

    except RiskIntegrityError:
        blocked = True

    if not blocked:
        raise RiskError(
            "SELF-TEST: variável FUTURE_RETURN "
            "não foi bloqueada."
        )

    return True


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("B3 STOCK SELECTOR — RISK ENGINE")
    print("=" * 72)

    _self_test()

    print("Self-test: OK")
    print("Ranking econômico: PRESERVADO")
    print("Número de posições: CONTROLADO")
    print("Concentração setorial: CONTROLÁVEL")
    print("Pesos: EQUAL WEIGHT")
    print("Retorno futuro: NÃO UTILIZADO")
    print("Score 50/30/20 do Ranking: ACEITO E PRESERVADO")
    print("Turnaround recalculado pelo Risk: NÃO")
    print("Otimização por retorno: NÃO")
    print("Seleção final: AUDITÁVEL")

    print("=" * 72)
