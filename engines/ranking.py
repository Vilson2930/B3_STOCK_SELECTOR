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
# - o ranking final representa exclusivamente o modelo fundamental do estudo;
# - score do estudo: 50% Margem Bruta + 30% Margem Líquida + 20% ROE;
# - todos os três fatores possuem direção LOW;
# - Quality e Valuation não recebem peso no score principal;
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
# MODELO DO ESTUDO
#
# Evidência utilizada pelo robô:
#
# Margem Bruta   50%
# Margem Líquida 30%
# ROE             20%
#
# Todos os fatores têm direção LOW: valores relativamente menores recebem
# maior score cross-sectional. Os pesos refletem a hipótese operacional
# definida a partir da força relativa encontrada no estudo.
#
# Quality e Valuation permanecem como camadas independentes do pipeline,
# mas NÃO entram no score principal deste robô.
# =============================================================================

STUDY_WEIGHTS = {
    "MARGEM_BRUTA": 0.50,
    "MARGEM_LIQUIDA": 0.30,
    "ROE": 0.20,
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
# PESOS DO ESTUDO
# =============================================================================

def resolve_engine_weights() -> dict:
    """
    Mantém o nome da API por compatibilidade com o restante do projeto.
    Retorna exclusivamente os pesos do modelo fundamental do estudo.
    """
    configured = RANKING.get(
        "study_weights",
        STUDY_WEIGHTS,
    )

    if configured is None:
        configured = STUDY_WEIGHTS

    if not isinstance(configured, dict):
        raise RankingIntegrityError(
            "FAIL-SAFE: study_weights deve ser dicionário."
        )

    required = set(STUDY_WEIGHTS)
    extra = set(configured) - required
    missing = required - set(configured)

    if extra or missing:
        raise RankingIntegrityError(
            "FAIL-SAFE: study_weights incompatível com o estudo. "
            f"Ausentes={sorted(missing)} Extras={sorted(extra)}"
        )

    weights = {
        factor: float(configured[factor])
        for factor in STUDY_WEIGHTS
    }

    for factor, weight in weights.items():
        if not np.isfinite(weight) or weight < 0:
            raise RankingIntegrityError(
                f"Peso inválido para {factor}: {weight}"
            )

    total = sum(weights.values())

    if not np.isclose(total, 1.0, atol=1e-12):
        raise RankingIntegrityError(
            "FAIL-SAFE: pesos do estudo devem somar exatamente 1. "
            f"Soma atual={total}"
        )

    # Trava contra reintrodução silenciosa do antigo 70/30.
    legacy = RANKING.get("engine_weights")
    if isinstance(legacy, dict):
        legacy_positive = {
            str(k).upper(): float(v)
            for k, v in legacy.items()
            if np.isfinite(float(v)) and float(v) > 0
        }
        if legacy_positive:
            raise RankingAuthorizationError(
                "FAIL-SAFE: engine_weights legado detectado em config.py. "
                "Remova/desative pesos de Quality/Valuation/Turnaround; "
                "o ranking deste robô usa somente study_weights."
            )

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

    result = df.copy()

    score = (
        pd.to_numeric(
            result["TURNAROUND_RESEARCH_SCORE"],
            errors="coerce",
        )
        if "TURNAROUND_RESEARCH_SCORE" in result.columns
        else pd.Series(np.nan, index=result.index, dtype=float)
    )

    complete = (
        pd.to_numeric(
            result["TURNAROUND_RESEARCH_SCORE_COMPLETE"],
            errors="coerce",
        )
        if "TURNAROUND_RESEARCH_SCORE_COMPLETE" in result.columns
        else score
    )

    study_valid = score.notna() & complete.notna()

    result["RANKING_ELIGIBLE"] = study_valid

    reason = pd.Series(
        "OK",
        index=result.index,
        dtype="object",
    )
    reason.loc[~study_valid] = "STUDY_SCORE_INVALID"

    result["RANKING_ELIGIBILITY_REASON"] = reason

    return result


# =============================================================================
# SCORE FINAL
# =============================================================================

def calculate_final_score(
    df: pd.DataFrame,
    weights: dict,
) -> pd.DataFrame:

    result = df.copy()
    result["FINAL_SCORE"] = np.nan

    eligible = (
        result["RANKING_ELIGIBLE"]
        .fillna(False)
        .astype(bool)
    )

    if not eligible.any():
        return result

    # O Turnaround Engine já calcula o score 50/30/20 com direção LOW.
    # Aqui não recalculamos percentis nem fatores: apenas promovemos
    # exatamente esse score para FINAL_SCORE.
    study_score = validate_score(
        result["TURNAROUND_RESEARCH_SCORE"],
        "TURNAROUND_RESEARCH_SCORE",
    )

    result.loc[eligible, "FINAL_SCORE"] = study_score.loc[eligible]

    result["FINAL_SCORE"] = (
        result["FINAL_SCORE"]
        .clip(lower=0.0, upper=1.0)
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
    ] = "0.2.0"

    result[
        "RANKING_MODEL"
    ] = "MB_50_ML_30_ROE_20_STUDY"

    result["MARGEM_BRUTA_WEIGHT"] = weights["MARGEM_BRUTA"]
    result["MARGEM_LIQUIDA_WEIGHT"] = weights["MARGEM_LIQUIDA"]
    result["ROE_WEIGHT"] = weights["ROE"]
    result["QUALITY_WEIGHT"] = 0.0
    result["VALUATION_WEIGHT"] = 0.0
    result["TURNAROUND_WEIGHT"] = 1.0

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
    ] = True

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
            "MB_50_ML_30_ROE_20_STUDY",

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
            True,

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
            "ISSUER_ID": ["1", "2", "3"],
            "QUALITY_SCORE": [0.10, 0.90, 0.50],
            "QUALITY_SCORE_VALID": [False, True, True],
        }
    )

    valuation = pd.DataFrame(
        {
            "ISSUER_ID": ["1", "2", "3"],
            "VALUATION_SCORE": [np.nan, 0.10, 0.90],
            "VALUATION_SCORE_VALID": [False, True, True],
        }
    )

    turnaround = pd.DataFrame(
        {
            "ISSUER_ID": ["1", "2", "3"],
            "TURNAROUND_RESEARCH_SCORE": [0.90, 0.60, 0.20],
            "TURNAROUND_RESEARCH_SCORE_COMPLETE": [0.90, 0.60, 0.20],
            "TURNAROUND_PRODUCTION_AUTHORIZED": [False, False, False],
            "TURNAROUND_SCORE_MODEL": [
                "MB_50_ML_30_ROE_20_RESEARCH",
                "MB_50_ML_30_ROE_20_RESEARCH",
                "MB_50_ML_30_ROE_20_RESEARCH",
            ],
        }
    )

    class FakeContext:
        formation_date = pd.Timestamp("2025-12-31")
        accounting_cutoff = pd.Timestamp("2025-09-30")

    result = run_ranking_engine(
        quality=quality,
        valuation=valuation,
        turnaround=turnaround,
        context=FakeContext(),
    )

    indexed = result.set_index("ISSUER_ID")

    if not np.isclose(indexed.loc["1", "FINAL_SCORE"], 0.90):
        raise RankingError(
            "SELF-TEST: score do estudo não foi preservado."
        )

    if int(indexed.loc["1", "FINAL_RANK"]) != 1:
        raise RankingError(
            "SELF-TEST: ranking não respeitou o score 50/30/20."
        )

    # Quality/Valuation inválidos não podem bloquear uma empresa que possua
    # score completo do estudo, pois não pertencem ao score principal.
    if not bool(indexed.loc["1", "RANKING_ELIGIBLE"]):
        raise RankingError(
            "SELF-TEST: Quality/Valuation bloquearam indevidamente o estudo."
        )

    if not bool(indexed.loc["1", "TURNAROUND_RESEARCH_USED_IN_RANKING"]):
        raise RankingError(
            "SELF-TEST: score do estudo não foi marcado como utilizado."
        )

    if (
        result["FUTURE_RETURN_USED_RANKING"]
        .fillna(False)
        .astype(bool)
        .any()
    ):
        raise RankingError(
            "SELF-TEST: retorno futuro utilizado."
        )

    contaminated = quality.copy()
    contaminated["FUTURE_RETURN"] = [1.0, 2.0, 3.0]

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
    print("Modelo principal: ESTUDO FUNDAMENTAL 50/30/20")
    print("Quality no score principal: 0%")
    print("Valuation no score principal: 0%")
    print("Retorno futuro: PROIBIDO")
    print(
        "Pesos atuais: "
        f"{weights['MARGEM_BRUTA']:.0%} Margem Bruta / "
        f"{weights['MARGEM_LIQUIDA']:.0%} Margem Líquida / "
        f"{weights['ROE']:.0%} ROE"
    )
    print("Direção dos três fatores: LOW")
    print("Missing score: NÃO RECEBE IMPUTAÇÃO FAVORÁVEL")
    print("Ranking: DETERMINÍSTICO")
    print("Auditoria: ATIVA")

    print("=" * 72)
