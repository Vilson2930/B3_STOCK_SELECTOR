# =============================================================================
# B3 STOCK SELECTOR
# Arquivo: main.py
#
# ORQUESTRADOR PRINCIPAL
#
# PIPELINE:
#
# CVM + MARKET
#      ↓
# PIT
#      ↓
# UNIVERSE
#      ↓
# INVESTABILITY
#      ↓
# FUNDAMENTALS
#      ↓
# ┌─────────────┬──────────────┬───────────────┐
# │   QUALITY   │  TURNAROUND  │   VALUATION   │
# └─────────────┴──────────────┴───────────────┘
#                         ↓
#                      RANKING
#                         ↓
#                        RISK
#                         ↓
#                       REPORT
#
# REGRAS:
# - sem look-ahead;
# - retorno futuro proibido;
# - Turnaround Research não entra no ranking;
# - uma empresa = um emissor na análise fundamental;
# - execução determinística;
# - fail-safe;
# - auditoria em todas as etapas.
# =============================================================================

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

try:
    import config as project_config

    from config import (
        OUTPUT_DIR,
        PROJECT_NAME,
        VERSION,
        validate_config,
    )

except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar config.py."
    ) from exc


# =============================================================================
# PIT
# =============================================================================

try:
    from data.pit import (
        PITContext,
        create_pit_context,
        save_pit_manifest,
    )

except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar data/pit.py."
    ) from exc


# =============================================================================
# MARKET
# =============================================================================

try:
    import data.market as market_module

except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar data/market.py."
    ) from exc


# =============================================================================
# UNIVERSE
# =============================================================================

try:
    import engines.universe as universe_module

except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar engines/universe.py."
    ) from exc


# =============================================================================
# INVESTABILITY
# =============================================================================

try:
    import engines.investability as investability_module

except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar engines/investability.py."
    ) from exc


# =============================================================================
# FUNDAMENTALS
# =============================================================================

try:
    import engines.fundamentals as fundamentals_module

except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar engines/fundamentals.py."
    ) from exc


# =============================================================================
# QUALITY
# =============================================================================

try:
    import engines.quality as quality_module

except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar engines/quality.py."
    ) from exc


# =============================================================================
# TURNAROUND
# =============================================================================

try:
    import engines.turnaround as turnaround_module

except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar engines/turnaround.py."
    ) from exc


# =============================================================================
# VALUATION
# =============================================================================

try:
    import engines.valuation as valuation_module

except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar engines/valuation.py."
    ) from exc


# =============================================================================
# RANKING
# =============================================================================

try:
    import engines.ranking as ranking_module

except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar engines/ranking.py."
    ) from exc


# =============================================================================
# RISK
# =============================================================================

try:
    import engines.risk as risk_module

except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar engines/risk.py."
    ) from exc


# =============================================================================
# REPORT
# =============================================================================

try:
    import reports.report as report_module

except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar reports/report.py."
    ) from exc


# =============================================================================
# DIRETÓRIOS
# =============================================================================

RUN_DIR = (
    OUTPUT_DIR
    / "runs"
)

RUN_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# EXCEÇÕES
# =============================================================================

class PipelineError(RuntimeError):
    """Erro geral do pipeline."""


class PipelineIntegrityError(PipelineError):
    """Erro de integridade."""


class PipelineStageError(PipelineError):
    """Falha em uma etapa específica."""


# =============================================================================
# COLUNAS PROIBIDAS
# =============================================================================

FORBIDDEN_FUTURE_TERMS = (
    "FUTURE_RETURN",
    "RETORNO_FUTURO",
    "RETURN_FUTURE",
    "FUTURE_WINNER",
    "WINNER_LABEL",
    "TARGET_RETURN",
)

ALLOWED_FUTURE_METADATA = {
    "FUTURE_RETURN_USED",
    "FUTURE_RETURN_USED_QUALITY",
    "FUTURE_RETURN_USED_TURNAROUND",
    "FUTURE_RETURN_USED_VALUATION",
    "FUTURE_RETURN_USED_RANKING",
    "FUTURE_RETURN_USED_RISK",
}


# =============================================================================
# UTILIDADES
# =============================================================================

def utc_now() -> datetime:

    return datetime.now(
        timezone.utc
    )


def utc_now_iso() -> str:

    return utc_now().isoformat()


def date_tag(
    value,
) -> str:

    return (
        pd.Timestamp(
            value
        )
        .strftime(
            "%Y%m%d"
        )
    )


def require_dataframe(
    df: pd.DataFrame,
    name: str,
) -> None:

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise PipelineIntegrityError(
            f"{name} não é DataFrame."
        )

    if df.empty:

        raise PipelineIntegrityError(
            f"FAIL-SAFE: {name} está vazio."
        )


# =============================================================================
# PROTEÇÃO CONTRA RETORNO FUTURO
# =============================================================================

def assert_no_future_information(
    df: pd.DataFrame,
    stage: str,
) -> None:

    require_dataframe(
        df,
        stage,
    )

    forbidden = []

    for column in df.columns:

        normalized = (
            str(column)
            .upper()
        )

        if normalized in ALLOWED_FUTURE_METADATA:
            continue

        if any(
            term in normalized
            for term in FORBIDDEN_FUTURE_TERMS
        ):

            forbidden.append(
                column
            )

    if forbidden:

        raise PipelineIntegrityError(
            "FAIL-SAFE: informação futura detectada "
            f"na etapa {stage}: {forbidden}"
        )


# =============================================================================
# RESOLUÇÃO DE FUNÇÕES
#
# Permite compatibilidade entre versões dos módulos sem duplicar lógica.
# =============================================================================

def resolve_callable(
    module,
    names: tuple[str, ...],
    *,
    required: bool = True,
) -> Optional[Callable]:

    for name in names:

        candidate = getattr(
            module,
            name,
            None,
        )

        if callable(
            candidate
        ):

            return candidate

    if required:

        raise PipelineIntegrityError(
            "FAIL-SAFE: nenhuma função compatível encontrada "
            f"em {module.__name__}. "
            f"Esperado uma entre: {names}"
        )

    return None


# =============================================================================
# CHAMADA COMPATÍVEL
# =============================================================================

def call_stage(
    stage_name: str,
    function: Callable,
    attempts: list[tuple[tuple, dict]],
):
    """
    Tenta assinaturas conhecidas do próprio pipeline.

    TypeError de assinatura permite tentar a próxima forma.
    Qualquer outro erro interrompe imediatamente.
    """

    last_type_error = None

    for args, kwargs in attempts:

        try:

            return function(
                *args,
                **kwargs,
            )

        except TypeError as exc:

            last_type_error = exc

            continue

        except Exception as exc:

            raise PipelineStageError(
                f"Falha na etapa {stage_name}: {exc}"
            ) from exc

    raise PipelineStageError(
        f"Não foi possível executar {stage_name} "
        "com nenhuma assinatura compatível."
    ) from last_type_error


# =============================================================================
# MANIFESTO DA EXECUÇÃO
# =============================================================================

def save_run_manifest(
    manifest: dict,
    formation_date,
) -> Path:

    path = (
        RUN_DIR
        / (
            "run_manifest_"
            f"{date_tag(formation_date)}.json"
        )
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    return path


# =============================================================================
# CONTEXTO PIT
# =============================================================================

def build_context(
    formation_date,
    accounting_cutoff,
    market_cutoff=None,
) -> PITContext:

    if market_cutoff is None:

        market_cutoff = formation_date

    context = create_pit_context(
        formation_date=formation_date,
        accounting_cutoff=accounting_cutoff,
        market_cutoff=market_cutoff,
    )

    return context


# =============================================================================
# MARKET
# =============================================================================

def run_market_stage(
    context: PITContext,
) -> pd.DataFrame:

    function = resolve_callable(
        market_module,
        (
            "build_market_universe",
        ),
    )

    years = list(
        range(
            context.market_cutoff.year,
            context.market_cutoff.year + 1,
        )
    )

    result = call_stage(
        "MARKET",
        function,
        [
            (
                (),
                {
                    "context": context,
                },
            ),
            (
                (context,),
                {},
            ),
            (
                (),
                {
                    "years": years,
                    "context": context,
                },
            ),
            (
                (years, context),
                {},
            ),
        ],
    )

    require_dataframe(
        result,
        "MARKET",
    )

    assert_no_future_information(
        result,
        "MARKET",
    )

    return result


# =============================================================================
# IDENTITY MAP
#
# O Universe Engine exige a ponte ticker → emissor.
#
# O main não inventa essa identidade.
# Ela deve ser fornecida por uma fonte válida do pipeline.
# =============================================================================

def resolve_identity_map(
    identity_map: Optional[pd.DataFrame],
) -> pd.DataFrame:

    if identity_map is None:

        raise PipelineIntegrityError(
            "FAIL-SAFE: identity_map ausente. "
            "O pipeline não pode inventar a relação "
            "ticker → emissor."
        )

    require_dataframe(
        identity_map,
        "IDENTITY_MAP",
    )

    assert_no_future_information(
        identity_map,
        "IDENTITY_MAP",
    )

    return identity_map.copy()


# =============================================================================
# UNIVERSE
# =============================================================================

def run_universe_stage(
    market_universe: pd.DataFrame,
    identity_map: pd.DataFrame,
    context: PITContext,
) -> dict:

    function = resolve_callable(
        universe_module,
        (
            "build_universe",
        ),
    )

    result = call_stage(
        "UNIVERSE",
        function,
        [
            (
                (),
                {
                    "market_universe": market_universe,
                    "identity_map": identity_map,
                    "context": context,
                },
            ),
            (
                (
                    market_universe,
                    identity_map,
                    context,
                ),
                {},
            ),
            (
                (
                    market_universe,
                    identity_map,
                ),
                {
                    "context": context,
                },
            ),
        ],
    )

    if isinstance(
        result,
        dict,
    ):

        security_universe = (
            result.get(
                "security_universe"
            )
            or
            result.get(
                "securities"
            )
        )

        issuer_universe = (
            result.get(
                "issuer_universe"
            )
            or
            result.get(
                "issuers"
            )
        )

    elif (
        isinstance(
            result,
            tuple,
        )
        and
        len(result) == 2
    ):

        security_universe = result[0]
        issuer_universe = result[1]

    else:

        raise PipelineIntegrityError(
            "FAIL-SAFE: build_universe deve retornar "
            "dict ou tuple com universos de security e issuer."
        )

    require_dataframe(
        security_universe,
        "SECURITY_UNIVERSE",
    )

    require_dataframe(
        issuer_universe,
        "ISSUER_UNIVERSE",
    )

    assert_no_future_information(
        security_universe,
        "SECURITY_UNIVERSE",
    )

    assert_no_future_information(
        issuer_universe,
        "ISSUER_UNIVERSE",
    )

    return {
        "security_universe":
            security_universe,

        "issuer_universe":
            issuer_universe,
    }


# =============================================================================
# INVESTABILITY
# =============================================================================

def run_investability_stage(
    issuer_universe: pd.DataFrame,
    context: PITContext,
    safety_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:

    function = resolve_callable(
        investability_module,
        (
            "run_investability_engine",
            "build_investability",
        ),
    )

    attempts = [
        (
            (),
            {
                "issuer_universe": issuer_universe,
                "safety_df": safety_data,
                "context": context,
            },
        ),
        (
            (),
            {
                "issuer_universe": issuer_universe,
                "safety_data": safety_data,
                "context": context,
            },
        ),
        (
            (
                issuer_universe,
                safety_data,
                context,
            ),
            {},
        ),
        (
            (
                issuer_universe,
            ),
            {
                "context": context,
            },
        ),
        (
            (
                issuer_universe,
            ),
            {},
        ),
    ]

    result = call_stage(
        "INVESTABILITY",
        function,
        attempts,
    )

    require_dataframe(
        result,
        "INVESTABILITY",
    )

    assert_no_future_information(
        result,
        "INVESTABILITY",
    )

    return result


# =============================================================================
# FUNDAMENTALS
# =============================================================================

def run_fundamentals_stage(
    investability: pd.DataFrame,
    context: PITContext,
    accounting_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:

    function = resolve_callable(
        fundamentals_module,
        (
            "run_fundamentals_engine",
            "build_fundamentals",
        ),
    )

    attempts = [
        (
            (),
            {
                "issuer_universe": investability,
                "accounting_data": accounting_data,
                "context": context,
            },
        ),
        (
            (),
            {
                "investability": investability,
                "accounting_data": accounting_data,
                "context": context,
            },
        ),
        (
            (
                investability,
                accounting_data,
                context,
            ),
            {},
        ),
        (
            (
                investability,
                accounting_data,
            ),
            {
                "context": context,
            },
        ),
        (
            (
                investability,
            ),
            {
                "context": context,
            },
        ),
    ]

    result = call_stage(
        "FUNDAMENTALS",
        function,
        attempts,
    )

    require_dataframe(
        result,
        "FUNDAMENTALS",
    )

    assert_no_future_information(
        result,
        "FUNDAMENTALS",
    )

    return result


# =============================================================================
# QUALITY
# =============================================================================

def run_quality_stage(
    fundamentals: pd.DataFrame,
    context: PITContext,
) -> pd.DataFrame:

    function = resolve_callable(
        quality_module,
        (
            "run_quality_engine",
        ),
    )

    result = call_stage(
        "QUALITY",
        function,
        [
            (
                (),
                {
                    "fundamentals": fundamentals,
                    "context": context,
                },
            ),
            (
                (
                    fundamentals,
                    context,
                ),
                {},
            ),
        ],
    )

    require_dataframe(
        result,
        "QUALITY",
    )

    assert_no_future_information(
        result,
        "QUALITY",
    )

    return result


# =============================================================================
# TURNAROUND
# =============================================================================

def run_turnaround_stage(
    fundamentals: pd.DataFrame,
    context: PITContext,
) -> pd.DataFrame:

    function = resolve_callable(
        turnaround_module,
        (
            "run_turnaround_engine",
        ),
    )

    result = call_stage(
        "TURNAROUND",
        function,
        [
            (
                (),
                {
                    "fundamentals": fundamentals,
                    "context": context,
                },
            ),
            (
                (
                    fundamentals,
                    context,
                ),
                {},
            ),
        ],
    )

    require_dataframe(
        result,
        "TURNAROUND",
    )

    assert_no_future_information(
        result,
        "TURNAROUND",
    )

    return result


# =============================================================================
# VALUATION
# =============================================================================

def run_valuation_stage(
    fundamentals: pd.DataFrame,
    context: PITContext,
) -> pd.DataFrame:

    function = resolve_callable(
        valuation_module,
        (
            "run_valuation_engine",
        ),
    )

    result = call_stage(
        "VALUATION",
        function,
        [
            (
                (),
                {
                    "fundamentals": fundamentals,
                    "context": context,
                },
            ),
            (
                (
                    fundamentals,
                    context,
                ),
                {},
            ),
        ],
    )

    require_dataframe(
        result,
        "VALUATION",
    )

    assert_no_future_information(
        result,
        "VALUATION",
    )

    return result


# =============================================================================
# RANKING
# =============================================================================

def run_ranking_stage(
    quality: pd.DataFrame,
    valuation: pd.DataFrame,
    turnaround: pd.DataFrame,
    context: PITContext,
) -> pd.DataFrame:

    result = ranking_module.run_ranking_engine(
        quality=quality,
        valuation=valuation,
        turnaround=turnaround,
        context=context,
    )

    require_dataframe(
        result,
        "RANKING",
    )

    assert_no_future_information(
        result,
        "RANKING",
    )

    return result


# =============================================================================
# RISK
# =============================================================================

def run_risk_stage(
    ranking: pd.DataFrame,
    context: PITContext,
) -> pd.DataFrame:

    result = risk_module.run_risk_engine(
        ranking=ranking,
        context=context,
    )

    require_dataframe(
        result,
        "RISK",
    )

    assert_no_future_information(
        result,
        "RISK",
    )

    return result


# =============================================================================
# REPORT
# =============================================================================

def run_report_stage(
    risk_result: pd.DataFrame,
    context: PITContext,
) -> dict:

    result = report_module.save_report(
        risk_result=risk_result,
        context=context,
    )

    if not isinstance(
        result,
        dict,
    ):

        raise PipelineIntegrityError(
            "FAIL-SAFE: Report Engine não retornou dict."
        )

    return result


# =============================================================================
# SALVAMENTO OPCIONAL DOS MOTORES
# =============================================================================

def save_stage_if_supported(
    module,
    function_names: tuple[str, ...],
    result: pd.DataFrame,
    context: PITContext,
) -> Optional[Any]:

    function = resolve_callable(
        module,
        function_names,
        required=False,
    )

    if function is None:
        return None

    try:

        return function(
            result,
            context,
        )

    except TypeError:

        try:

            return function(
                result=result,
                context=context,
            )

        except TypeError:

            return None


# =============================================================================
# AUDITORIA GLOBAL
# =============================================================================

def build_global_audit(
    *,
    context: PITContext,
    security_universe: pd.DataFrame,
    issuer_universe: pd.DataFrame,
    investability: pd.DataFrame,
    fundamentals: pd.DataFrame,
    quality: pd.DataFrame,
    turnaround: pd.DataFrame,
    valuation: pd.DataFrame,
    ranking: pd.DataFrame,
    risk: pd.DataFrame,
) -> dict:

    selected = (
        risk[
            "RISK_SELECTED"
        ]
        .fillna(False)
        .astype(bool)
    )

    ranking_eligible = (
        ranking[
            "RANKING_ELIGIBLE"
        ]
        .fillna(False)
        .astype(bool)
    )

    turnaround_used = False

    if (
        "TURNAROUND_RESEARCH_USED_IN_RANKING"
        in ranking.columns
    ):

        turnaround_used = bool(
            ranking[
                "TURNAROUND_RESEARCH_USED_IN_RANKING"
            ]
            .fillna(False)
            .astype(bool)
            .any()
        )

    return {
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

        "security_universe":
            int(
                len(
                    security_universe
                )
            ),

        "issuer_universe":
            int(
                len(
                    issuer_universe
                )
            ),

        "investability_rows":
            int(
                len(
                    investability
                )
            ),

        "fundamental_rows":
            int(
                len(
                    fundamentals
                )
            ),

        "quality_rows":
            int(
                len(
                    quality
                )
            ),

        "turnaround_rows":
            int(
                len(
                    turnaround
                )
            ),

        "valuation_rows":
            int(
                len(
                    valuation
                )
            ),

        "ranking_rows":
            int(
                len(
                    ranking
                )
            ),

        "ranking_eligible":
            int(
                ranking_eligible.sum()
            ),

        "selected_positions":
            int(
                selected.sum()
            ),

        "portfolio_weight":
            float(
                risk.loc[
                    selected,
                    "PORTFOLIO_WEIGHT",
                ]
                .sum()
            ),

        "future_return_used":
            False,

        "turnaround_research_used_in_ranking":
            turnaround_used,
    }


# =============================================================================
# PIPELINE COMPLETO
# =============================================================================

def run_pipeline(
    *,
    formation_date,
    accounting_cutoff,
    market_cutoff=None,
    identity_map: pd.DataFrame,
    accounting_data: Optional[pd.DataFrame] = None,
    safety_data: Optional[pd.DataFrame] = None,
) -> dict:

    started_at = utc_now()

    # -------------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------------

    validate_config()

    # -------------------------------------------------------------------------
    # PIT
    # -------------------------------------------------------------------------

    context = build_context(
        formation_date=formation_date,
        accounting_cutoff=accounting_cutoff,
        market_cutoff=market_cutoff,
    )

    try:
        save_pit_manifest(
            context
        )
    except TypeError:
        pass

    # -------------------------------------------------------------------------
    # INPUT IDENTITY
    # -------------------------------------------------------------------------

    identity = resolve_identity_map(
        identity_map
    )

    # -------------------------------------------------------------------------
    # MARKET
    # -------------------------------------------------------------------------

    market_universe = run_market_stage(
        context
    )

    # -------------------------------------------------------------------------
    # UNIVERSE
    # -------------------------------------------------------------------------

    universes = run_universe_stage(
        market_universe=market_universe,
        identity_map=identity,
        context=context,
    )

    security_universe = (
        universes[
            "security_universe"
        ]
    )

    issuer_universe = (
        universes[
            "issuer_universe"
        ]
    )

    # -------------------------------------------------------------------------
    # INVESTABILITY
    # -------------------------------------------------------------------------

    investability = (
        run_investability_stage(
            issuer_universe=issuer_universe,
            context=context,
            safety_data=safety_data,
        )
    )

    # -------------------------------------------------------------------------
    # FUNDAMENTALS
    # -------------------------------------------------------------------------

    fundamentals = (
        run_fundamentals_stage(
            investability=investability,
            context=context,
            accounting_data=accounting_data,
        )
    )

    # -------------------------------------------------------------------------
    # QUALITY
    # -------------------------------------------------------------------------

    quality = run_quality_stage(
        fundamentals,
        context,
    )

    # -------------------------------------------------------------------------
    # TURNAROUND
    #
    # Executa para pesquisa/auditoria.
    # NÃO significa autorização no ranking.
    # -------------------------------------------------------------------------

    turnaround = (
        run_turnaround_stage(
            fundamentals,
            context,
        )
    )

    # -------------------------------------------------------------------------
    # VALUATION
    # -------------------------------------------------------------------------

    valuation = (
        run_valuation_stage(
            fundamentals,
            context,
        )
    )

    # -------------------------------------------------------------------------
    # RANKING
    # -------------------------------------------------------------------------

    ranking = run_ranking_stage(
        quality=quality,
        valuation=valuation,
        turnaround=turnaround,
        context=context,
    )

    # -------------------------------------------------------------------------
    # RISK
    # -------------------------------------------------------------------------

    risk = run_risk_stage(
        ranking,
        context,
    )

    # -------------------------------------------------------------------------
    # AUDITORIA GLOBAL ANTES DO REPORT
    # -------------------------------------------------------------------------

    audit = build_global_audit(
        context=context,
        security_universe=security_universe,
        issuer_universe=issuer_universe,
        investability=investability,
        fundamentals=fundamentals,
        quality=quality,
        turnaround=turnaround,
        valuation=valuation,
        ranking=ranking,
        risk=risk,
    )

    if audit[
        "future_return_used"
    ]:

        raise PipelineIntegrityError(
            "FAIL-SAFE: retorno futuro utilizado."
        )

    if audit[
        "turnaround_research_used_in_ranking"
    ]:

        raise PipelineIntegrityError(
            "FAIL-SAFE: Turnaround Research "
            "entrou no ranking."
        )

    # -------------------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------------------

    report_files = run_report_stage(
        risk,
        context,
    )

    # -------------------------------------------------------------------------
    # SALVA OUTPUTS DOS MOTORES
    # -------------------------------------------------------------------------

    stage_files = {}

    stage_files[
        "universe"
    ] = save_stage_if_supported(
        universe_module,
        (
            "save_universe",
        ),
        issuer_universe,
        context,
    )

    stage_files[
        "investability"
    ] = save_stage_if_supported(
        investability_module,
        (
            "save_investability",
        ),
        investability,
        context,
    )

    stage_files[
        "fundamentals"
    ] = save_stage_if_supported(
        fundamentals_module,
        (
            "save_fundamentals",
        ),
        fundamentals,
        context,
    )

    stage_files[
        "quality"
    ] = save_stage_if_supported(
        quality_module,
        (
            "save_quality",
        ),
        quality,
        context,
    )

    stage_files[
        "turnaround"
    ] = save_stage_if_supported(
        turnaround_module,
        (
            "save_turnaround",
        ),
        turnaround,
        context,
    )

    stage_files[
        "valuation"
    ] = save_stage_if_supported(
        valuation_module,
        (
            "save_valuation",
        ),
        valuation,
        context,
    )

    stage_files[
        "ranking"
    ] = save_stage_if_supported(
        ranking_module,
        (
            "save_ranking",
        ),
        ranking,
        context,
    )

    stage_files[
        "risk"
    ] = save_stage_if_supported(
        risk_module,
        (
            "save_risk",
        ),
        risk,
        context,
    )

    # -------------------------------------------------------------------------
    # MANIFESTO FINAL
    # -------------------------------------------------------------------------

    finished_at = utc_now()

    manifest = {
        "project":
            PROJECT_NAME,

        "version":
            VERSION,

        "pipeline_version":
            "0.1.0",

        "status":
            "SUCCESS",

        "started_at_utc":
            started_at.isoformat(),

        "finished_at_utc":
            finished_at.isoformat(),

        "duration_seconds":
            float(
                (
                    finished_at
                    -
                    started_at
                ).total_seconds()
            ),

        "audit":
            audit,

        "report_files": {
            key:
                str(value)

            for key, value
            in report_files.items()
        },

        "stage_files": {
            key:
                str(value)

            for key, value
            in stage_files.items()

            if value is not None
        },
    }

    manifest_path = save_run_manifest(
        manifest,
        context.formation_date,
    )

    # -------------------------------------------------------------------------
    # RESULTADO
    # -------------------------------------------------------------------------

    return {
        "context":
            context,

        "market_universe":
            market_universe,

        "security_universe":
            security_universe,

        "issuer_universe":
            issuer_universe,

        "investability":
            investability,

        "fundamentals":
            fundamentals,

        "quality":
            quality,

        "turnaround":
            turnaround,

        "valuation":
            valuation,

        "ranking":
            ranking,

        "risk":
            risk,

        "audit":
            audit,

        "report_files":
            report_files,

        "manifest":
            manifest_path,
    }


# =============================================================================
# FAIL-SAFE DE EXECUÇÃO
# =============================================================================

def run_pipeline_safe(
    **kwargs,
) -> dict:

    formation_date = kwargs.get(
        "formation_date"
    )

    try:

        return run_pipeline(
            **kwargs
        )

    except Exception as exc:

        failure = {
            "project":
                PROJECT_NAME,

            "version":
                VERSION,

            "pipeline_version":
                "0.1.0",

            "status":
                "FAILED",

            "created_at_utc":
                utc_now_iso(),

            "error_type":
                type(
                    exc
                ).__name__,

            "error":
                str(
                    exc
                ),

            "traceback":
                traceback.format_exc(),

            "future_return_used":
                False,
        }

        if formation_date is not None:

            try:

                failure_path = (
                    RUN_DIR
                    / (
                        "run_failure_"
                        f"{date_tag(formation_date)}.json"
                    )
                )

                with failure_path.open(
                    "w",
                    encoding="utf-8",
                ) as file:

                    json.dump(
                        failure,
                        file,
                        indent=2,
                        ensure_ascii=False,
                    )

            except Exception:
                pass

        raise


# =============================================================================
# RESUMO DE CONSOLE
# =============================================================================

def print_run_summary(
    result: dict,
) -> None:

    audit = (
        result[
            "audit"
        ]
    )

    print()
    print("=" * 78)
    print(PROJECT_NAME)
    print("=" * 78)

    print(
        "Formation date:",
        audit[
            "formation_date"
        ],
    )

    print(
        "Accounting cutoff:",
        audit[
            "accounting_cutoff"
        ],
    )

    print(
        "Market cutoff:",
        audit[
            "market_cutoff"
        ],
    )

    print("-" * 78)

    print(
        "Security universe:",
        audit[
            "security_universe"
        ],
    )

    print(
        "Issuer universe:",
        audit[
            "issuer_universe"
        ],
    )

    print(
        "Ranking eligible:",
        audit[
            "ranking_eligible"
        ],
    )

    print(
        "Selected positions:",
        audit[
            "selected_positions"
        ],
    )

    print(
        "Portfolio weight:",
        f"{audit['portfolio_weight']:.4f}",
    )

    print("-" * 78)

    print(
        "Future return used:",
        audit[
            "future_return_used"
        ],
    )

    print(
        "Turnaround Research used in ranking:",
        audit[
            "turnaround_research_used_in_ranking"
        ],
    )

    print("-" * 78)

    print(
        "Report HTML:",
        result[
            "report_files"
        ].get(
            "html"
        ),
    )

    print(
        "Run manifest:",
        result[
            "manifest"
        ],
    )

    print("=" * 78)
    print("PIPELINE FINALIZADO COM SUCESSO")
    print("=" * 78)


# =============================================================================
# EXECUÇÃO DIRETA
#
# IMPORTANTE:
# main.py não inventa datas nem arquivos.
#
# Para execução real, chamar run_pipeline_safe(...) fornecendo:
#
# - formation_date
# - accounting_cutoff
# - market_cutoff
# - identity_map
# - accounting_data
# - safety_data, quando aplicável
#
# Isso evita que uma execução acidental use datas ou bases erradas.
# =============================================================================

if __name__ == "__main__":

    print("=" * 78)
    print(f"{PROJECT_NAME} — MAIN")
    print("=" * 78)

    print(
        "Orquestrador carregado com sucesso."
    )

    print()
    print(
        "Execução automática sem parâmetros foi bloqueada "
        "por segurança."
    )

    print()
    print(
        "Use run_pipeline_safe(...) com contexto PIT "
        "e bases explicitamente definidas."
    )

    print()
    print(
        "Pipeline:"
    )

    print(
        "PIT → MARKET → UNIVERSE → INVESTABILITY → FUNDAMENTALS "
        "→ QUALITY / TURNAROUND / VALUATION → RANKING → RISK → REPORT"
    )

    print()
    print(
        "Turnaround Research: BLOQUEADO NO RANKING"
    )

    print(
        "Retorno futuro: PROIBIDO"
    )

    print(
        "Look-ahead: PROIBIDO"
    )

    print("=" * 78)
