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
# SHARE CLASSES (FRE) + MARKET CAP
#      ↓
# FUNDAMENTALS
#      ↓
# INVESTABILITY
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
# - somente o modelo do estudo 50/30/20 autorizado pode entrar no ranking;
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
except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar config.py."
    ) from exc

if not hasattr(project_config, "OUTPUT_DIR"):
    raise RuntimeError(
        "FAIL-SAFE: OUTPUT_DIR ausente em config.py."
    )

OUTPUT_DIR = project_config.OUTPUT_DIR

PROJECT_NAME = getattr(
    project_config,
    "PROJECT_NAME",
    getattr(project_config, "PROJECT", "B3_STOCK_SELECTOR"),
)

VERSION = getattr(
    project_config,
    "VERSION",
    getattr(project_config, "PROJECT_VERSION", "0.1.0"),
)

validate_config = getattr(
    project_config,
    "validate_config",
    None,
)

if not callable(validate_config):
    raise RuntimeError(
        "FAIL-SAFE: validate_config() ausente em config.py."
    )


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
# ACCOUNTING
# =============================================================================

try:
    import data.accounting as accounting_module

except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar data/accounting.py."
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
# SHARE CLASSES / MARKET CAP
# =============================================================================

def run_market_cap_stage(
    security_universe: pd.DataFrame,
    accounting_data: pd.DataFrame,
    context: PITContext,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Constrói a ponte:
        FRE por classe -> preço B3 por classe -> MARKET_CAP por emissor.

    Retorna:
        accounting_enriched
        share_classes
        issuer_market_cap

    Regras:
    - não altera o security_universe;
    - não usa QUATOT como quantidade de ações;
    - não escolhe preço arbitrário;
    - não cria MARKET_CAP quando a classe não pode ser casada com segurança;
    - mantém uma linha contábil por ISSUER_ID.
    """

    build_share_classes = resolve_callable(
        accounting_module,
        (
            "build_share_class_data",
        ),
    )

    calculate_market_cap = resolve_callable(
        market_module,
        (
            "calculate_issuer_market_cap",
        ),
    )

    share_classes = call_stage(
        "SHARE_CLASSES_FRE",
        build_share_classes,
        [
            (
                (),
                {
                    "formation_date": context.formation_date,
                    "accounting_cutoff": context.accounting_cutoff,
                },
            ),
            (
                (
                    context.formation_date,
                    context.accounting_cutoff,
                ),
                {},
            ),
        ],
    )

    require_dataframe(
        share_classes,
        "SHARE_CLASSES_FRE",
    )

    assert_no_future_information(
        share_classes,
        "SHARE_CLASSES_FRE",
    )

    # accounting.py usa SHARE_CLASS.
    # market.py usa SECURITY_CLASS.
    # A conversão é explícita e local; nenhuma classe é inferida aqui.
    if (
        "SHARE_CLASS" in share_classes.columns
        and "SECURITY_CLASS" not in share_classes.columns
    ):
        share_classes_for_market = (
            share_classes.rename(
                columns={
                    "SHARE_CLASS": "SECURITY_CLASS",
                }
            )
        )
    else:
        share_classes_for_market = (
            share_classes.copy()
        )

    required_share_columns = {
        "ISSUER_ID",
        "SECURITY_CLASS",
        "SHARES_CLASS_OUTSTANDING",
    }

    missing_share_columns = (
        required_share_columns
        - set(share_classes_for_market.columns)
    )

    if missing_share_columns:
        raise PipelineIntegrityError(
            "FAIL-SAFE: SHARE_CLASSES_FRE sem colunas "
            f"necessárias ao MARKET_CAP: {sorted(missing_share_columns)}"
        )

    market_cap_result = call_stage(
        "MARKET_CAP",
        calculate_market_cap,
        [
            (
                (),
                {
                    "security_universe": security_universe,
                    "share_classes": share_classes_for_market,
                },
            ),
            (
                (
                    security_universe,
                    share_classes_for_market,
                ),
                {},
            ),
        ],
    )

    if (
        not isinstance(market_cap_result, tuple)
        or len(market_cap_result) != 2
    ):
        raise PipelineIntegrityError(
            "FAIL-SAFE: calculate_issuer_market_cap deve retornar "
            "(issuer_market_cap, class_detail)."
        )

    issuer_market_cap, class_detail = (
        market_cap_result
    )

    require_dataframe(
        issuer_market_cap,
        "ISSUER_MARKET_CAP",
    )

    # class_detail pode estar vazio quando nenhuma classe consegue ser
    # casada com segurança. Isso não autoriza inventar MARKET_CAP.
    if not isinstance(
        class_detail,
        pd.DataFrame,
    ):
        raise PipelineIntegrityError(
            "FAIL-SAFE: MARKET_CAP class_detail não é DataFrame."
        )

    required_market_cap_columns = {
        "ISSUER_ID",
        "MARKET_CAP",
        "MARKET_CAP_VALID",
        "MARKET_CAP_STATUS",
    }

    missing_market_cap_columns = (
        required_market_cap_columns
        - set(issuer_market_cap.columns)
    )

    if missing_market_cap_columns:
        raise PipelineIntegrityError(
            "FAIL-SAFE: ISSUER_MARKET_CAP sem colunas: "
            f"{sorted(missing_market_cap_columns)}"
        )

    if issuer_market_cap[
        "ISSUER_ID"
    ].duplicated().any():
        raise PipelineIntegrityError(
            "FAIL-SAFE: ISSUER_MARKET_CAP possui emissor duplicado."
        )

    if "ISSUER_ID" not in accounting_data.columns:
        raise PipelineIntegrityError(
            "FAIL-SAFE: ACCOUNTING_DATA sem ISSUER_ID."
        )

    if accounting_data[
        "ISSUER_ID"
    ].duplicated().any():
        raise PipelineIntegrityError(
            "FAIL-SAFE: ACCOUNTING_DATA possui emissor duplicado "
            "antes do MARKET_CAP."
        )

    # Não propaga MARKET_VALUE para a base contábil: valuation.py já sabe
    # construir MARKET_VALUE a partir de MARKET_CAP. Mantemos uma única
    # fonte canônica para evitar conflito de colunas.
    merge_columns = [
        column
        for column in (
            "ISSUER_ID",
            "MARKET_CAP",
            "MARKET_CAP_VALID",
            "MARKET_CAP_STATUS",
            "MARKET_CAP_N_CLASSES",
            "MARKET_CAP_SOURCE",
        )
        if column in issuer_market_cap.columns
    ]

    accounting_enriched = (
        accounting_data.merge(
            issuer_market_cap[
                merge_columns
            ],
            on="ISSUER_ID",
            how="left",
            validate="one_to_one",
        )
    )

    if len(accounting_enriched) != len(accounting_data):
        raise PipelineIntegrityError(
            "FAIL-SAFE: MARKET_CAP alterou a cardinalidade "
            "da base contábil."
        )

    assert_no_future_information(
        accounting_enriched,
        "ACCOUNTING_WITH_MARKET_CAP",
    )

    return (
        accounting_enriched,
        share_classes,
        issuer_market_cap,
    )


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
    "FUTURE_RETURN_USED_REPORT",
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

    # O Market Engine possui duas etapas explícitas:
    # 1) carregar o histórico COTAHIST em DataFrame;
    # 2) construir o universo usando esse DataFrame + contexto PIT.
    #
    # build_market_universe NÃO recebe uma lista de anos.

    read_function = resolve_callable(
        market_module,
        (
            "read_cotahist_years",
        ),
    )

    build_function = resolve_callable(
        market_module,
        (
            "build_market_universe",
        ),
    )

    # Carrega somente informação que pode ser conhecida até o market_cutoff.
    # O ano corrente é suficiente para a regra de sessões do projeto e evita
    # baixar períodos desnecessários.
    years = [
        int(context.market_cutoff.year)
    ]

    history = call_stage(
        "MARKET_READ",
        read_function,
        [
            (
                (years,),
                {},
            ),
            (
                (),
                {
                    "years": years,
                },
            ),
        ],
    )

    require_dataframe(
        history,
        "MARKET_HISTORY",
    )

    assert_no_future_information(
        history,
        "MARKET_HISTORY",
    )

    result = call_stage(
        "MARKET",
        build_function,
        [
            (
                (),
                {
                    "df": history,
                    "context": context,
                },
            ),
            (
                (
                    history,
                    context,
                ),
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

    # Salva a camada de mercado quando o módulo oferece a função oficial.
    save_function = resolve_callable(
        market_module,
        (
            "save_market_universe",
        ),
        required=False,
    )

    if save_function is not None:
        call_stage(
            "MARKET_SAVE",
            save_function,
            [
                (
                    (
                        result,
                        context,
                    ),
                    {},
                ),
                (
                    (),
                    {
                        "universe": result,
                        "context": context,
                    },
                ),
            ],
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
        len(result) in (2, 3)
    ):

        security_universe = result[0]
        issuer_universe = result[1]

        # A terceira posição, quando presente, é a auditoria produzida
        # pelo Universe Engine. Ela não altera seleção nem ranking.
        universe_audit = (
            result[2]
            if len(result) == 3
            else None
        )

    else:

        raise PipelineIntegrityError(
            "FAIL-SAFE: build_universe deve retornar "
            "dict ou tuple com universos de security e issuer "
            "(auditoria opcional como terceiro item)."
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
    security_universe: pd.DataFrame,
    fundamentals: pd.DataFrame,
    context: PITContext,
) -> pd.DataFrame:

    function = resolve_callable(
        investability_module,
        (
            "run_investability_engine",
        ),
    )

    result = call_stage(
        "INVESTABILITY",
        function,
        [
            (
                (),
                {
                    "issuer_universe": issuer_universe,
                    "security_universe": security_universe,
                    "fundamentals": fundamentals,
                    "context": context,
                },
            ),
            (
                (
                    issuer_universe,
                    security_universe,
                    fundamentals,
                    context,
                ),
                {},
            ),
        ],
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
    fundamental_base: pd.DataFrame,
    context: PITContext,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    function = resolve_callable(
        fundamentals_module,
        (
            "build_fundamentals",
        ),
    )

    result = call_stage(
        "FUNDAMENTALS",
        function,
        [
            (
                (),
                {
                    "fundamental_base": fundamental_base,
                    "context": context,
                },
            ),
            (
                (
                    fundamental_base,
                    context,
                ),
                {},
            ),
        ],
    )

    if (
        not isinstance(result, tuple)
        or len(result) != 2
    ):
        raise PipelineIntegrityError(
            "FAIL-SAFE: build_fundamentals deve retornar "
            "(fundamentals, coverage)."
        )

    fundamentals, coverage = result

    require_dataframe(
        fundamentals,
        "FUNDAMENTALS",
    )

    require_dataframe(
        coverage,
        "FUNDAMENTAL_COVERAGE",
    )

    assert_no_future_information(
        fundamentals,
        "FUNDAMENTALS",
    )

    return fundamentals, coverage

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
    identity_map: Optional[pd.DataFrame] = None,
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
    # MARKET
    # -------------------------------------------------------------------------

    market_universe = run_market_stage(
        context
    )

    # -------------------------------------------------------------------------
    # ACCOUNTING / IDENTITY
    #
    # Em produção, quando as bases não forem fornecidas externamente,
    # constrói automaticamente a ponte ticker → emissor e a base contábil
    # oficial CVM respeitando o contexto PIT.
    # -------------------------------------------------------------------------

    if identity_map is None or accounting_data is None:

        build_production_accounting = resolve_callable(
            accounting_module,
            (
                "build_production_accounting",
            ),
        )

        market_tickers = (
            market_universe["TICKER"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
            .tolist()
        )

        generated_identity, generated_accounting = call_stage(
            "ACCOUNTING",
            build_production_accounting,
            [
                (
                    (),
                    {
                        "formation_date": context.formation_date,
                        "accounting_cutoff": context.accounting_cutoff,
                        "market_tickers": market_tickers,
                    },
                ),
                (
                    (
                        context.formation_date,
                        context.accounting_cutoff,
                    ),
                    {
                        "market_tickers": market_tickers,
                    },
                ),
            ],
        )

        if identity_map is None:
            identity_map = generated_identity

        if accounting_data is None:
            accounting_data = generated_accounting

    identity = resolve_identity_map(
        identity_map
    )

    require_dataframe(
        accounting_data,
        "ACCOUNTING_DATA",
    )

    assert_no_future_information(
        accounting_data,
        "ACCOUNTING_DATA",
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
    # SHARE CLASSES (FRE) / MARKET CAP
    #
    # O valor de mercado é calculado por classe de ação:
    # preço da classe no COTAHIST x quantidade outstanding da classe no FRE,
    # somado por emissor. A base enriquecida segue para Fundamentals e,
    # consequentemente, para Valuation.
    # -------------------------------------------------------------------------

    (
        accounting_data,
        share_classes,
        issuer_market_cap,
    ) = run_market_cap_stage(
        security_universe=security_universe,
        accounting_data=accounting_data,
        context=context,
    )

    # -------------------------------------------------------------------------
    # IDENTIDADE DO EMISSOR -> BASE FUNDAMENTAL
    #
    # O Universe Engine já consolidou a relação econômica por ISSUER_ID e
    # preservou TICKERS / COMPANY_NAME quando disponíveis. A base contábil,
    # porém, é a entrada efetiva de Fundamentals e, sem esta ponte explícita,
    # esses campos de identidade se perdem antes de Ranking/Risk/Report.
    #
    # Regras:
    # - merge somente por ISSUER_ID;
    # - somente campos descritivos, nunca fatores/scoring;
    # - cardinalidade one_to_one obrigatória;
    # - não sobrescreve coluna já existente na base contábil;
    # - não altera o modelo 50/30/20;
    # - não introduz informação futura.
    # -------------------------------------------------------------------------

    if "ISSUER_ID" not in issuer_universe.columns:
        raise PipelineIntegrityError(
            "FAIL-SAFE: ISSUER_UNIVERSE sem ISSUER_ID "
            "para propagação de identidade."
        )

    if issuer_universe["ISSUER_ID"].duplicated().any():
        raise PipelineIntegrityError(
            "FAIL-SAFE: ISSUER_UNIVERSE possui ISSUER_ID duplicado "
            "antes da propagação de identidade."
        )

    if "ISSUER_ID" not in accounting_data.columns:
        raise PipelineIntegrityError(
            "FAIL-SAFE: ACCOUNTING_DATA sem ISSUER_ID "
            "antes da propagação de identidade."
        )

    identity_columns = [
        column
        for column in (
            "TICKERS",
            "COMPANY_NAME",
            "CNPJ",
            "CD_CVM",
        )
        if (
            column in issuer_universe.columns
            and column not in accounting_data.columns
        )
    ]

    if identity_columns:

        rows_before_identity_merge = len(
            accounting_data
        )

        accounting_data = (
            accounting_data.merge(
                issuer_universe[
                    [
                        "ISSUER_ID",
                        *identity_columns,
                    ]
                ],
                on="ISSUER_ID",
                how="left",
                validate="one_to_one",
            )
        )

        if (
            len(accounting_data)
            != rows_before_identity_merge
        ):
            raise PipelineIntegrityError(
                "FAIL-SAFE: propagação de identidade alterou "
                "a cardinalidade da base contábil."
            )

        assert_no_future_information(
            accounting_data,
            "ACCOUNTING_WITH_ISSUER_IDENTITY",
        )

    # -------------------------------------------------------------------------
    # FUNDAMENTALS
    #
    # O Fundamental Engine transforma a base contábil PIT em indicadores.
    # Ele precisa existir ANTES do Investability Gate, pois o gate valida
    # cobertura fundamental mínima por emissor.
    #
    # A partir daqui TICKERS / COMPANY_NAME, quando disponíveis no Universe,
    # acompanham a base como metadados descritivos até o relatório.
    # -------------------------------------------------------------------------

    fundamentals, fundamental_coverage = (
        run_fundamentals_stage(
            fundamental_base=accounting_data,
            context=context,
        )
    )

    # -------------------------------------------------------------------------
    # INVESTABILITY
    #
    # Assinatura oficial do engine:
    # issuer_universe + security_universe + fundamentals + context.
    # -------------------------------------------------------------------------

    investability = (
        run_investability_stage(
            issuer_universe=issuer_universe,
            security_universe=security_universe,
            fundamentals=fundamentals,
            context=context,
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
    # Executa o engine que contém o score fundamental do estudo.
    # A autorização no ranking continua condicionada ao config e aos
    # fail-safes do Ranking, Risk e auditoria global.
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

    # O score 50/30/20 do estudo pode entrar no Ranking quando sua
    # integridade já foi validada pelo Ranking Engine e pelo Risk Engine.
    # O main não recalcula o modelo; apenas confirma os metadados finais.
    if audit[
        "turnaround_research_used_in_ranking"
    ]:

        required_study_columns = {
            "RANKING_MODEL",
            "MARGEM_BRUTA_WEIGHT",
            "MARGEM_LIQUIDA_WEIGHT",
            "ROE_WEIGHT",
        }

        missing_study_columns = (
            required_study_columns
            - set(ranking.columns)
        )

        if missing_study_columns:
            raise PipelineIntegrityError(
                "FAIL-SAFE: score do estudo entrou no ranking sem "
                f"metadados obrigatórios: {sorted(missing_study_columns)}"
            )

        used_mask = (
            ranking[
                "TURNAROUND_RESEARCH_USED_IN_RANKING"
            ]
            .fillna(False)
            .astype(bool)
        )

        models = (
            ranking.loc[
                used_mask,
                "RANKING_MODEL",
            ]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        if models != ["MB_50_ML_30_ROE_20_STUDY"]:
            raise PipelineIntegrityError(
                "FAIL-SAFE: modelo fundamental não autorizado "
                f"detectado no ranking: {models}"
            )

        expected_weights = {
            "MARGEM_BRUTA_WEIGHT": 0.50,
            "MARGEM_LIQUIDA_WEIGHT": 0.30,
            "ROE_WEIGHT": 0.20,
        }

        for column, expected in expected_weights.items():

            values = pd.to_numeric(
                ranking.loc[
                    used_mask,
                    column,
                ],
                errors="coerce",
            )

            if values.isna().any():
                raise PipelineIntegrityError(
                    "FAIL-SAFE: peso ausente/inválido no ranking: "
                    f"{column}"
                )

            if not (
                (values - expected).abs() <= 1e-12
            ).all():
                raise PipelineIntegrityError(
                    "FAIL-SAFE: peso divergente do estudo no ranking: "
                    f"{column}; esperado={expected}."
                )

        for column in (
            "QUALITY_WEIGHT",
            "VALUATION_WEIGHT",
        ):
            if column in ranking.columns:

                values = pd.to_numeric(
                    ranking.loc[
                        used_mask,
                        column,
                    ],
                    errors="coerce",
                )

                if (
                    values.isna().any()
                    or not (
                        values.abs() <= 1e-12
                    ).all()
                ):
                    raise PipelineIntegrityError(
                        "FAIL-SAFE: peso legado diferente de zero "
                        f"detectado em {column}."
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

    save_fundamentals_function = resolve_callable(
        fundamentals_module,
        (
            "save_fundamentals",
        ),
        required=False,
    )

    if save_fundamentals_function is not None:
        stage_files[
            "fundamentals"
        ] = call_stage(
            "FUNDAMENTALS_SAVE",
            save_fundamentals_function,
            [
                (
                    (
                        fundamentals,
                        fundamental_coverage,
                        context,
                    ),
                    {},
                ),
                (
                    (),
                    {
                        "fundamentals": fundamentals,
                        "coverage": fundamental_coverage,
                        "context": context,
                    },
                ),
            ],
        )
    else:
        stage_files[
            "fundamentals"
        ] = None

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
            "0.3.0",

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

        "share_classes":
            share_classes,

        "issuer_market_cap":
            issuer_market_cap,

        "investability":
            investability,

        "fundamentals":
            fundamentals,

        "fundamental_coverage":
            fundamental_coverage,

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
                "0.3.0",

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

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "B3 STOCK SELECTOR — execução segura do pipeline."
        )
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="Executa o pipeline real de produção.",
    )

    parser.add_argument(
        "--formation-date",
        default=None,
        help="Data de formação YYYY-MM-DD. Padrão: data UTC da execução.",
    )

    parser.add_argument(
        "--accounting-cutoff",
        default=None,
        help=(
            "Cutoff contábil YYYY-MM-DD. "
            "Se omitido, usa o último trimestre civil encerrado "
            "com defasagem conservadora de divulgação."
        ),
    )

    parser.add_argument(
        "--market-cutoff",
        default=None,
        help="Cutoff de mercado YYYY-MM-DD. Padrão: formation-date.",
    )

    args = parser.parse_args()

    print("=" * 78)
    print(f"{PROJECT_NAME} — MAIN")
    print("=" * 78)

    if not args.run:
        print("Orquestrador carregado com sucesso.")
        print()
        print(
            "Execução real requer --run. "
            "Nenhum dado de produção foi processado."
        )
        print()
        print(
            "Pipeline: PIT → MARKET → ACCOUNTING → UNIVERSE → "
            "FRE/CLASSES → MARKET_CAP → FUNDAMENTALS → INVESTABILITY → "
            "QUALITY / TURNAROUND / "
            "VALUATION → RANKING → RISK → REPORT"
        )
        print()
        print("Modelo do estudo 50/30/20: AUTORIZADO COM FAIL-SAFE")
        print("Retorno futuro: PROIBIDO")
        print("Look-ahead: PROIBIDO")
        print("=" * 78)
        sys.exit(0)

    # -------------------------------------------------------------------------
    # DATAS DE PRODUÇÃO
    # -------------------------------------------------------------------------

    formation_date = (
        pd.Timestamp(args.formation_date).normalize()
        if args.formation_date
        else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    )

    market_cutoff = (
        pd.Timestamp(args.market_cutoff).normalize()
        if args.market_cutoff
        else formation_date
    )

    if args.accounting_cutoff:

        accounting_cutoff = (
            pd.Timestamp(args.accounting_cutoff)
            .normalize()
        )

    else:
        # Regra conservadora e determinística:
        # usa o trimestre encerrado anterior ao trimestre corrente.
        # Isso evita presumir que demonstrações do trimestre recém-encerrado
        # já estejam publicadas no momento da execução.
        quarter_start_month = (
            ((formation_date.month - 1) // 3) * 3 + 1
        )

        quarter_start = pd.Timestamp(
            year=formation_date.year,
            month=quarter_start_month,
            day=1,
        )

        accounting_cutoff = (
            quarter_start - pd.offsets.Day(1)
        ).normalize()

    if accounting_cutoff > formation_date:
        raise PipelineIntegrityError(
            "FAIL-SAFE: accounting_cutoff posterior à formation_date."
        )

    if market_cutoff > formation_date:
        raise PipelineIntegrityError(
            "FAIL-SAFE: market_cutoff posterior à formation_date."
        )

    print("Modo: PRODUÇÃO")
    print("Formation date:", formation_date.date())
    print("Accounting cutoff:", accounting_cutoff.date())
    print("Market cutoff:", market_cutoff.date())
    print("=" * 78)

    result = run_pipeline_safe(
        formation_date=formation_date,
        accounting_cutoff=accounting_cutoff,
        market_cutoff=market_cutoff,
        identity_map=None,
        accounting_data=None,
        safety_data=None,
    )

    print_run_summary(
        result
    )
