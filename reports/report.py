# =============================================================================
# B3 STOCK SELECTOR
# Arquivo: reports/report.py
#
# REPORT ENGINE
#
# OBJETIVO:
# Consolidar os resultados finais do B3 STOCK SELECTOR em um relatório
# determinístico, auditável e legível.
#
# MOSTRA:
# - contexto Point-in-Time;
# - universo analisado;
# - quantidade elegível;
# - carteira selecionada;
# - peso de cada posição;
# - Final Score;
# - Quality;
# - Valuation;
# - controles de risco;
# - histórico do modelo legado desativado;
# - motivos de exclusão;
# - alertas metodológicos.
#
# NÃO FAZ:
# - recalcular indicadores;
# - alterar scores;
# - alterar ranking;
# - alterar pesos;
# - descobrir fatores;
# - usar retorno futuro;
# - liberar recomendações sem evidências das 34 etapas.
#
# PRINCÍPIO:
# O relatório descreve exatamente o que os motores decidiram.
# Ele não toma novas decisões.
# =============================================================================

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
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


try:
    from data.pit import PITContext

except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar PITContext."
    ) from exc


# =============================================================================
# DIRETÓRIO
# =============================================================================

REPORT_DIR = (
    OUTPUT_DIR
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# EXCEÇÕES
# =============================================================================

class ReportError(RuntimeError):
    """Erro geral do Report Engine."""


class ReportIntegrityError(ReportError):
    """Erro de integridade dos dados."""


# =============================================================================
# PROTEÇÕES
# =============================================================================

FORBIDDEN_TERMS = (
    "FUTURE_RETURN",
    "RETORNO_FUTURO",
    "RETURN_FUTURE",
    "FUTURE_WINNER",
    "WINNER_LABEL",
    "TARGET_RETURN",
)

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

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise ReportIntegrityError(
            f"{name} não é DataFrame."
        )

    if df.empty:
        raise ReportIntegrityError(
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
        raise ReportIntegrityError(
            f"FAIL-SAFE: {name} sem colunas obrigatórias: "
            f"{sorted(missing)}"
        )


def _json_safe(
    value,
):
    """
    Converte tipos numpy/pandas para tipos serializáveis em JSON.
    """

    if value is None:
        return None

    if value is pd.NA:
        return None

    if isinstance(
        value,
        (
            np.integer,
        ),
    ):
        return int(
            value
        )

    if isinstance(
        value,
        (
            np.floating,
        ),
    ):

        if np.isnan(
            value
        ):
            return None

        return float(
            value
        )

    if isinstance(
        value,
        (
            np.bool_,
        ),
    ):
        return bool(
            value
        )

    if isinstance(
        value,
        (
            pd.Timestamp,
            datetime,
        ),
    ):
        return value.isoformat()

    try:

        if pd.isna(
            value
        ):
            return None

    except Exception:
        pass

    return value


# =============================================================================
# PROTEÇÃO CONTRA FUTURO
# =============================================================================

def assert_no_future_information(
    df: pd.DataFrame,
) -> None:

    forbidden = []

    for column in df.columns:

        normalized = (
            str(column)
            .upper()
        )

        if normalized in ALLOWED_METADATA:
            continue

        if any(
            term in normalized
            for term in FORBIDDEN_TERMS
        ):
            forbidden.append(
                column
            )

    if forbidden:
        raise ReportIntegrityError(
            "FAIL-SAFE: informação futura detectada "
            "no Report Engine: "
            f"{forbidden}"
        )


# =============================================================================
# VALIDAÇÃO DO RESULTADO DE RISCO
# =============================================================================

def prepare_risk_result(risk_result: pd.DataFrame) -> pd.DataFrame:
    """Aceita resultados pendentes, mas nunca carteira ou ranking não validados."""
    _require_columns(risk_result, {"ISSUER_ID"}, "risk_result")
    assert_no_future_information(risk_result)
    result = risk_result.copy()
    if result["ISSUER_ID"].isna().any() or result["ISSUER_ID"].duplicated().any():
        raise ReportIntegrityError("FAIL-SAFE: emissor ausente ou duplicado.")
    for column in ("FINAL_SCORE", "FINAL_RANK", "RANKING_ELIGIBLE",
                   "RISK_SELECTED", "RISK_STATUS", "RISK_REASON",
                   "PORTFOLIO_WEIGHT"):
        if column not in result:
            result[column] = pd.NA
    def explicitly_true(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ("true", "sim", "1")
        return False
    for column in ("RISK_SELECTED", "RANKING_ELIGIBLE",
                   "TURNAROUND_RESEARCH_USED_IN_RANKING",
                   "TURNAROUND_RESEARCH_USED_RISK"):
        if column in result:
            result[column] = result[column].map(explicitly_true).astype(bool)
    for column in ("RISK_SELECTED", "RANKING_ELIGIBLE",
                   "TURNAROUND_RESEARCH_USED_IN_RANKING",
                   "TURNAROUND_RESEARCH_USED_RISK"):
        if column in result and result[column].any():
            raise ReportIntegrityError(
                f"FAIL-SAFE: {column} indica seleção sem aprovação das 34 etapas."
            )
    weights = pd.to_numeric(result["PORTFOLIO_WEIGHT"], errors="coerce")
    if weights.fillna(0).ne(0).any():
        raise ReportIntegrityError("FAIL-SAFE: carteira ponderada sem aprovação.")
    result["PORTFOLIO_WEIGHT"] = 0.0
    for column in ("FINAL_SCORE", "FINAL_RANK"):
        if pd.to_numeric(result[column], errors="coerce").notna().any():
            raise ReportIntegrityError(
                f"FAIL-SAFE: {column} legado não validado."
            )
    result["RISK_SELECTED"] = False
    result["RANKING_ELIGIBLE"] = False
    result["STUDY_34_STATUS"] = "PENDENTE"
    result["STUDY_34_EVIDENCE_VERIFIED"] = False
    return result


# =============================================================================
# IDENTIFICAÇÃO
# =============================================================================

def _first_existing_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:

    for column in candidates:

        if column in df.columns:
            return column

    return None


def _company_name_column(
    df: pd.DataFrame,
) -> Optional[str]:

    return _first_existing_column(
        df,
        (
            "NOME_EMPRESARIAL",
            "Nome_Empresarial",
            "COMPANY_NAME",
            "ISSUER_NAME",
            "DENOM_SOCIAL",
        ),
    )


def _ticker_column(
    df: pd.DataFrame,
) -> Optional[str]:

    return _first_existing_column(
        df,
        (
            "TICKERS",
            "TICKER",
        ),
    )


def _sector_column(
    df: pd.DataFrame,
) -> Optional[str]:

    return _first_existing_column(
        df,
        (
            "FCA_SETOR_ATIVIDADE",
            "SECTOR",
            "SETOR",
            "SECTOR_BUCKET",
        ),
    )


# =============================================================================
# CARTEIRA
# =============================================================================

def build_portfolio_table(
    result: pd.DataFrame,
) -> pd.DataFrame:

    selected = (
        result.loc[
            result[
                "RISK_SELECTED"
            ]
        ]
        .copy()
    )

    if selected.empty:
        return selected

    columns = [
        "ISSUER_ID",
    ]

    ticker_col = _ticker_column(
        selected
    )

    company_col = _company_name_column(
        selected
    )

    sector_col = _sector_column(
        selected
    )

    if ticker_col is not None:
        columns.append(
            ticker_col
        )

    if company_col is not None:
        columns.append(
            company_col
        )

    if sector_col is not None:
        columns.append(
            sector_col
        )

    optional = (
        "FINAL_RANK",
        "FINAL_SCORE",
        "TURNAROUND_RESEARCH_SCORE",
        "QUALITY_SCORE",
        "VALUATION_SCORE",
        "TURNAROUND_RESEARCH_STATUS",
        "GROSS_MARGIN_ZONE",
        "PORTFOLIO_WEIGHT",
    )

    for column in optional:

        if column in selected.columns:
            columns.append(
                column
            )

    portfolio = (
        selected[
            columns
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
        .reset_index(
            drop=True
        )
    )

    return portfolio


# =============================================================================
# RESUMO DO UNIVERSO
# =============================================================================

def build_universe_summary(
    result: pd.DataFrame,
) -> dict:

    ranking_eligible = (
        result[
            "RANKING_ELIGIBLE"
        ]
        .fillna(False)
        .astype(bool)
    )

    selected = (
        result[
            "RISK_SELECTED"
        ]
        .fillna(False)
        .astype(bool)
    )

    summary = {
        "issuers_total":
            int(
                len(
                    result
                )
            ),

        "ranking_eligible":
            int(
                ranking_eligible.sum()
            ),

        "ranking_ineligible":
            int(
                (
                    ~ranking_eligible
                ).sum()
            ),

        "portfolio_selected":
            int(
                selected.sum()
            ),

        "portfolio_weight_total":
            float(
                result.loc[
                    selected,
                    "PORTFOLIO_WEIGHT",
                ]
                .sum()
            ),
    }

    return summary


# =============================================================================
# RESUMO DOS SCORES
# =============================================================================

def build_score_summary(
    result: pd.DataFrame,
) -> dict:

    summary = {}

    score_columns = (
        "FINAL_SCORE",
        "QUALITY_SCORE",
        "VALUATION_SCORE",
        "TURNAROUND_RESEARCH_SCORE",
    )

    for column in score_columns:

        if column not in result.columns:
            continue

        numeric = pd.to_numeric(
            result[
                column
            ],
            errors="coerce",
        )

        valid = numeric.dropna()

        if valid.empty:

            summary[
                column
            ] = {
                "available": 0,
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
            }

            continue

        summary[
            column
        ] = {
            "available":
                int(
                    len(
                        valid
                    )
                ),

            "mean":
                float(
                    valid.mean()
                ),

            "median":
                float(
                    valid.median()
                ),

            "min":
                float(
                    valid.min()
                ),

            "max":
                float(
                    valid.max()
                ),
        }

    return summary


# =============================================================================
# RISCO / CONCENTRAÇÃO
# =============================================================================

def build_risk_summary(
    result: pd.DataFrame,
) -> dict:

    selected = (
        result.loc[
            result[
                "RISK_SELECTED"
            ]
        ]
        .copy()
    )

    if selected.empty:

        return {
            "positions": 0,
            "largest_position": None,
            "sector_weights": {},
        }

    largest_position = float(
        selected[
            "PORTFOLIO_WEIGHT"
        ]
        .max()
    )

    sector_col = _sector_column(
        selected
    )

    sector_weights = {}

    if sector_col is not None:

        sector_data = (
            selected[
                [
                    sector_col,
                    "PORTFOLIO_WEIGHT",
                ]
            ]
            .copy()
        )

        sector_data[
            sector_col
        ] = (
            sector_data[
                sector_col
            ]
            .fillna(
                "UNKNOWN"
            )
            .astype(str)
        )

        grouped = (
            sector_data
            .groupby(
                sector_col
            )[
                "PORTFOLIO_WEIGHT"
            ]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        sector_weights = {
            str(key):
                float(value)

            for key, value
            in grouped.items()
        }

    return {
        "positions":
            int(
                len(
                    selected
                )
            ),

        "largest_position":
            largest_position,

        "sector_weights":
            sector_weights,
    }


# =============================================================================
# MOTIVOS DE EXCLUSÃO
# =============================================================================

def build_exclusion_summary(
    result: pd.DataFrame,
) -> dict:

    summary = {}

    for column in (
        "RANKING_ELIGIBILITY_REASON",
        "RISK_REASON",
    ):

        if column not in result.columns:
            continue

        counts = (
            result[
                column
            ]
            .fillna(
                "UNKNOWN"
            )
            .astype(str)
            .value_counts()
            .to_dict()
        )

        summary[
            column
        ] = {
            str(key):
                int(value)

            for key, value
            in counts.items()
        }

    return summary


# =============================================================================
# TURNAROUND — SOMENTE PESQUISA
# =============================================================================

def build_turnaround_research_summary(result: pd.DataFrame) -> dict:
    """Informação histórica; não reproduz pesos nem aval do modelo antigo."""
    status_col = "TURNAROUND_RESEARCH_STATUS"
    zone_col = "GROSS_MARGIN_ZONE"
    return {
        "production_authorized": False,
        "used_in_final_ranking": False,
        "used_in_risk": False,
        "model": "DISABLED_LEGACY_MB100",
        "weights": {},
        "status_distribution": (
            result[status_col].value_counts(dropna=False).to_dict()
            if status_col in result else {}
        ),
        "gross_margin_zone_distribution": (
            result[zone_col].value_counts(dropna=False).to_dict()
            if zone_col in result else {}
        ),
    }


# =============================================================================
# ALERTAS METODOLÓGICOS
# =============================================================================

def build_methodology_notes(result: pd.DataFrame) -> list[str]:
    return [
        "Relatório de pesquisa das 34 etapas; nenhuma empresa integralmente aprovada.",
        "A metodologia antiga DISABLED_LEGACY_MB100 está desativada.",
        "Pontuações e rankings antigos não são critérios de investimento.",
        "CAPEX, fluxo de caixa livre, dívida líquida, ROIC/WACC e valor justo "
        "dependem de evidências documentais independentes.",
        "Sem comprovação integral, o status é PENDENTE, não APROVADO.",
        "Retornos futuros são proibidos nas decisões Point-in-Time.",
        "Nenhuma carteira ou recomendação de compra é emitida.",
    ]


# =============================================================================
# RELATÓRIO ESTRUTURADO
# =============================================================================

def build_report(
    risk_result: pd.DataFrame,
    context: PITContext,
) -> dict:

    result = prepare_risk_result(
        risk_result
    )

    portfolio = build_portfolio_table(
        result
    )

    report = {
        "project": {
            "name":
                PROJECT_NAME,

            "version":
                VERSION,

            "report_engine_version":
                "0.2.0",

            "created_at_utc":
                _utc_now_iso(),
        },

        "point_in_time": {
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

            "future_return_used":
                False,
        },

        "universe":
            build_universe_summary(
                result
            ),

        "scores":
            build_score_summary(
                result
            ),

        "risk":
            build_risk_summary(
                result
            ),

        "exclusions":
            build_exclusion_summary(
                result
            ),

        "turnaround_research":
            build_turnaround_research_summary(
                result
            ),

        "methodology_notes":
            build_methodology_notes(
                result
            ),

        "study_34": {
            "status": "PENDENTE",
            "fully_approved_issuers": 0,
            "evidence_verified": False,
            "investment_recommendations_authorized": False,
        },

        "portfolio":
            [
                {
                    str(key):
                        _json_safe(
                            value
                        )

                    for key, value
                    in row.items()
                }

                for row
                in portfolio.to_dict(
                    orient="records"
                )
            ],
    }

    return report


# =============================================================================
# HTML
# =============================================================================

def _format_percent(
    value,
) -> str:

    if value is None:
        return "-"

    try:

        if pd.isna(
            value
        ):
            return "-"

    except Exception:
        pass

    return (
        f"{float(value) * 100:.2f}%"
    )


def _format_score(
    value,
) -> str:

    if value is None:
        return "-"

    try:

        if pd.isna(
            value
        ):
            return "-"

    except Exception:
        pass

    return (
        f"{float(value):.4f}"
    )


def portfolio_to_html(
    portfolio: pd.DataFrame,
) -> str:

    if portfolio.empty:

        return (
            "<p>Nenhuma empresa foi selecionada.</p>"
        )

    display = (
        portfolio
        .copy()
    )

    if (
        "PORTFOLIO_WEIGHT"
        in display.columns
    ):

        display[
            "PORTFOLIO_WEIGHT"
        ] = (
            display[
                "PORTFOLIO_WEIGHT"
            ]
            .map(
                _format_percent
            )
        )

    for column in (
        "FINAL_SCORE",
        "QUALITY_SCORE",
        "VALUATION_SCORE",
        "TURNAROUND_RESEARCH_SCORE",
    ):

        if column in display.columns:

            display[
                column
            ] = (
                display[
                    column
                ]
                .map(
                    _format_score
                )
            )

    return display.to_html(
        index=False,
        border=0,
        classes="portfolio-table",
        escape=True,
    )


def build_html_report(
    risk_result: pd.DataFrame,
    context: PITContext,
) -> str:

    result = prepare_risk_result(
        risk_result
    )

    report = build_report(
        result,
        context,
    )

    portfolio = build_portfolio_table(
        result
    )

    universe = report[
        "universe"
    ]

    risk = report[
        "risk"
    ]

    turnaround = report[
        "turnaround_research"
    ]

    notes_html = "".join(
        f"<li>{note}</li>"
        for note
        in report[
            "methodology_notes"
        ]
    )

    sector_rows = ""

    for (
        sector,
        weight,
    ) in risk[
        "sector_weights"
    ].items():

        sector_rows += (
            "<tr>"
            f"<td>{sector}</td>"
            f"<td>{_format_percent(weight)}</td>"
            "</tr>"
        )

    if not sector_rows:

        sector_rows = (
            "<tr>"
            "<td colspan='2'>"
            "Informação setorial não disponível."
            "</td>"
            "</tr>"
        )

    html = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>{PROJECT_NAME} - Relatório</title>

<style>
body {{
    font-family: Arial, Helvetica, sans-serif;
    margin: 32px;
    color: #222;
    background: #fff;
}}

h1 {{
    margin-bottom: 4px;
}}

h2 {{
    margin-top: 32px;
    border-bottom: 1px solid #ddd;
    padding-bottom: 6px;
}}

.meta {{
    color: #555;
    margin-bottom: 24px;
}}

.cards {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}}

.card {{
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 14px 18px;
    min-width: 170px;
}}

.card strong {{
    display: block;
    font-size: 22px;
    margin-top: 5px;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    margin-top: 12px;
}}

th, td {{
    border-bottom: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}}

th {{
    background: #f4f4f4;
}}

.warning {{
    border: 1px solid #d6b656;
    padding: 14px;
    margin-top: 16px;
}}

.research {{
    border: 1px solid #aaa;
    padding: 14px;
    margin-top: 16px;
}}

.footer {{
    margin-top: 40px;
    color: #666;
    font-size: 12px;
}}
</style>
</head>

<body>

<h1>{PROJECT_NAME}</h1>

<div class="meta">
Versão {VERSION}<br>
Data de formação:
{context.formation_date.date()}<br>
Cutoff contábil:
{context.accounting_cutoff.date()}<br>
Cutoff de mercado:
{context.market_cutoff.date()}
</div>

<h2>Resumo</h2>

<div class="cards">

<div class="card">
Emissores
<strong>{universe["issuers_total"]}</strong>
</div>

<div class="card">
Elegíveis ao ranking
<strong>{universe["ranking_eligible"]}</strong>
</div>

<div class="card">
Selecionados
<strong>{universe["portfolio_selected"]}</strong>
</div>

<div class="card">
Maior posição
<strong>{_format_percent(risk["largest_position"])}</strong>
</div>

</div>

<h2>Carteira selecionada</h2>

{portfolio_to_html(portfolio)}

<h2>Concentração setorial</h2>

<table>
<thead>
<tr>
<th>Setor</th>
<th>Peso</th>
</tr>
</thead>
<tbody>
{sector_rows}
</tbody>
</table>

<h2>Modelo fundamental do estudo</h2>

<div class="research">

<strong>Status:</strong>
OPERACIONAL COM FAIL-SAFE<br><br>

<strong>Modelo:</strong>
{turnaround["model"] or "DISABLED_LEGACY_MB100"}<br>

<strong>Pesos:</strong>
Margem Bruta 100% / Margem Líquida 0% / ROE 0%<br>

<strong>Direção:</strong>
LOW para Margem Bruta; Margem Líquida e ROE sem peso no score<br>

<strong>Produção autorizada:</strong>
{turnaround["production_authorized"]}<br>

<strong>Usado no ranking:</strong>
{turnaround["used_in_final_ranking"]}<br>

<strong>Recalculado pelo Risk:</strong>
{turnaround["used_in_risk"]}<br><br>

O relatório não recalcula o modelo e não utiliza retorno futuro.

</div>

<h2>Controles metodológicos</h2>

<div class="warning">
<ul>
{notes_html}
</ul>
</div>

<div class="footer">
Relatório produzido automaticamente pelo {PROJECT_NAME}.<br>
Retorno futuro utilizado na execução: NÃO.
</div>

</body>
</html>
"""

    return html


# =============================================================================
# SALVAR
# =============================================================================

def save_report(
    risk_result: pd.DataFrame,
    context: PITContext,
) -> dict:

    result = prepare_risk_result(
        risk_result
    )

    report = build_report(
        result,
        context,
    )

    portfolio = build_portfolio_table(
        result
    )

    html = build_html_report(
        result,
        context,
    )

    date_tag = (
        context
        .formation_date
        .strftime(
            "%Y%m%d"
        )
    )

    json_path = (
        REPORT_DIR
        / f"report_{date_tag}.json"
    )

    html_path = (
        REPORT_DIR
        / f"report_{date_tag}.html"
    )

    portfolio_path = (
        REPORT_DIR
        / f"portfolio_report_{date_tag}.csv"
    )

    manifest_path = (
        REPORT_DIR
        / f"report_manifest_{date_tag}.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
            default=_json_safe,
        )

    with html_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            html
        )

    portfolio.to_csv(
        portfolio_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "engine":
            "REPORT_ENGINE",

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

        "future_return_used":
            False,

        "study_model":
            "DISABLED_LEGACY_MB100",

        "study_weights": {
            "MARGEM_BRUTA": 1.00,
            "MARGEM_LIQUIDA": 0.00,
            "ROE": 0.00,
        },

        "study_model_used_in_ranking":
            bool(
                result.get(
                    "TURNAROUND_RESEARCH_USED_IN_RANKING",
                    pd.Series(False, index=result.index),
                )
                .fillna(False)
                .astype(bool)
                .any()
            ),

        "turnaround_recalculated_by_report":
            False,

        "portfolio_positions":
            int(
                len(
                    portfolio
                )
            ),

        "files": {
            "json":
                str(
                    json_path
                ),

            "html":
                str(
                    html_path
                ),

            "portfolio_csv":
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
        "json":
            json_path,

        "html":
            html_path,

        "portfolio":
            portfolio_path,

        "manifest":
            manifest_path,
    }


# =============================================================================
# SELF-TEST
# =============================================================================

def _self_test() -> bool:
    class Context:
        formation_date = pd.Timestamp("2026-01-15")
        accounting_cutoff = pd.Timestamp("2025-12-31")
        market_cutoff = pd.Timestamp("2026-01-15")
    sample = pd.DataFrame({
        "ISSUER_ID": ["1", "2"],
        "RISK_SELECTED": [False, False],
        "RANKING_ELIGIBLE": [False, False],
        "PORTFOLIO_WEIGHT": [0.0, 0.0],
        "FINAL_SCORE": [np.nan, np.nan],
        "FINAL_RANK": [np.nan, np.nan],
        "RISK_STATUS": ["PENDING", "PENDING"],
        "RISK_REASON": ["STUDY34_PENDING", "STUDY34_PENDING"],
    })
    report = build_report(sample, Context())
    if report["portfolio"] or not report["methodology_notes"]:
        raise ReportError("SELF-TEST: carteira ou metodologia incorreta.")
    for changes in (
        {"RISK_SELECTED": ["True", "False"]},
        {"PORTFOLIO_WEIGHT": [0.5, 0.0]},
        {"FINAL_SCORE": [0.9, np.nan]},
        {"FUTURE_RETURN": [0.5, 0.2]},
    ):
        try:
            prepare_risk_result(sample.assign(**changes))
        except ReportIntegrityError:
            pass
        else:
            raise ReportError("SELF-TEST: dado proibido não bloqueado.")
    return True


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("B3 STOCK SELECTOR — REPORT ENGINE")
    print("=" * 72)

    _self_test()

    print("Self-test: OK")
    print("Carteira: REPORTADA")
    print("Modelo legado desativado: REPORTADO E VALIDADO")
    print("Quality: AUXILIAR / SEM PESO NO SCORE PRINCIPAL")
    print("Valuation: AUXILIAR / SEM PESO NO SCORE PRINCIPAL")
    print("Risk: REPORTADO")
    print("Turnaround recalculado pelo Report: NÃO")
    print("Retorno futuro: NÃO UTILIZADO")
    print("Decisões recalculadas no relatório: NÃO")
    print("JSON: SUPORTADO")
    print("HTML: SUPORTADO")
    print("CSV da carteira: SUPORTADO")
    print("Auditoria: ATIVA")

    print("=" * 72)
