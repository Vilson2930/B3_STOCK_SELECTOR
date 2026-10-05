# =============================================================================
# B3 STOCK SELECTOR
# Arquivo: engines/investability.py
#
# INVESTABILITY / SAFETY GATE
#
# OBJETIVO:
# Antes de procurar oportunidades, verificar se a empresa possui condições
# mínimas para ser analisada pelo robô.
#
# RESPONSABILIDADE:
# - Validar identidade do emissor
# - Validar dados de mercado
# - Validar dados fundamentais mínimos
# - Validar conformidade PIT
# - Controlar missing data
# - Separar instituições financeiras do modelo fundamental padrão
# - Registrar exclusões e motivos
#
# NÃO FAZ:
# - Quality Score
# - Turnaround Score
# - Valuation
# - Ranking
# - Previsão de retorno
# - Uso de retorno futuro
#
# FILOSOFIA:
# O Investability Engine é um GATE, não um score.
#
# Uma empresa:
#
#     PASSA
#       ou
#     NÃO PASSA
#
# Nenhuma empresa recebe pontos extras por simplesmente cumprir requisitos
# mínimos de segurança.
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
    from config import (
        OUTPUT_DIR,
        INVESTABILITY,
        EXCLUDE_FINANCIALS_FROM_STANDARD_FUNDAMENTAL_MODEL,
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
# DIRETÓRIOS
# =============================================================================

INVESTABILITY_DIR = (
    OUTPUT_DIR
    / "investability"
)

INVESTABILITY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# EXCEÇÕES
# =============================================================================

class InvestabilityError(RuntimeError):
    """Erro geral do Investability Engine."""


class InvestabilityIntegrityError(InvestabilityError):
    """Erro estrutural ou de integridade."""


class InvestabilityPITError(InvestabilityError):
    """Violação Point-in-Time."""


# =============================================================================
# STATUS
# =============================================================================

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"

REASON_OK = "OK"

REASON_IDENTITY = "IDENTITY_NOT_RESOLVED"
REASON_MARKET = "MARKET_DATA_INVALID"
REASON_PIT = "PIT_VIOLATION"
REASON_FINANCIAL = "FINANCIAL_SPECIAL_MODEL_REQUIRED"
REASON_FUNDAMENTALS = "FUNDAMENTAL_DATA_INSUFFICIENT"
REASON_DUPLICATE = "DUPLICATE_ISSUER"


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
        raise InvestabilityIntegrityError(
            f"{name} não é DataFrame."
        )

    if df.empty:
        raise InvestabilityIntegrityError(
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
        raise InvestabilityIntegrityError(
            f"FAIL-SAFE: {name} sem colunas obrigatórias: "
            f"{sorted(missing)}"
        )


# =============================================================================
# NORMALIZAÇÃO ISSUER ID
# =============================================================================

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
# PREPARAÇÃO DO UNIVERSO
# =============================================================================

def prepare_issuer_universe(
    issuer_universe: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        "ISSUER_ID",
        "TICKERS",
        "N_SECURITIES",
        "IS_FINANCIAL",
        "FUNDAMENTAL_ELIGIBLE",
    }

    _require_columns(
        issuer_universe,
        required,
        "issuer_universe",
    )

    result = (
        issuer_universe
        .copy()
    )

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
        .isna()
        .any()
    ):
        raise InvestabilityIntegrityError(
            "FAIL-SAFE: ISSUER_ID ausente no universo."
        )

    duplicate = (
        result[
            "ISSUER_ID"
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicate.any():

        examples = (
            result.loc[
                duplicate,
                "ISSUER_ID",
            ]
            .head(20)
            .tolist()
        )

        raise InvestabilityIntegrityError(
            "FAIL-SAFE: emissor duplicado no universo. "
            f"Exemplos={examples}"
        )

    return result


# =============================================================================
# PREPARAÇÃO DOS FUNDAMENTOS
# =============================================================================

def prepare_fundamentals(
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:

    _require_columns(
        fundamentals,
        ["ISSUER_ID"],
        "fundamentals",
    )

    result = (
        fundamentals
        .copy()
    )

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

    result = result.loc[
        result[
            "ISSUER_ID"
        ]
        .notna()
    ].copy()

    if result.empty:
        raise InvestabilityIntegrityError(
            "FAIL-SAFE: nenhuma empresa válida "
            "nos fundamentos."
        )

    duplicates = (
        result[
            "ISSUER_ID"
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicates.any():

        examples = (
            result.loc[
                duplicates,
                "ISSUER_ID",
            ]
            .head(20)
            .tolist()
        )

        raise InvestabilityIntegrityError(
            "FAIL-SAFE: fundamentos possuem múltiplas "
            "linhas por emissor. "
            f"Exemplos={examples}"
        )

    return result


# =============================================================================
# COLUNAS FUNDAMENTAIS MÍNIMAS
#
# IMPORTANTE:
# Estes campos não significam que serão utilizados no ranking.
#
# Servem apenas para confirmar que existe informação contábil suficiente
# para que os motores seguintes façam sua análise.
# =============================================================================

DEFAULT_REQUIRED_FUNDAMENTALS = (
    "ATIVO_TOTAL",
    "PL",
    "RECEITA",
)


# =============================================================================
# VALIDAÇÃO PIT DOS FUNDAMENTOS
# =============================================================================

def validate_fundamental_pit(
    df: pd.DataFrame,
    context: PITContext,
) -> pd.Series:
    """
    Retorna True/False por empresa.

    Procura, em ordem de prioridade, colunas que indiquem
    explicitamente a data máxima usada na construção dos fundamentos.

    O engine NÃO assume silenciosamente que os dados são PIT.
    """

    candidate_columns = (
        "ACCOUNTING_CUTOFF",
        "PIT_CUTOFF",
        "DATA_CORTE",
        "REFERENCE_DATE",
        "DT_REFER",
    )

    selected_column = None

    for column in candidate_columns:

        if column in df.columns:
            selected_column = column
            break

    if selected_column is None:

        # Se o pipeline anterior já marcou explicitamente PIT válido,
        # essa evidência pode ser usada.
        if "PIT_VALID" in df.columns:

            raw = df[
                "PIT_VALID"
            ]

            if pd.api.types.is_bool_dtype(
                raw
            ):
                return (
                    raw
                    .fillna(False)
                    .astype(bool)
                )

            normalized = (
                raw
                .fillna("")
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
                    "OK",
                    "VALID",
                }
            )

        raise InvestabilityPITError(
            "FAIL-SAFE: fundamentos não possuem "
            "evidência temporal suficiente para validar PIT."
        )

    dates = pd.to_datetime(
        df[
            selected_column
        ],
        errors="coerce",
    )

    valid_date = (
        dates.notna()
        &
        (
            dates.dt.normalize()
            <= context.accounting_cutoff
        )
    )

    return valid_date


# =============================================================================
# COBERTURA FUNDAMENTAL
# =============================================================================

def fundamental_coverage(
    df: pd.DataFrame,
    required_columns: Iterable[str],
) -> pd.DataFrame:
    """
    Calcula cobertura dos campos fundamentais mínimos.
    """

    required_columns = list(
        required_columns
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise InvestabilityIntegrityError(
            "FAIL-SAFE: fundamentos sem campos mínimos: "
            f"{missing_columns}"
        )

    result = pd.DataFrame(
        index=df.index
    )

    available = (
        df[
            required_columns
        ]
        .notna()
    )

    result[
        "FUNDAMENTAL_FIELDS_REQUIRED"
    ] = len(
        required_columns
    )

    result[
        "FUNDAMENTAL_FIELDS_AVAILABLE"
    ] = (
        available
        .sum(
            axis=1
        )
        .astype(int)
    )

    if len(
        required_columns
    ) == 0:

        result[
            "FUNDAMENTAL_COVERAGE"
        ] = 1.0

    else:

        result[
            "FUNDAMENTAL_COVERAGE"
        ] = (
            result[
                "FUNDAMENTAL_FIELDS_AVAILABLE"
            ]
            / len(
                required_columns
            )
        )

    result[
        "FUNDAMENTAL_COMPLETE"
    ] = (
        result[
            "FUNDAMENTAL_FIELDS_AVAILABLE"
        ]
        ==
        result[
            "FUNDAMENTAL_FIELDS_REQUIRED"
        ]
    )

    return result


# =============================================================================
# VALIDAÇÃO DO MERCADO POR EMISSOR
# =============================================================================

def market_identity_status(
    security_universe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Consolida o estado de mercado por emissor.

    O Investability Gate não escolhe a melhor classe.
    Apenas verifica se o emissor possui pelo menos uma security
    negociável e válida.
    """

    required = {
        "ISSUER_ID",
        "TICKER",
        "FORMATION_PRICE",
        "FORMATION_PRICE_DATE",
        "IDENTITY_RESOLVED",
    }

    _require_columns(
        security_universe,
        required,
        "security_universe",
    )

    data = (
        security_universe
        .copy()
    )

    data[
        "ISSUER_ID"
    ] = (
        data[
            "ISSUER_ID"
        ]
        .map(
            _normalize_issuer_id
        )
    )

    prices = pd.to_numeric(
        data[
            "FORMATION_PRICE"
        ],
        errors="coerce",
    )

    dates = pd.to_datetime(
        data[
            "FORMATION_PRICE_DATE"
        ],
        errors="coerce",
    )

    data[
        "__MARKET_VALID"
    ] = (
        data[
            "IDENTITY_RESOLVED"
        ].fillna(False)
        &
        prices.notna()
        &
        (prices > 0)
        &
        dates.notna()
    )

    resolved = data.loc[
        data[
            "ISSUER_ID"
        ]
        .notna()
    ].copy()

    if resolved.empty:

        raise InvestabilityIntegrityError(
            "FAIL-SAFE: nenhuma security ligada a emissor."
        )

    grouped = (
        resolved
        .groupby(
            "ISSUER_ID",
            as_index=False,
        )
        .agg(
            MARKET_SECURITIES=(
                "TICKER",
                "nunique",
            ),
            MARKET_DATA_OK=(
                "__MARKET_VALID",
                "max",
            ),
        )
    )

    grouped[
        "MARKET_DATA_OK"
    ] = (
        grouped[
            "MARKET_DATA_OK"
        ]
        .fillna(False)
        .astype(bool)
    )

    return grouped


# =============================================================================
# MOTOR PRINCIPAL
# =============================================================================

def run_investability_gate(
    issuer_universe: pd.DataFrame,
    security_universe: pd.DataFrame,
    fundamentals: pd.DataFrame,
    context: PITContext,
    *,
    required_fundamentals: Optional[
        Iterable[str]
    ] = None,
) -> pd.DataFrame:
    """
    Executa o gate de investabilidade.

    Resultado:
        uma linha por emissor.

    INVESTABLE=True somente quando todos os requisitos mínimos
    aplicáveis forem satisfeitos.
    """

    if required_fundamentals is None:
        required_fundamentals = (
            DEFAULT_REQUIRED_FUNDAMENTALS
        )

    required_fundamentals = tuple(
        required_fundamentals
    )

    issuers = prepare_issuer_universe(
        issuer_universe
    )

    fundamental_data = prepare_fundamentals(
        fundamentals
    )

    market_status = market_identity_status(
        security_universe
    )

    # -------------------------------------------------------------------------
    # JUNÇÃO UNIVERSO + MERCADO
    # -------------------------------------------------------------------------

    result = issuers.merge(
        market_status,
        on="ISSUER_ID",
        how="left",
        validate="one_to_one",
    )

    # -------------------------------------------------------------------------
    # JUNÇÃO COM FUNDAMENTOS
    # -------------------------------------------------------------------------

    fundamental_columns = [
        column
        for column in fundamental_data.columns
        if column != "ISSUER_ID"
    ]

    result = result.merge(
        fundamental_data[
            [
                "ISSUER_ID",
                *fundamental_columns,
            ]
        ],
        on="ISSUER_ID",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_FUND",
        ),
    )

    # -------------------------------------------------------------------------
    # IDENTIDADE
    # -------------------------------------------------------------------------

    result[
        "IDENTITY_OK"
    ] = (
        result[
            "ISSUER_ID"
        ]
        .notna()
    )

    # -------------------------------------------------------------------------
    # MERCADO
    # -------------------------------------------------------------------------

    result[
        "MARKET_DATA_OK"
    ] = (
        result[
            "MARKET_DATA_OK"
        ]
        .fillna(False)
        .astype(bool)
    )

    # -------------------------------------------------------------------------
    # EXISTÊNCIA DE FUNDAMENTOS
    # -------------------------------------------------------------------------

    result[
        "FUNDAMENTAL_RECORD_FOUND"
    ] = (
        result[
            required_fundamentals
        ]
        .notna()
        .any(
            axis=1
        )
    )

    # -------------------------------------------------------------------------
    # COBERTURA
    # -------------------------------------------------------------------------

    coverage = fundamental_coverage(
        result,
        required_fundamentals,
    )

    for column in coverage.columns:

        result[
            column
        ] = coverage[
            column
        ]

    # -------------------------------------------------------------------------
    # PIT
    # -------------------------------------------------------------------------

    result[
        "PIT_OK"
    ] = False

    has_fundamental = (
        result[
            "FUNDAMENTAL_RECORD_FOUND"
        ]
    )

    if has_fundamental.any():

        pit_subset = (
            result.loc[
                has_fundamental
            ]
            .copy()
        )

        pit_valid = validate_fundamental_pit(
            pit_subset,
            context,
        )

        result.loc[
            has_fundamental,
            "PIT_OK",
        ] = (
            pit_valid
            .fillna(False)
            .astype(bool)
            .values
        )

    # -------------------------------------------------------------------------
    # FINANCEIRO
    # -------------------------------------------------------------------------

    result[
        "FINANCIAL_MODEL_OK"
    ] = True

    if (
        EXCLUDE_FINANCIALS_FROM_STANDARD_FUNDAMENTAL_MODEL
    ):

        result[
            "FINANCIAL_MODEL_OK"
        ] = (
            ~result[
                "IS_FINANCIAL"
            ]
            .fillna(False)
            .astype(bool)
        )

    # -------------------------------------------------------------------------
    # REGRAS CONFIGURADAS
    # -------------------------------------------------------------------------

    conditions = []

    if INVESTABILITY.get(
        "require_valid_issuer",
        True,
    ):
        conditions.append(
            result[
                "IDENTITY_OK"
            ]
        )

    if INVESTABILITY.get(
        "require_market_price",
        True,
    ):
        conditions.append(
            result[
                "MARKET_DATA_OK"
            ]
        )

    if INVESTABILITY.get(
        "require_fundamental_data",
        True,
    ):
        conditions.append(
            result[
                "FUNDAMENTAL_COMPLETE"
            ]
        )

    if INVESTABILITY.get(
        "require_pit_compliance",
        True,
    ):
        conditions.append(
            result[
                "PIT_OK"
            ]
        )

    conditions.append(
        result[
            "FINANCIAL_MODEL_OK"
        ]
    )

    if not conditions:

        raise InvestabilityIntegrityError(
            "FAIL-SAFE: nenhuma regra de investabilidade configurada."
        )

    investable = pd.Series(
        True,
        index=result.index,
        dtype=bool,
    )

    for condition in conditions:

        investable &= (
            condition
            .fillna(False)
            .astype(bool)
        )

    result[
        "INVESTABLE"
    ] = investable

    result[
        "INVESTABILITY_STATUS"
    ] = np.where(
        result[
            "INVESTABLE"
        ],
        STATUS_PASS,
        STATUS_FAIL,
    )

    # -------------------------------------------------------------------------
    # MOTIVOS
    # -------------------------------------------------------------------------

    def exclusion_reasons(
        row,
    ) -> str:

        reasons = []

        if not bool(
            row[
                "IDENTITY_OK"
            ]
        ):
            reasons.append(
                REASON_IDENTITY
            )

        if not bool(
            row[
                "MARKET_DATA_OK"
            ]
        ):
            reasons.append(
                REASON_MARKET
            )

        if not bool(
            row[
                "FUNDAMENTAL_COMPLETE"
            ]
        ):
            reasons.append(
                REASON_FUNDAMENTALS
            )

        if not bool(
            row[
                "PIT_OK"
            ]
        ):
            reasons.append(
                REASON_PIT
            )

        if not bool(
            row[
                "FINANCIAL_MODEL_OK"
            ]
        ):
            reasons.append(
                REASON_FINANCIAL
            )

        if not reasons:
            return REASON_OK

        return "|".join(
            reasons
        )

    result[
        "INVESTABILITY_REASON"
    ] = (
        result.apply(
            exclusion_reasons,
            axis=1,
        )
    )

    # -------------------------------------------------------------------------
    # METADADOS
    # -------------------------------------------------------------------------

    result[
        "FORMATION_DATE"
    ] = context.formation_date

    result[
        "ACCOUNTING_CUTOFF"
    ] = context.accounting_cutoff

    result[
        "MARKET_CUTOFF"
    ] = context.market_cutoff

    result[
        "FUTURE_RETURN_USED"
    ] = False

    result = (
        result
        .sort_values(
            "ISSUER_ID",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
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
        raise InvestabilityIntegrityError(
            "FAIL-SAFE: emissor duplicado após Investability Gate."
        )

    if result.empty:
        raise InvestabilityIntegrityError(
            "FAIL-SAFE: resultado do Investability Gate vazio."
        )

    return result


# =============================================================================
# AUDITORIA
# =============================================================================

def audit_investability(
    result: pd.DataFrame,
) -> dict:

    required = {
        "ISSUER_ID",
        "INVESTABLE",
        "INVESTABILITY_STATUS",
        "INVESTABILITY_REASON",
        "IDENTITY_OK",
        "MARKET_DATA_OK",
        "FUNDAMENTAL_COMPLETE",
        "PIT_OK",
        "FINANCIAL_MODEL_OK",
    }

    _require_columns(
        result,
        required,
        "investability_result",
    )

    total = int(
        len(result)
    )

    passed = int(
        result[
            "INVESTABLE"
        ]
        .sum()
    )

    failed = (
        total
        - passed
    )

    reasons = (
        result.loc[
            ~result[
                "INVESTABLE"
            ],
            "INVESTABILITY_REASON",
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    coverage = {}

    if (
        "FUNDAMENTAL_COVERAGE"
        in result.columns
    ):

        coverage = {
            "mean":
                float(
                    result[
                        "FUNDAMENTAL_COVERAGE"
                    ]
                    .mean()
                ),

            "median":
                float(
                    result[
                        "FUNDAMENTAL_COVERAGE"
                    ]
                    .median()
                ),
        }

    return {
        "status": "OK",
        "issuers_total": total,
        "issuers_passed": passed,
        "issuers_failed": failed,
        "pass_rate": (
            passed / total
            if total
            else 0.0
        ),
        "failure_reasons": reasons,
        "fundamental_coverage": coverage,
        "future_return_used": False,
    }


# =============================================================================
# SALVAR RESULTADOS
# =============================================================================

def save_investability(
    result: pd.DataFrame,
    context: PITContext,
) -> dict:

    audit = audit_investability(
        result
    )

    date_tag = (
        context
        .formation_date
        .strftime(
            "%Y%m%d"
        )
    )

    full_path = (
        INVESTABILITY_DIR
        / f"investability_{date_tag}.csv"
    )

    passed_path = (
        INVESTABILITY_DIR
        / f"investable_issuers_{date_tag}.csv"
    )

    rejected_path = (
        INVESTABILITY_DIR
        / f"investability_rejected_{date_tag}.csv"
    )

    manifest_path = (
        INVESTABILITY_DIR
        / f"investability_manifest_{date_tag}.json"
    )

    result.to_csv(
        full_path,
        index=False,
        encoding="utf-8-sig",
    )

    result.loc[
        result[
            "INVESTABLE"
        ]
    ].to_csv(
        passed_path,
        index=False,
        encoding="utf-8-sig",
    )

    result.loc[
        ~result[
            "INVESTABLE"
        ]
    ].to_csv(
        rejected_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "engine":
            "INVESTABILITY_ENGINE",

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
            "full":
                str(
                    full_path
                ),

            "passed":
                str(
                    passed_path
                ),

            "rejected":
                str(
                    rejected_path
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
        "full":
            full_path,

        "passed":
            passed_path,

        "rejected":
            rejected_path,

        "manifest":
            manifest_path,
    }


# =============================================================================
# SELF-TEST
# =============================================================================

def _self_test():

    issuer_universe = pd.DataFrame(
        {
            "ISSUER_ID": [
                "1",
                "2",
                "3",
            ],

            "TICKERS": [
                "AAAA3",
                "BBBB3",
                "CCCC3",
            ],

            "N_SECURITIES": [
                1,
                1,
                1,
            ],

            "IS_FINANCIAL": [
                False,
                False,
                True,
            ],

            "FUNDAMENTAL_ELIGIBLE": [
                True,
                True,
                False,
            ],
        }
    )

    security_universe = pd.DataFrame(
        {
            "ISSUER_ID": [
                "1",
                "2",
                "3",
            ],

            "TICKER": [
                "AAAA3",
                "BBBB3",
                "CCCC3",
            ],

            "FORMATION_PRICE": [
                10.0,
                20.0,
                30.0,
            ],

            "FORMATION_PRICE_DATE": [
                "2025-12-30",
                "2025-12-30",
                "2025-12-30",
            ],

            "IDENTITY_RESOLVED": [
                True,
                True,
                True,
            ],
        }
    )

    fundamentals = pd.DataFrame(
        {
            "ISSUER_ID": [
                "1",
                "2",
                "3",
            ],

            "ATIVO_TOTAL": [
                1000.0,
                2000.0,
                3000.0,
            ],

            "PL": [
                500.0,
                None,
                1500.0,
            ],

            "RECEITA": [
                800.0,
                1000.0,
                2000.0,
            ],

            "ACCOUNTING_CUTOFF": [
                "2025-09-30",
                "2025-09-30",
                "2025-09-30",
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

    result = run_investability_gate(
        issuer_universe=issuer_universe,
        security_universe=security_universe,
        fundamentals=fundamentals,
        context=FakeContext(),
    )

    company_1 = (
        result.loc[
            result[
                "ISSUER_ID"
            ]
            == "1"
        ]
        .iloc[0]
    )

    company_2 = (
        result.loc[
            result[
                "ISSUER_ID"
            ]
            == "2"
        ]
        .iloc[0]
    )

    company_3 = (
        result.loc[
            result[
                "ISSUER_ID"
            ]
            == "3"
        ]
        .iloc[0]
    )

    if not bool(
        company_1[
            "INVESTABLE"
        ]
    ):
        raise InvestabilityError(
            "SELF-TEST: empresa 1 deveria passar."
        )

    if bool(
        company_2[
            "INVESTABLE"
        ]
    ):
        raise InvestabilityError(
            "SELF-TEST: empresa 2 deveria falhar "
            "por fundamentos incompletos."
        )

    if bool(
        company_3[
            "INVESTABLE"
        ]
    ):
        raise InvestabilityError(
            "SELF-TEST: instituição financeira "
            "não deveria entrar no modelo padrão."
        )

    return True


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("B3 STOCK SELECTOR — INVESTABILITY ENGINE")
    print("=" * 72)

    _self_test()

    print("Self-test: OK")
    print("Modelo: GATE")
    print("Identidade: VALIDADA")
    print("Mercado: VALIDADO")
    print("Fundamentos mínimos: VALIDADOS")
    print("PIT: OBRIGATÓRIO")
    print("Financeiras: MODELO SEPARADO")
    print("Retorno futuro: NÃO UTILIZADO")
    print("Score de oportunidade: NÃO CALCULADO")

    print("=" * 72)
