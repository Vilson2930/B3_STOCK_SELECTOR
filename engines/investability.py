# =============================================================================
# B3 STOCK SELECTOR
# Arquivo: engines/investability.py
#
# INVESTABILITY ENGINE
#
# OBJETIVO:
# Aplicar gates mínimos de integridade antes que um emissor possa seguir para
# Fundamentals → Quality / Turnaround / Valuation.
#
# NÃO FAZ:
# - ranking;
# - scoring;
# - descoberta de fatores;
# - uso de retorno futuro.
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
    from config import OUTPUT_DIR

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


INVESTABILITY_CONFIG = getattr(
    project_config,
    "INVESTABILITY",
    {},
)

if not isinstance(
    INVESTABILITY_CONFIG,
    dict,
):
    INVESTABILITY_CONFIG = {}


EXCLUDE_FINANCIALS = bool(
    getattr(
        project_config,
        "EXCLUDE_FINANCIALS_FROM_STANDARD_FUNDAMENTAL_MODEL",
        True,
    )
)


# =============================================================================
# DIRETÓRIO
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
    pass


class InvestabilityIntegrityError(InvestabilityError):
    pass


# =============================================================================
# FUNDAMENTOS MÍNIMOS
# =============================================================================

DEFAULT_REQUIRED_FUNDAMENTALS = (
    "ATIVO_TOTAL",
    "PL",
    "RECEITA",
)


# =============================================================================
# PROTEÇÃO CONTRA FUTURO
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

    columns = list(
        columns
    )

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise InvestabilityIntegrityError(
            f"FAIL-SAFE: {name} sem colunas obrigatórias: "
            f"{missing}"
        )


def _normalize_issuer_id(
    value,
):

    if pd.isna(
        value
    ):
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
                int(
                    numeric
                )
            )

    except Exception:
        pass

    return value


def assert_no_future_information(
    df: pd.DataFrame,
) -> None:

    forbidden = []

    for column in df.columns:

        normalized = str(
            column
        ).upper()

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
        raise InvestabilityIntegrityError(
            "FAIL-SAFE: informação futura detectada: "
            f"{forbidden}"
        )




def _strict_bool_series(values: pd.Series, *, field: str) -> pd.Series:
    """Converte booleanos explícitos; valores desconhecidos ficam pendentes."""
    def convert(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if pd.isna(value):
            return pd.NA
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("true", "1", "sim"):
                return True
            if normalized in ("false", "0", "não", "nao"):
                return False
        return pd.NA
    return values.map(convert).astype("boolean")


def _publication_evidence(df: pd.DataFrame, context: PITContext) -> pd.Series:
    """Sem publicação documental verificada não há PASS PIT."""
    date_cols = ("DT_PUBLICACAO_CVM_VERIFICADA", "PUBLICATION_DATE_VERIFIED")
    doc_cols = ("PROTOCOLO_CVM_DOCUMENTO", "CVM_DOCUMENT_ID")
    date_col = next((x for x in date_cols if x in df.columns), None)
    doc_col = next((x for x in doc_cols if x in df.columns), None)
    if date_col is None or doc_col is None:
        return pd.Series(False, index=df.index)
    publication = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    cutoff = pd.Timestamp(context.accounting_cutoff)
    formation = pd.Timestamp(context.formation_date)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    if formation.tzinfo is None:
        formation = formation.tz_localize("UTC")
    docs = df[doc_col].astype("string")
    return (
        publication.notna()
        & publication.le(cutoff)
        & publication.le(formation)
        & docs.notna()
        & docs.str.strip().ne("").fillna(False)
    ).fillna(False)


# =============================================================================
# PREPARAÇÃO DO UNIVERSO
# =============================================================================

def prepare_issuer_universe(
    issuer_universe: pd.DataFrame,
) -> pd.DataFrame:

    required = (
        "ISSUER_ID",
        "TICKERS",
        "N_SECURITIES",
        "IS_FINANCIAL",
        "FUNDAMENTAL_ELIGIBLE",
    )

    _require_columns(
        issuer_universe,
        required,
        "issuer_universe",
    )

    assert_no_future_information(
        issuer_universe
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
            "FAIL-SAFE: ISSUER_ID ausente."
        )

    if (
        result[
            "ISSUER_ID"
        ]
        .duplicated()
        .any()
    ):
        raise InvestabilityIntegrityError(
            "FAIL-SAFE: emissor duplicado."
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
        (
            "ISSUER_ID",
        ),
        "fundamentals",
    )

    assert_no_future_information(
        fundamentals
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

    if (
        result[
            "ISSUER_ID"
        ]
        .duplicated()
        .any()
    ):
        raise InvestabilityIntegrityError(
            "FAIL-SAFE: fundamentals possui "
            "mais de uma linha por emissor."
        )

    return result


# =============================================================================
# PIT DOS FUNDAMENTOS
# =============================================================================

def validate_fundamental_pit(
    df: pd.DataFrame,
    context: PITContext,
) -> pd.Series:
    """A data de referência contábil nunca substitui a publicação oficial."""
    return _publication_evidence(df, context)


# =============================================================================
# COBERTURA FUNDAMENTAL
# =============================================================================

def fundamental_coverage(
    df: pd.DataFrame,
    required_fields: Iterable[str] = DEFAULT_REQUIRED_FUNDAMENTALS,
) -> pd.Series:
    """
    CORREÇÃO IMPORTANTE:

    required_fields originalmente é uma tuple:
        ("ATIVO_TOTAL", "PL", "RECEITA")

    Para selecionar múltiplas colunas no pandas é necessário utilizar LIST:
        df[["ATIVO_TOTAL", "PL", "RECEITA"]]

    e não:
        df[("ATIVO_TOTAL", "PL", "RECEITA")]
    """

    required_fields = list(
        required_fields
    )

    _require_columns(
        df,
        required_fields,
        "fundamentals",
    )

    coverage = (
        df[
            required_fields
        ]
        .notna()
        .all(
            axis=1
        )
    )

    return coverage


# =============================================================================
# STATUS DE MERCADO / IDENTIDADE
# =============================================================================

def market_identity_status(
    security_universe: pd.DataFrame,
    context: Optional[PITContext] = None,
) -> pd.DataFrame:

    required = (
        "ISSUER_ID",
        "TICKER",
        "FORMATION_PRICE",
        "FORMATION_PRICE_DATE",
        "IDENTITY_RESOLVED",
    )

    _require_columns(
        security_universe,
        required,
        "security_universe",
    )

    assert_no_future_information(
        security_universe
    )

    temp = (
        security_universe
        .copy()
    )

    temp[
        "ISSUER_ID"
    ] = (
        temp[
            "ISSUER_ID"
        ]
        .map(
            _normalize_issuer_id
        )
    )

    temp[
        "_MARKET_VALID"
    ] = (
        pd.to_numeric(
            temp[
                "FORMATION_PRICE"
            ],
            errors="coerce",
        )
        .gt(
            0
        )
        &
        pd.to_datetime(
            temp[
                "FORMATION_PRICE_DATE"
            ],
            errors="coerce",
        )
        .notna()
    )

    temp["_MARKET_VALID"] = (
        temp["_MARKET_VALID"]
        & pd.to_datetime(
            temp["FORMATION_PRICE_DATE"], errors="coerce", utc=True
        ).le(pd.Timestamp(context.market_cutoff).tz_localize("UTC")
             if pd.Timestamp(context.market_cutoff).tzinfo is None
             else pd.Timestamp(context.market_cutoff).tz_convert("UTC"))
    ) if context is not None else temp["_MARKET_VALID"]

    temp[
        "_IDENTITY_VALID"
    ] = (
        temp[
            "IDENTITY_RESOLVED"
        ]
        .pipe(_strict_bool_series, field="IDENTITY_RESOLVED")
        .fillna(False)
        .astype(bool)
    )

    result = (
        temp
        .groupby(
            "ISSUER_ID",
            as_index=False,
        )
        .agg(
            MARKET_VALID=(
                "_MARKET_VALID",
                "any",
            ),
            IDENTITY_VALID=(
                "_IDENTITY_VALID",
                "all",
            ),
        )
    )

    return result


# =============================================================================
# GATE PRINCIPAL
# =============================================================================

def run_investability_gate(
    issuer_universe: pd.DataFrame,
    security_universe: pd.DataFrame,
    fundamentals: pd.DataFrame,
    context: PITContext,
    required_fundamentals: Iterable[str] = DEFAULT_REQUIRED_FUNDAMENTALS,
) -> pd.DataFrame:

    issuers = prepare_issuer_universe(
        issuer_universe
    )

    fundamental = prepare_fundamentals(
        fundamentals
    )

    market = market_identity_status(
        security_universe,
        context,
    )

    # =========================================================================
    # COBERTURA
    # =========================================================================

    fundamental[
        "FUNDAMENTAL_COMPLETE"
    ] = fundamental_coverage(
        fundamental,
        required_fundamentals,
    )

    fundamental[
        "FUNDAMENTAL_PIT_VALID"
    ] = validate_fundamental_pit(
        fundamental,
        context,
    )

    # =========================================================================
    # MERGE
    # =========================================================================

    result = (
        issuers
        .merge(
            market,
            on="ISSUER_ID",
            how="left",
            validate="one_to_one",
        )
        .merge(
            fundamental,
            on="ISSUER_ID",
            how="left",
            validate="one_to_one",
            suffixes=(
                "",
                "_FUND",
            ),
        )
    )

    # =========================================================================
    # NORMALIZAÇÃO
    # =========================================================================

    for column in (
        "MARKET_VALID",
        "IDENTITY_VALID",
        "FUNDAMENTAL_COMPLETE",
        "FUNDAMENTAL_PIT_VALID",
        "FUNDAMENTAL_ELIGIBLE",
        "IS_FINANCIAL",
    ):

        if column not in result.columns:
            result[
                column
            ] = False

        result[column] = (
            _strict_bool_series(result[column], field=column)
            .fillna(False)
            .astype(bool)
        )

    # =========================================================================
    # REGRAS CONFIGURÁVEIS
    # =========================================================================

    require_identity = bool(
        INVESTABILITY_CONFIG.get(
            "require_identity",
            True,
        )
    )

    require_market = bool(
        INVESTABILITY_CONFIG.get(
            "require_market",
            True,
        )
    )

    require_fundamentals = bool(
        INVESTABILITY_CONFIG.get(
            "require_fundamentals",
            True,
        )
    )

    require_pit = bool(
        INVESTABILITY_CONFIG.get(
            "require_pit",
            True,
        )
    )

    # =========================================================================
    # CONDIÇÕES
    # =========================================================================

    conditions = []

    if require_identity:
        conditions.append(
            result[
                "IDENTITY_VALID"
            ]
        )

    if require_market:
        conditions.append(
            result[
                "MARKET_VALID"
            ]
        )

    if require_fundamentals:
        conditions.append(
            result[
                "FUNDAMENTAL_COMPLETE"
            ]
        )

    if require_pit:
        conditions.append(
            result[
                "FUNDAMENTAL_PIT_VALID"
            ]
        )

    # Respeita a elegibilidade fundamental já definida pelo Universe Engine.
    conditions.append(
        result[
            "FUNDAMENTAL_ELIGIBLE"
        ]
    )

    # Financeiras permanecem fora do modelo fundamental padrão.
    if EXCLUDE_FINANCIALS:

        conditions.append(
            ~result[
                "IS_FINANCIAL"
            ]
        )

    if not conditions:

        investable = pd.Series(
            True,
            index=result.index,
        )

    else:

        investable = conditions[0].copy()

        for condition in conditions[1:]:

            investable = (
                investable
                &
                condition
            )

    result[
        "INVESTABLE"
    ] = investable.astype(
        bool
    )

    # =========================================================================
    # MOTIVOS
    # =========================================================================

    def determine_reason(
        row,
    ) -> str:

        if (
            require_identity
            and
            not row[
                "IDENTITY_VALID"
            ]
        ):
            return "IDENTITY_INVALID"

        if (
            require_market
            and
            not row[
                "MARKET_VALID"
            ]
        ):
            return "MARKET_INVALID"

        if (
            require_fundamentals
            and
            not row[
                "FUNDAMENTAL_COMPLETE"
            ]
        ):
            return "FUNDAMENTAL_INCOMPLETE"

        if (
            require_pit
            and
            not row[
                "FUNDAMENTAL_PIT_VALID"
            ]
        ):
            return "PIT_INVALID"

        if not row[
            "FUNDAMENTAL_ELIGIBLE"
        ]:
            return "FUNDAMENTAL_MODEL_INELIGIBLE"

        if (
            EXCLUDE_FINANCIALS
            and
            row[
                "IS_FINANCIAL"
            ]
        ):
            return "FINANCIAL_EXCLUDED_FROM_STANDARD_MODEL"

        return "OK"

    result[
        "INVESTABILITY_REASON"
    ] = result.apply(
        determine_reason,
        axis=1,
    )

    result[
        "INVESTABILITY_STATUS"
    ] = np.where(
        result[
            "INVESTABLE"
        ],
        "PASS",
        "FAIL",
    )

    # =========================================================================
    # STATUS CIENTÍFICO: NÃO CONFUNDIR TRIAGEM COM APROVAÇÃO
    # =========================================================================
    result["LEGACY_TECHNICAL_GATE_PASS"] = result["INVESTABLE"].copy()
    result["STUDY_34_CRITERIA_VERIFIED"] = False
    result["STUDY_34_STATUS"] = "PENDENTE"
    result["STUDY_34_REASON"] = (
        "Matriz das 34 etapas e evidências de CAPEX, FCL, dívida líquida, "
        "ROIC/WACC e valor justo ainda não validadas integralmente."
    )
    # Não habilitar a seleção com a elegibilidade do robô antigo.
    result["INVESTABLE"] = False
    result["INVESTABILITY_STATUS"] = "PENDING"
    result["INVESTABILITY_REASON"] = "STUDY_34_EVIDENCE_PENDING"

    # =========================================================================
    # METADADOS
    # =========================================================================

    result[
        "FORMATION_DATE_INVESTABILITY"
    ] = context.formation_date

    result[
        "ACCOUNTING_CUTOFF_INVESTABILITY"
    ] = context.accounting_cutoff

    result[
        "MARKET_CUTOFF_INVESTABILITY"
    ] = context.market_cutoff

    result[
        "FUTURE_RETURN_USED"
    ] = False

    # =========================================================================
    # FAIL-SAFE FINAL
    # =========================================================================

    if (
        result[
            "ISSUER_ID"
        ]
        .duplicated()
        .any()
    ):
        raise InvestabilityIntegrityError(
            "FAIL-SAFE: emissor duplicado "
            "após Investability Gate."
        )

    assert_no_future_information(
        result
    )

    return result


# =============================================================================
# ALIAS OPERACIONAL
# =============================================================================

def run_investability_engine(
    issuer_universe: pd.DataFrame,
    security_universe: pd.DataFrame,
    fundamentals: pd.DataFrame,
    context: PITContext,
    required_fundamentals: Iterable[str] = DEFAULT_REQUIRED_FUNDAMENTALS,
) -> pd.DataFrame:

    return run_investability_gate(
        issuer_universe=issuer_universe,
        security_universe=security_universe,
        fundamentals=fundamentals,
        context=context,
        required_fundamentals=required_fundamentals,
    )


# =============================================================================
# AUDITORIA
# =============================================================================

def audit_investability(
    result: pd.DataFrame,
) -> dict:

    _require_dataframe(
        result,
        "investability_result",
    )

    _require_columns(
        result,
        (
            "INVESTABLE",
            "INVESTABILITY_REASON",
        ),
        "investability_result",
    )

    investable = (
        result[
            "INVESTABLE"
        ]
        .fillna(False)
        .astype(bool)
    )

    reasons = (
        result[
            "INVESTABILITY_REASON"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    return {
        "status":
            "OK",

        "issuers_total":
            int(
                len(
                    result
                )
            ),

        "investable":
            int(
                investable.sum()
            ),

        "rejected":
            int(
                (
                    ~investable
                ).sum()
            ),

        "investable_rate":
            float(
                investable.mean()
            ),

        "reasons": {
            str(key):
                int(value)

            for key, value
            in reasons.items()
        },

        "future_return_used":
            False,
    }


# =============================================================================
# SALVAR
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

    csv_path = (
        INVESTABILITY_DIR
        / f"investability_{date_tag}.csv"
    )

    manifest_path = (
        INVESTABILITY_DIR
        / f"investability_manifest_{date_tag}.json"
    )

    result.to_csv(
        csv_path,
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
            "csv":
                str(
                    csv_path
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
        "csv":
            csv_path,

        "manifest":
            manifest_path,
    }


# =============================================================================
# SELF-TEST
# =============================================================================

def _self_test():

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

    issuers = pd.DataFrame(
        {
            "ISSUER_ID": [
                "1",
                "2",
            ],

            "TICKERS": [
                "AAAA3",
                "BBBB3",
            ],

            "N_SECURITIES": [
                1,
                1,
            ],

            "IS_FINANCIAL": [
                False,
                False,
            ],

            "FUNDAMENTAL_ELIGIBLE": [
                True,
                True,
            ],
        }
    )

    securities = pd.DataFrame(
        {
            "ISSUER_ID": [
                "1",
                "2",
            ],

            "TICKER": [
                "AAAA3",
                "BBBB3",
            ],

            "FORMATION_PRICE": [
                10.0,
                20.0,
            ],

            "FORMATION_PRICE_DATE": [
                "2025-12-30",
                "2025-12-30",
            ],

            "IDENTITY_RESOLVED": [
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
            ],

            "ATIVO_TOTAL": [
                1000.0,
                2000.0,
            ],

            "PL": [
                500.0,
                np.nan,
            ],

            "RECEITA": [
                800.0,
                1500.0,
            ],

            "ACCOUNTING_CUTOFF": [
                "2025-09-30",
                "2025-09-30",
            ],
        }
    )

    result = run_investability_gate(
        issuer_universe=issuers,
        security_universe=securities,
        fundamentals=fundamentals,
        context=FakeContext(),
    )

    indexed = (
        result
        .set_index(
            "ISSUER_ID"
        )
    )

    if indexed["INVESTABLE"].any():
        raise InvestabilityError("SELF-TEST: aprovação sem evidências das 34 etapas.")
    if not indexed["STUDY_34_STATUS"].eq("PENDENTE").all():
        raise InvestabilityError("SELF-TEST: pendência não registrada.")
    if indexed["FUNDAMENTAL_PIT_VALID"].any():
        raise InvestabilityError("SELF-TEST: data de referência não é publicação.")

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
    print("Cobertura fundamental: OK")
    print("PIT: bloqueado sem publicação CVM comprovada")
    print("Identidade: requer evidência documental")
    print("Mercado: testes de integração pendentes")
    print("34 etapas: aprovação pendente")
    print("Retorno futuro: NÃO UTILIZADO")
    print("=" * 72)
