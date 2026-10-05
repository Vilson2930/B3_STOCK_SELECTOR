# =============================================================================
# B3 STOCK SELECTOR
# Arquivo: engines/universe.py
#
# UNIVERSE ENGINE
#
# RESPONSABILIDADE:
# - Receber o universo de mercado já filtrado pelo Market Engine
# - Validar identidade do ticker
# - Fazer ligação TICKER -> EMISSOR / CVM
# - Evitar duplicidade econômica entre classes da mesma empresa
# - Preservar simultaneamente:
#       1) universo por security/ticker
#       2) universo por issuer/empresa
# - Aplicar exclusões estruturais
# - Produzir auditoria completa do universo
#
# NÃO FAZ:
# - Download CVM
# - Download COTAHIST
# - Cálculo de fundamentos
# - Quality Score
# - Turnaround Score
# - Valuation
# - Ranking
# - Retorno futuro
#
# PRINCÍPIO CIENTÍFICO:
# - Ticker e empresa NÃO são a mesma unidade.
# - Diferentes classes da mesma companhia devem permanecer identificáveis,
#   mas a análise fundamental deve contar o emissor apenas uma vez.
# =============================================================================

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

try:
    from config import (
        DATA_DIR,
        OUTPUT_DIR,
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

UNIVERSE_DIR = OUTPUT_DIR / "universe"

UNIVERSE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# EXCEÇÕES
# =============================================================================

class UniverseError(RuntimeError):
    """Erro geral do Universe Engine."""


class UniverseIdentityError(UniverseError):
    """Erro na identidade ticker -> emissor."""


class UniverseIntegrityError(UniverseError):
    """Erro de integridade do universo."""


# =============================================================================
# COLUNAS CANÔNICAS
# =============================================================================

SECURITY_REQUIRED_COLUMNS = {
    "TICKER",
    "FORMATION_PRICE",
    "FORMATION_PRICE_DATE",
    "TRADING_SESSIONS",
}

IDENTITY_REQUIRED_COLUMNS = {
    "TICKER",
    "ISSUER_ID",
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

        raise UniverseIntegrityError(
            f"{name} não é DataFrame."
        )

    if df.empty:

        raise UniverseIntegrityError(
            f"FAIL-SAFE: {name} está vazio."
        )


def _require_columns(
    df: pd.DataFrame,
    columns,
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

        raise UniverseIntegrityError(
            f"FAIL-SAFE: {name} sem colunas obrigatórias: "
            f"{sorted(missing)}"
        )


# =============================================================================
# NORMALIZAÇÃO DE TICKER
# =============================================================================

def normalize_ticker(
    value,
) -> Optional[str]:

    if pd.isna(value):
        return None

    ticker = (
        str(value)
        .strip()
        .upper()
    )

    if not ticker:
        return None

    return ticker


# =============================================================================
# NORMALIZAÇÃO DO ISSUER ID
# =============================================================================

def normalize_issuer_id(
    value,
):

    if pd.isna(value):
        return None

    value = str(
        value
    ).strip()

    if not value:
        return None

    # Evita problemas comuns de CSV:
    # 22187.0 -> 22187
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
# PREPARAÇÃO DA IDENTIDADE
# =============================================================================

def prepare_identity_map(
    identity_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Padroniza o bridge TICKER -> ISSUER.

    Colunas mínimas:
        TICKER
        ISSUER_ID

    Colunas opcionais úteis:
        COMPANY_NAME
        CNPJ
        SECTOR
        SECTOR_BUCKET
        FINANCIAL_SPECIAL
    """

    _require_columns(
        identity_df,
        IDENTITY_REQUIRED_COLUMNS,
        "identity_df",
    )

    result = (
        identity_df
        .copy()
    )

    result[
        "TICKER"
    ] = (
        result[
            "TICKER"
        ]
        .map(
            normalize_ticker
        )
    )

    result[
        "ISSUER_ID"
    ] = (
        result[
            "ISSUER_ID"
        ]
        .map(
            normalize_issuer_id
        )
    )

    invalid = (
        result["TICKER"].isna()
        |
        result["ISSUER_ID"].isna()
    )

    if invalid.any():

        result = result.loc[
            ~invalid
        ].copy()

    if result.empty:

        raise UniverseIdentityError(
            "FAIL-SAFE: bridge de identidade vazio "
            "após normalização."
        )

    # -------------------------------------------------------------------------
    # TICKER NÃO PODE APONTAR PARA DOIS EMISSORES
    # -------------------------------------------------------------------------

    ticker_issuer_count = (
        result
        .groupby(
            "TICKER"
        )[
            "ISSUER_ID"
        ]
        .nunique()
    )

    conflicts = (
        ticker_issuer_count[
            ticker_issuer_count > 1
        ]
    )

    if not conflicts.empty:

        examples = (
            conflicts
            .head(20)
            .index
            .tolist()
        )

        raise UniverseIdentityError(
            "FAIL-SAFE: ticker associado a múltiplos "
            f"emissores. Exemplos={examples}"
        )

    # -------------------------------------------------------------------------
    # REMOVE DUPLICATAS EXATAS
    # -------------------------------------------------------------------------

    result = (
        result
        .sort_values(
            [
                "TICKER",
                "ISSUER_ID",
            ],
            kind="mergesort",
        )
        .drop_duplicates(
            subset=[
                "TICKER",
                "ISSUER_ID",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return result


# =============================================================================
# PREPARAÇÃO DO UNIVERSO DE MERCADO
# =============================================================================

def prepare_market_universe(
    market_df: pd.DataFrame,
) -> pd.DataFrame:

    _require_columns(
        market_df,
        SECURITY_REQUIRED_COLUMNS,
        "market_df",
    )

    result = (
        market_df
        .copy()
    )

    result[
        "TICKER"
    ] = (
        result[
            "TICKER"
        ]
        .map(
            normalize_ticker
        )
    )

    if (
        result["TICKER"]
        .isna()
        .any()
    ):

        raise UniverseIntegrityError(
            "FAIL-SAFE: ticker ausente no universo de mercado."
        )

    # Um ticker só pode aparecer uma vez no snapshot.
    duplicated = (
        result[
            "TICKER"
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicated.any():

        examples = (
            result.loc[
                duplicated,
                "TICKER",
            ]
            .head(20)
            .tolist()
        )

        raise UniverseIntegrityError(
            "FAIL-SAFE: ticker duplicado no snapshot "
            f"de mercado. Exemplos={examples}"
        )

    prices = pd.to_numeric(
        result[
            "FORMATION_PRICE"
        ],
        errors="coerce",
    )

    invalid_price = (
        prices.isna()
        |
        (prices <= 0)
    )

    if invalid_price.any():

        examples = (
            result.loc[
                invalid_price,
                "TICKER",
            ]
            .head(20)
            .tolist()
        )

        raise UniverseIntegrityError(
            "FAIL-SAFE: preço de formação inválido. "
            f"Exemplos={examples}"
        )

    result[
        "FORMATION_PRICE"
    ] = prices

    return result


# =============================================================================
# BRIDGE SECURITY -> ISSUER
# =============================================================================

def attach_issuer_identity(
    market_df: pd.DataFrame,
    identity_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Faz a ligação entre cada ticker elegível e seu emissor.

    Tickers sem emissor são preservados inicialmente para auditoria,
    mas NÃO recebem autorização fundamental.
    """

    market = prepare_market_universe(
        market_df
    )

    identity = prepare_identity_map(
        identity_df
    )

    merged = market.merge(
        identity,
        on="TICKER",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_IDENTITY",
        ),
    )

    merged[
        "IDENTITY_RESOLVED"
    ] = (
        merged[
            "ISSUER_ID"
        ]
        .notna()
    )

    return merged


# =============================================================================
# CLASSIFICAÇÃO FINANCEIRA
# =============================================================================

def identify_financial_issuer(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Identifica instituições financeiras quando houver informação
    setorial disponível.

    Não inventa classificação.

    Se nenhuma coluna adequada existir, retorna False para todos e
    registra posteriormente que a classificação não estava disponível.
    """

    index = df.index

    # -------------------------------------------------------------------------
    # PRIORIDADE 1: flag explícito
    # -------------------------------------------------------------------------

    if (
        "FINANCIAL_SPECIAL"
        in df.columns
    ):

        raw = (
            df[
                "FINANCIAL_SPECIAL"
            ]
        )

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
            .astype(str)
            .str.strip()
            .str.upper()
        )

        return normalized.isin(
            {
                "1",
                "TRUE",
                "SIM",
                "YES",
                "FINANCIAL",
                "FINANCEIRO",
            }
        )

    # -------------------------------------------------------------------------
    # PRIORIDADE 2: bucket setorial
    # -------------------------------------------------------------------------

    candidate_columns = [
        "SECTOR_BUCKET",
        "SECTOR",
        "FCA_SETOR_ATIVIDADE",
    ]

    for column in candidate_columns:

        if column not in df.columns:
            continue

        normalized = (
            df[
                column
            ]
            .fillna("")
            .astype(str)
            .str.upper()
        )

        financial = (
            normalized.str.contains(
                "FINANCEIR",
                regex=False,
            )
            |
            normalized.str.contains(
                "BANCO",
                regex=False,
            )
            |
            normalized.str.contains(
                "SEGURO",
                regex=False,
            )
        )

        return financial

    return pd.Series(
        False,
        index=index,
        dtype=bool,
    )


# =============================================================================
# UNIVERSO SECURITY-LEVEL
# =============================================================================

def build_security_universe(
    market_df: pd.DataFrame,
    identity_df: pd.DataFrame,
    context: PITContext,
) -> pd.DataFrame:
    """
    Universo por ticker/security.

    Mantém todas as classes elegíveis separadas.

    Exemplo:
        UNIP3
        UNIP5
        UNIP6

    permanecem três securities, embora pertençam ao mesmo emissor.
    """

    result = attach_issuer_identity(
        market_df=market_df,
        identity_df=identity_df,
    )

    # -------------------------------------------------------------------------
    # AUDITORIA TEMPORAL
    # -------------------------------------------------------------------------

    dates = pd.to_datetime(
        result[
            "FORMATION_PRICE_DATE"
        ],
        errors="coerce",
    )

    if dates.isna().any():

        raise UniverseIntegrityError(
            "FAIL-SAFE: data de preço de formação inválida."
        )

    lookahead = (
        dates.dt.normalize()
        >
        context.market_cutoff
    )

    if lookahead.any():

        offenders = (
            result.loc[
                lookahead,
                [
                    "TICKER",
                    "FORMATION_PRICE_DATE",
                ],
            ]
            .head(20)
            .to_dict(
                orient="records"
            )
        )

        raise UniverseIntegrityError(
            "FAIL-SAFE: look-ahead no universo. "
            f"Exemplos={offenders}"
        )

    # -------------------------------------------------------------------------
    # FINANCEIRO
    # -------------------------------------------------------------------------

    result[
        "IS_FINANCIAL"
    ] = identify_financial_issuer(
        result
    )

    # -------------------------------------------------------------------------
    # ELEGIBILIDADE FUNDAMENTAL
    # -------------------------------------------------------------------------

    result[
        "FUNDAMENTAL_ELIGIBLE"
    ] = (
        result[
            "IDENTITY_RESOLVED"
        ]
    )

    if (
        EXCLUDE_FINANCIALS_FROM_STANDARD_FUNDAMENTAL_MODEL
    ):

        result[
            "FUNDAMENTAL_ELIGIBLE"
        ] = (
            result[
                "FUNDAMENTAL_ELIGIBLE"
            ]
            &
            ~result[
                "IS_FINANCIAL"
            ]
        )

    # -------------------------------------------------------------------------
    # MOTIVO DE EXCLUSÃO
    # -------------------------------------------------------------------------

    result[
        "EXCLUSION_REASON"
    ] = ""

    unresolved = (
        ~result[
            "IDENTITY_RESOLVED"
        ]
    )

    result.loc[
        unresolved,
        "EXCLUSION_REASON",
    ] = "ISSUER_ID_NOT_RESOLVED"

    if (
        EXCLUDE_FINANCIALS_FROM_STANDARD_FUNDAMENTAL_MODEL
    ):

        financial = (
            result[
                "IDENTITY_RESOLVED"
            ]
            &
            result[
                "IS_FINANCIAL"
            ]
        )

        result.loc[
            financial,
            "EXCLUSION_REASON",
        ] = "FINANCIAL_SPECIAL_MODEL_REQUIRED"

    result[
        "FORMATION_DATE"
    ] = context.formation_date

    result[
        "MARKET_CUTOFF"
    ] = context.market_cutoff

    result[
        "ACCOUNTING_CUTOFF"
    ] = context.accounting_cutoff

    result = (
        result
        .sort_values(
            "TICKER",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    return result


# =============================================================================
# UNIVERSO ISSUER-LEVEL
# =============================================================================

def build_issuer_universe(
    security_universe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Constrói o universo econômico por empresa.

    Esta é a unidade utilizada pelos motores fundamentais.

    Diferentes classes da mesma empresa contam UMA vez.
    """

    _require_columns(
        security_universe,
        {
            "TICKER",
            "ISSUER_ID",
            "IDENTITY_RESOLVED",
            "FUNDAMENTAL_ELIGIBLE",
            "IS_FINANCIAL",
        },
        "security_universe",
    )

    resolved = (
        security_universe.loc[
            security_universe[
                "IDENTITY_RESOLVED"
            ]
        ]
        .copy()
    )

    if resolved.empty:

        raise UniverseIdentityError(
            "FAIL-SAFE: nenhum emissor resolvido."
        )

    # -------------------------------------------------------------------------
    # GARANTIA DE CONSISTÊNCIA DENTRO DO EMISSOR
    # -------------------------------------------------------------------------

    issuer_rows = []

    for (
        issuer_id,
        group,
    ) in resolved.groupby(
        "ISSUER_ID",
        sort=True,
    ):

        tickers = sorted(
            group[
                "TICKER"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        row = {
            "ISSUER_ID": issuer_id,

            "TICKERS": ",".join(
                tickers
            ),

            "N_SECURITIES": int(
                len(tickers)
            ),

            "IS_FINANCIAL": bool(
                group[
                    "IS_FINANCIAL"
                ]
                .any()
            ),

            "FUNDAMENTAL_ELIGIBLE": bool(
                group[
                    "FUNDAMENTAL_ELIGIBLE"
                ]
                .any()
            ),
        }

        # ---------------------------------------------------------------------
        # CAMPOS OPCIONAIS DE IDENTIDADE
        # ---------------------------------------------------------------------

        optional_identity_columns = [
            "COMPANY_NAME",
            "CNPJ",
            "SECTOR",
            "SECTOR_BUCKET",
            "FCA_SETOR_ATIVIDADE",
        ]

        for column in optional_identity_columns:

            if column not in group.columns:
                continue

            values = (
                group[
                    column
                ]
                .dropna()
                .astype(str)
                .str.strip()
            )

            values = (
                values[
                    values != ""
                ]
                .unique()
                .tolist()
            )

            if len(values) > 1:

                raise UniverseIdentityError(
                    "FAIL-SAFE: identidade inconsistente "
                    f"para ISSUER_ID={issuer_id}, "
                    f"coluna={column}, valores={values}"
                )

            row[
                column
            ] = (
                values[0]
                if values
                else None
            )

        issuer_rows.append(
            row
        )

    issuer_universe = pd.DataFrame(
        issuer_rows
    )

    if issuer_universe.empty:

        raise UniverseIntegrityError(
            "FAIL-SAFE: universo de emissores vazio."
        )

    if (
        issuer_universe[
            "ISSUER_ID"
        ]
        .duplicated()
        .any()
    ):

        raise UniverseIntegrityError(
            "FAIL-SAFE: emissor duplicado no "
            "universo issuer-level."
        )

    issuer_universe = (
        issuer_universe
        .sort_values(
            "ISSUER_ID",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    return issuer_universe


# =============================================================================
# AUDITORIA SECURITY -> ISSUER
# =============================================================================

def audit_universe(
    security_universe: pd.DataFrame,
    issuer_universe: pd.DataFrame,
) -> dict:

    _require_dataframe(
        security_universe,
        "security_universe",
    )

    _require_dataframe(
        issuer_universe,
        "issuer_universe",
    )

    total_securities = int(
        len(
            security_universe
        )
    )

    resolved_securities = int(
        security_universe[
            "IDENTITY_RESOLVED"
        ]
        .sum()
    )

    unresolved_securities = (
        total_securities
        - resolved_securities
    )

    total_issuers = int(
        len(
            issuer_universe
        )
    )

    eligible_issuers = int(
        issuer_universe[
            "FUNDAMENTAL_ELIGIBLE"
        ]
        .sum()
    )

    financial_issuers = int(
        issuer_universe[
            "IS_FINANCIAL"
        ]
        .sum()
    )

    multiple_classes = int(
        (
            issuer_universe[
                "N_SECURITIES"
            ]
            > 1
        )
        .sum()
    )

    return {
        "status": "OK",

        "security_count":
            total_securities,

        "security_identity_resolved":
            resolved_securities,

        "security_identity_unresolved":
            unresolved_securities,

        "issuer_count":
            total_issuers,

        "fundamental_eligible_issuers":
            eligible_issuers,

        "financial_issuers":
            financial_issuers,

        "issuers_with_multiple_securities":
            multiple_classes,

        "future_return_used":
            False,
    }


# =============================================================================
# BUILD COMPLETO
# =============================================================================

def build_universe(
    market_df: pd.DataFrame,
    identity_df: pd.DataFrame,
    context: PITContext,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    """
    Executa o Universe Engine.

    Retorna:
        security_universe
        issuer_universe
        audit
    """

    security_universe = (
        build_security_universe(
            market_df=market_df,
            identity_df=identity_df,
            context=context,
        )
    )

    issuer_universe = (
        build_issuer_universe(
            security_universe
        )
    )

    audit = audit_universe(
        security_universe,
        issuer_universe,
    )

    return (
        security_universe,
        issuer_universe,
        audit,
    )


# =============================================================================
# SALVAR RESULTADOS
# =============================================================================

def save_universe(
    security_universe: pd.DataFrame,
    issuer_universe: pd.DataFrame,
    audit: dict,
    context: PITContext,
) -> dict:

    date_tag = (
        context
        .formation_date
        .strftime(
            "%Y%m%d"
        )
    )

    security_path = (
        UNIVERSE_DIR
        / f"security_universe_{date_tag}.csv"
    )

    issuer_path = (
        UNIVERSE_DIR
        / f"issuer_universe_{date_tag}.csv"
    )

    exclusions_path = (
        UNIVERSE_DIR
        / f"universe_exclusions_{date_tag}.csv"
    )

    manifest_path = (
        UNIVERSE_DIR
        / f"universe_manifest_{date_tag}.json"
    )

    # -------------------------------------------------------------------------
    # SECURITY
    # -------------------------------------------------------------------------

    security_universe.to_csv(
        security_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # ISSUER
    # -------------------------------------------------------------------------

    issuer_universe.to_csv(
        issuer_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # EXCLUSÕES
    # -------------------------------------------------------------------------

    exclusions = (
        security_universe.loc[
            ~security_universe[
                "FUNDAMENTAL_ELIGIBLE"
            ]
        ]
        .copy()
    )

    exclusions.to_csv(
        exclusions_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # MANIFESTO
    # -------------------------------------------------------------------------

    manifest = {
        "engine":
            "UNIVERSE_ENGINE",

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
            "security_universe":
                str(
                    security_path
                ),

            "issuer_universe":
                str(
                    issuer_path
                ),

            "exclusions":
                str(
                    exclusions_path
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
        "security_universe":
            security_path,

        "issuer_universe":
            issuer_path,

        "exclusions":
            exclusions_path,

        "manifest":
            manifest_path,
    }


# =============================================================================
# SELF-TEST
# =============================================================================

def _self_test():

    # -------------------------------------------------------------------------
    # SIMULA DUAS CLASSES DA MESMA EMPRESA
    # -------------------------------------------------------------------------

    market = pd.DataFrame(
        {
            "TICKER": [
                "AAAA3",
                "BBBB3",
                "BBBB4",
            ],

            "FORMATION_PRICE": [
                10.0,
                20.0,
                21.0,
            ],

            "FORMATION_PRICE_DATE": [
                "2025-12-30",
                "2025-12-30",
                "2025-12-30",
            ],

            "TRADING_SESSIONS": [
                200,
                200,
                200,
            ],
        }
    )

    identity = pd.DataFrame(
        {
            "TICKER": [
                "AAAA3",
                "BBBB3",
                "BBBB4",
            ],

            "ISSUER_ID": [
                1,
                2,
                2,
            ],

            "COMPANY_NAME": [
                "EMPRESA A",
                "EMPRESA B",
                "EMPRESA B",
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

    security, issuer, audit = (
        build_universe(
            market_df=market,
            identity_df=identity,
            context=FakeContext(),
        )
    )

    if len(
        security
    ) != 3:

        raise UniverseIntegrityError(
            "SELF-TEST: deveria haver "
            "3 securities."
        )

    if len(
        issuer
    ) != 2:

        raise UniverseIntegrityError(
            "SELF-TEST: deveria haver "
            "2 emissores."
        )

    company_b = (
        issuer.loc[
            issuer[
                "ISSUER_ID"
            ]
            == "2"
        ]
    )

    if company_b.empty:

        raise UniverseIntegrityError(
            "SELF-TEST: emissor B ausente."
        )

    if (
        int(
            company_b.iloc[0][
                "N_SECURITIES"
            ]
        )
        != 2
    ):

        raise UniverseIntegrityError(
            "SELF-TEST: classes de ações "
            "não foram agrupadas corretamente."
        )

    return True


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("B3 STOCK SELECTOR — UNIVERSE ENGINE")
    print("=" * 72)

    _self_test()

    print("Self-test: OK")
    print("Security-level: PRESERVADO")
    print("Issuer-level: PRESERVADO")
    print("Classes do mesmo emissor: AGRUPADAS")
    print("Retorno futuro: NÃO UTILIZADO")
    print("Ranking: NÃO EXECUTADO")

    print("=" * 72)
