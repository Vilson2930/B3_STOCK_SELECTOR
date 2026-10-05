# =============================================================================
# B3 STOCK SELECTOR
# Arquivo: data/pit.py
#
# POINT-IN-TIME ENGINE
#
# RESPONSABILIDADE:
# - Controlar disponibilidade temporal dos dados
# - Impedir look-ahead
# - Aplicar data de formação
# - Aplicar cutoff contábil
# - Selecionar somente informações conhecidas na data da decisão
# - Produzir auditoria PIT
#
# NÃO FAZ:
# - Download CVM
# - Cálculo de fatores
# - Quality Score
# - Turnaround Score
# - Valuation
# - Ranking
# - Retorno futuro
#
# PRINCÍPIO CENTRAL:
#   dado econômico referente ao passado NÃO significa dado conhecido no passado.
#   O robô somente pode utilizar informação disponível/publicada até o cutoff.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

import json
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

try:
    from config import (
        OUTPUT_DIR,
        POINT_IN_TIME_REQUIRED,
        ALLOW_LOOKAHEAD,
    )

except ImportError as exc:

    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar config.py."
    ) from exc


# =============================================================================
# EXCEÇÕES
# =============================================================================

class PITError(RuntimeError):
    """Erro geral do Point-in-Time Engine."""


class PITLookAheadError(PITError):
    """Tentativa de utilização de informação futura."""


class PITDateError(PITError):
    """Erro relacionado às datas PIT."""


class PITDataError(PITError):
    """Erro estrutural nos dados."""


# =============================================================================
# CONFIGURAÇÃO PIT
# =============================================================================

@dataclass(frozen=True)
class PITContext:

    formation_date: pd.Timestamp

    accounting_cutoff: pd.Timestamp

    market_cutoff: pd.Timestamp

    created_at_utc: str

    strict: bool = True


# =============================================================================
# DATAS
# =============================================================================

def normalize_date(
    value,
    name: str = "date",
) -> pd.Timestamp:
    """
    Converte uma data para Timestamp normalizado.

    Horário é removido porque o robô trabalha,
    neste estágio, com fechamento diário.
    """

    if value is None:
        raise PITDateError(
            f"FAIL-SAFE: {name} não informada."
        )

    try:

        result = pd.Timestamp(
            value
        ).normalize()

    except Exception as exc:

        raise PITDateError(
            f"Data inválida em {name}: {value}"
        ) from exc

    if pd.isna(result):

        raise PITDateError(
            f"Data inválida em {name}: {value}"
        )

    return result


# =============================================================================
# CONTEXTO PIT
# =============================================================================

def create_pit_context(
    formation_date,
    accounting_cutoff,
    market_cutoff=None,
    strict: bool = True,
) -> PITContext:
    """
    Cria o contexto temporal de uma execução.

    formation_date:
        data em que a carteira/ranking é formada.

    accounting_cutoff:
        última data permitida para informação contábil
        conhecida/publicada.

    market_cutoff:
        última sessão de mercado permitida.

    Exemplo histórico validado no estudo:

        formation_date   = 2018-09-30
        accounting_cutoff = 2018-06-30
        market_cutoff     = 2018-09-28

    O exemplo acima NÃO é imposto ao robô.
    Cada execução recebe seu próprio contexto.
    """

    if not POINT_IN_TIME_REQUIRED:

        raise PITError(
            "FAIL-SAFE: POINT_IN_TIME_REQUIRED está desativado."
        )

    if ALLOW_LOOKAHEAD:

        raise PITLookAheadError(
            "FAIL-SAFE: ALLOW_LOOKAHEAD está ativado."
        )

    formation_date = normalize_date(
        formation_date,
        "formation_date",
    )

    accounting_cutoff = normalize_date(
        accounting_cutoff,
        "accounting_cutoff",
    )

    if market_cutoff is None:

        market_cutoff = formation_date

    else:

        market_cutoff = normalize_date(
            market_cutoff,
            "market_cutoff",
        )

    # -------------------------------------------------------------------------
    # REGRAS TEMPORAIS
    # -------------------------------------------------------------------------

    if accounting_cutoff > formation_date:

        raise PITLookAheadError(
            "FAIL-SAFE: accounting_cutoff ocorre "
            "depois da formation_date."
        )

    if market_cutoff > formation_date:

        raise PITLookAheadError(
            "FAIL-SAFE: market_cutoff ocorre "
            "depois da formation_date."
        )

    created_at = (
        pd.Timestamp.now(
            tz="UTC"
        ).isoformat()
    )

    return PITContext(
        formation_date=formation_date,
        accounting_cutoff=accounting_cutoff,
        market_cutoff=market_cutoff,
        created_at_utc=created_at,
        strict=bool(strict),
    )


# =============================================================================
# VALIDAÇÃO DE COLUNAS
# =============================================================================

def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    dataset_name: str = "dataset",
) -> None:
    """
    Fail-safe estrutural.
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise PITDataError(
            f"{dataset_name} não é DataFrame."
        )

    if df.empty:

        raise PITDataError(
            f"{dataset_name} está vazio."
        )

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:

        raise PITDataError(
            f"FAIL-SAFE: {dataset_name} sem colunas "
            f"obrigatórias: {missing}"
        )


# =============================================================================
# CONVERSÃO SEGURA DE COLUNA DE DATA
# =============================================================================

def parse_date_column(
    df: pd.DataFrame,
    column: str,
    dataset_name: str,
) -> pd.Series:
    """
    Converte uma coluna em datetime.

    Datas inválidas não são aceitas silenciosamente.
    """

    if column not in df.columns:

        raise PITDataError(
            f"Coluna de data ausente: {column}"
        )

    parsed = pd.to_datetime(
        df[column],
        errors="coerce",
        dayfirst=False,
    )

    invalid = (
        df[column].notna()
        &
        parsed.isna()
    )

    if invalid.any():

        examples = (
            df.loc[
                invalid,
                column,
            ]
            .astype(str)
            .head(10)
            .tolist()
        )

        raise PITDataError(
            f"FAIL-SAFE: datas inválidas em "
            f"{dataset_name}.{column}. "
            f"Exemplos: {examples}"
        )

    return parsed.dt.normalize()


# =============================================================================
# FILTRO GENÉRICO PIT
# =============================================================================

def filter_available_by_date(
    df: pd.DataFrame,
    availability_column: str,
    cutoff,
    *,
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """
    Mantém somente linhas cuja informação já estava disponível
    até a data de corte.

    Este é o filtro PIT mais importante.

    IMPORTANTE:
    availability_column deve representar a DATA DE DISPONIBILIDADE
    da informação, não simplesmente a data econômica do balanço.
    """

    require_columns(
        df,
        [availability_column],
        dataset_name,
    )

    cutoff = normalize_date(
        cutoff,
        "cutoff",
    )

    result = df.copy()

    result[
        "__PIT_AVAILABILITY_DATE"
    ] = parse_date_column(
        result,
        availability_column,
        dataset_name,
    )

    future_mask = (
        result["__PIT_AVAILABILITY_DATE"]
        >
        cutoff
    )

    result = result.loc[
        ~future_mask
    ].copy()

    return result


# =============================================================================
# FILTRO CONTÁBIL
# =============================================================================

def filter_accounting_pit(
    df: pd.DataFrame,
    context: PITContext,
    *,
    availability_column: str,
    reference_column: Optional[str] = None,
    dataset_name: str = "accounting",
) -> pd.DataFrame:
    """
    Filtra informações contábeis.

    Existem duas datas conceitualmente diferentes:

    1. reference_column
       período econômico ao qual a informação pertence.

    2. availability_column
       quando aquela informação ficou disponível.

    O filtro contra look-ahead é feito pela disponibilidade.
    """

    required = [
        availability_column
    ]

    if reference_column:
        required.append(
            reference_column
        )

    require_columns(
        df,
        required,
        dataset_name,
    )

    result = filter_available_by_date(
        df=df,
        availability_column=availability_column,
        cutoff=context.accounting_cutoff,
        dataset_name=dataset_name,
    )

    if reference_column:

        result[
            "__PIT_REFERENCE_DATE"
        ] = parse_date_column(
            result,
            reference_column,
            dataset_name,
        )

        # Um período econômico posterior ao cutoff
        # também não pode entrar.
        invalid_reference = (
            result["__PIT_REFERENCE_DATE"]
            >
            context.accounting_cutoff
        )

        result = result.loc[
            ~invalid_reference
        ].copy()

    return result


# =============================================================================
# FILTRO DE MERCADO
# =============================================================================

def filter_market_pit(
    df: pd.DataFrame,
    context: PITContext,
    *,
    date_column: str,
    dataset_name: str = "market",
) -> pd.DataFrame:
    """
    Remove qualquer preço posterior ao market_cutoff.
    """

    require_columns(
        df,
        [date_column],
        dataset_name,
    )

    result = df.copy()

    result[
        "__PIT_MARKET_DATE"
    ] = parse_date_column(
        result,
        date_column,
        dataset_name,
    )

    future_mask = (
        result["__PIT_MARKET_DATE"]
        >
        context.market_cutoff
    )

    result = result.loc[
        ~future_mask
    ].copy()

    return result


# =============================================================================
# ÚLTIMO REGISTRO CONHECIDO POR EMPRESA
# =============================================================================

def latest_available_record(
    df: pd.DataFrame,
    *,
    entity_column: str,
    availability_column: str,
    cutoff,
    reference_column: Optional[str] = None,
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """
    Retorna o último registro conhecido por entidade até o cutoff.

    Em caso de empate na data de disponibilidade, a data de referência
    pode ser utilizada como segundo critério.
    """

    required = [
        entity_column,
        availability_column,
    ]

    if reference_column:
        required.append(
            reference_column
        )

    require_columns(
        df,
        required,
        dataset_name,
    )

    result = filter_available_by_date(
        df=df,
        availability_column=availability_column,
        cutoff=cutoff,
        dataset_name=dataset_name,
    )

    result[
        "__PIT_AVAILABILITY_DATE"
    ] = parse_date_column(
        result,
        availability_column,
        dataset_name,
    )

    sort_columns = [
        entity_column,
        "__PIT_AVAILABILITY_DATE",
    ]

    if reference_column:

        result[
            "__PIT_REFERENCE_DATE"
        ] = parse_date_column(
            result,
            reference_column,
            dataset_name,
        )

        sort_columns.append(
            "__PIT_REFERENCE_DATE"
        )

    result = result.sort_values(
        sort_columns,
        kind="mergesort",
    )

    result = (
        result
        .groupby(
            entity_column,
            as_index=False,
            sort=False,
        )
        .tail(1)
        .copy()
    )

    return result


# =============================================================================
# SNAPSHOT CONTÁBIL POR EMPRESA
# =============================================================================

def accounting_snapshot(
    df: pd.DataFrame,
    context: PITContext,
    *,
    entity_column: str,
    availability_column: str,
    reference_column: str,
    dataset_name: str = "accounting",
) -> pd.DataFrame:
    """
    Produz snapshot contábil PIT por empresa.

    Somente documentos:
    - disponíveis até accounting_cutoff;
    - referentes a períodos não posteriores ao cutoff.

    Depois seleciona o registro mais recente permitido.
    """

    result = filter_accounting_pit(
        df=df,
        context=context,
        availability_column=availability_column,
        reference_column=reference_column,
        dataset_name=dataset_name,
    )

    if result.empty:

        raise PITDataError(
            "FAIL-SAFE: snapshot contábil PIT vazio."
        )

    result = result.sort_values(
        [
            entity_column,
            "__PIT_AVAILABILITY_DATE",
            "__PIT_REFERENCE_DATE",
        ],
        kind="mergesort",
    )

    result = (
        result
        .groupby(
            entity_column,
            sort=False,
        )
        .tail(1)
        .copy()
    )

    return result


# =============================================================================
# SNAPSHOT DE MERCADO
# =============================================================================

def market_snapshot(
    df: pd.DataFrame,
    context: PITContext,
    *,
    ticker_column: str,
    date_column: str,
    dataset_name: str = "market",
) -> pd.DataFrame:
    """
    Obtém o último registro de mercado conhecido por ticker
    até market_cutoff.
    """

    result = filter_market_pit(
        df=df,
        context=context,
        date_column=date_column,
        dataset_name=dataset_name,
    )

    if result.empty:

        raise PITDataError(
            "FAIL-SAFE: snapshot de mercado PIT vazio."
        )

    result = result.sort_values(
        [
            ticker_column,
            "__PIT_MARKET_DATE",
        ],
        kind="mergesort",
    )

    result = (
        result
        .groupby(
            ticker_column,
            sort=False,
        )
        .tail(1)
        .copy()
    )

    return result


# =============================================================================
# AUDITORIA CONTRA LOOK-AHEAD
# =============================================================================

def audit_no_lookahead(
    df: pd.DataFrame,
    *,
    date_columns: Iterable[str],
    cutoff,
    dataset_name: str = "dataset",
) -> dict:
    """
    Verifica explicitamente se alguma data ultrapassa o cutoff.
    """

    cutoff = normalize_date(
        cutoff,
        "cutoff",
    )

    report = {
        "dataset": dataset_name,
        "cutoff": str(
            cutoff.date()
        ),
        "rows": int(
            len(df)
        ),
        "columns_checked": [],
        "violations": {},
        "passed": True,
    }

    for column in date_columns:

        require_columns(
            df,
            [column],
            dataset_name,
        )

        parsed = parse_date_column(
            df,
            column,
            dataset_name,
        )

        violation = (
            parsed > cutoff
        )

        count = int(
            violation.sum()
        )

        report[
            "columns_checked"
        ].append(column)

        report[
            "violations"
        ][column] = count

        if count > 0:
            report["passed"] = False

    return report


# =============================================================================
# ASSERT NO LOOK-AHEAD
# =============================================================================

def assert_no_lookahead(
    df: pd.DataFrame,
    *,
    date_columns: Iterable[str],
    cutoff,
    dataset_name: str = "dataset",
) -> None:
    """
    Interrompe imediatamente se houver informação futura.
    """

    report = audit_no_lookahead(
        df=df,
        date_columns=date_columns,
        cutoff=cutoff,
        dataset_name=dataset_name,
    )

    if not report["passed"]:

        raise PITLookAheadError(
            "FAIL-SAFE: LOOK-AHEAD DETECTADO. "
            f"Dataset={dataset_name}. "
            f"Violações={report['violations']}"
        )


# =============================================================================
# AUDITORIA COMPLETA DO CONTEXTO
# =============================================================================

def context_to_dict(
    context: PITContext,
) -> dict:

    return {
        "formation_date": str(
            context.formation_date.date()
        ),

        "accounting_cutoff": str(
            context.accounting_cutoff.date()
        ),

        "market_cutoff": str(
            context.market_cutoff.date()
        ),

        "created_at_utc":
            context.created_at_utc,

        "strict":
            context.strict,
    }


# =============================================================================
# SALVAR MANIFESTO PIT
# =============================================================================

def save_pit_manifest(
    context: PITContext,
    *,
    filename: str = "pit_manifest.json",
    extra_metadata: Optional[dict] = None,
) -> Path:
    """
    Salva as datas efetivamente utilizadas pela execução.
    """

    output = (
        OUTPUT_DIR
        / filename
    )

    manifest = {
        "engine": "POINT_IN_TIME",
        "status": "VALID",
        **context_to_dict(context),
    }

    if extra_metadata:

        manifest[
            "metadata"
        ] = extra_metadata

    with output.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output


# =============================================================================
# REMOÇÃO DE COLUNAS INTERNAS
# =============================================================================

def remove_internal_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove colunas auxiliares criadas pelo PIT Engine.
    """

    internal = [
        column
        for column in df.columns
        if str(column).startswith(
            "__PIT_"
        )
    ]

    return df.drop(
        columns=internal,
        errors="ignore",
    ).copy()


# =============================================================================
# TESTE INTERNO
# =============================================================================

def _self_test():
    """
    Teste simples para verificar se o fail-safe está funcionando.
    """

    context = create_pit_context(
        formation_date="2018-09-30",
        accounting_cutoff="2018-06-30",
        market_cutoff="2018-09-28",
    )

    sample = pd.DataFrame(
        {
            "ISSUER_ID": [
                1,
                1,
                2,
                2,
            ],

            "REFERENCE_DATE": [
                "2017-12-31",
                "2018-06-30",
                "2017-12-31",
                "2018-09-30",
            ],

            "AVAILABLE_DATE": [
                "2018-03-30",
                "2018-06-30",
                "2018-03-30",
                "2018-10-30",
            ],

            "VALUE": [
                10,
                20,
                30,
                999999,
            ],
        }
    )

    filtered = filter_accounting_pit(
        df=sample,
        context=context,
        availability_column="AVAILABLE_DATE",
        reference_column="REFERENCE_DATE",
        dataset_name="SELF_TEST",
    )

    # O valor futuro jamais pode sobreviver.
    if (
        filtered["VALUE"]
        == 999999
    ).any():

        raise PITLookAheadError(
            "SELF-TEST FALHOU: dado futuro entrou no PIT."
        )

    return True


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("B3 STOCK SELECTOR — POINT-IN-TIME ENGINE")
    print("=" * 72)

    _self_test()

    print("Self-test PIT: OK")
    print("Look-ahead: BLOQUEADO")
    print("Dados futuros no ranking: BLOQUEADOS")

    print("=" * 72)
