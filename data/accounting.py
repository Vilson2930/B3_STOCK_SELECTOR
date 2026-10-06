"""
B3 STOCK SELECTOR — ACCOUNTING ENGINE

Responsabilidade
----------------
Transformar os arquivos oficiais brutos da CVM (FCA / ITR / DFP / FRE)
em uma base contábil canônica por emissor, respeitando estritamente
Point-in-Time (PIT).

Fluxo:
    CVM RAW
        ↓
    FCA → ticker → emissor
        ↓
    ITR / DFP
        ↓
    filtro PIT
        ↓
    seleção das contas contábeis
        ↓
    agregação por emissor
        ↓
    accounting_data canônico
        ↓
    Fundamentals Engine

Princípios
----------
- sem retorno futuro;
- sem look-ahead;
- sem ranking;
- sem score;
- sem imputação econômica;
- sem criação artificial de dados;
- emissor é a unidade fundamental;
- ticker permanece apenas como ponte de identidade.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
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


if not hasattr(project_config, "DATA_DIR"):
    raise RuntimeError(
        "FAIL-SAFE: DATA_DIR ausente em config.py."
    )


DATA_DIR = Path(project_config.DATA_DIR)

ACCOUNTING_DIR = DATA_DIR / "accounting"
ACCOUNTING_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CVM
# =============================================================================

try:
    from data.cvm import (
        download_dataset,
        list_zip_files,
        read_csv_from_zip,
    )
except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar data.cvm."
    ) from exc


# =============================================================================
# EXCEÇÕES
# =============================================================================

class AccountingError(RuntimeError):
    pass


class AccountingDataError(AccountingError):
    pass


class AccountingPITError(AccountingError):
    pass


class AccountingIdentityError(AccountingError):
    pass


# =============================================================================
# CONSTANTES
# =============================================================================

TICKER_PATTERN = re.compile(r"^[A-Z]{4}[3456]$")

FORBIDDEN_FUTURE_PATTERNS = (
    "FUTURE_RETURN",
    "RETORNO_FUTURO",
    "FUTURE_WINNER",
    "WINNER_LABEL",
)


# Contas estruturais do Balanço Patrimonial.
#
# A busca é prioritariamente por CD_CONTA. A descrição é utilizada apenas
# como fallback conservador quando necessário.

ACCOUNT_RULES = {
    "ATIVO_TOTAL": {
        "codes": ("1",),
        "descriptions": ("ATIVO TOTAL",),
        "statement": "BPA",
    },

    "ATIVO_CIRCULANTE": {
        "codes": ("1.01",),
        "descriptions": ("ATIVO CIRCULANTE",),
        "statement": "BPA",
    },

    "ESTOQUES": {
        "codes": ("1.01.04",),
        "descriptions": ("ESTOQUES",),
        "statement": "BPA",
    },

    "IMOBILIZADO": {
        "codes": ("1.02.03",),
        "descriptions": ("IMOBILIZADO",),
        "statement": "BPA",
    },

    "INTANGIVEL": {
        "codes": ("1.02.04",),
        "descriptions": ("INTANGIVEL",),
        "statement": "BPA",
    },

    "PASSIVO_CIRCULANTE": {
        "codes": ("2.01",),
        "descriptions": ("PASSIVO CIRCULANTE",),
        "statement": "BPP",
    },

    "PL": {
        "codes": ("2.03",),
        "descriptions": (
            "PATRIMONIO LIQUIDO",
            "PATRIMONIO LIQUIDO CONSOLIDADO",
        ),
        "statement": "BPP",
    },
}


# Contas de fluxo/cumulativas.
FLOW_RULES = {
    "RECEITA": {
        "codes": ("3.01",),
        "descriptions": (
            "RECEITA DE VENDA DE BENS E/OU SERVICOS",
            "RECEITA OPERACIONAL",
            "RECEITA LIQUIDA",
        ),
        "statement": "DRE",
    },

    "CUSTO": {
        "codes": ("3.02",),
        "descriptions": (
            "CUSTO DOS BENS E/OU SERVICOS VENDIDOS",
            "CUSTO DOS PRODUTOS VENDIDOS",
            "CUSTOS",
        ),
        "statement": "DRE",
    },

    "LUCRO_BRUTO": {
        "codes": ("3.03",),
        "descriptions": (
            "RESULTADO BRUTO",
            "LUCRO BRUTO",
        ),
        "statement": "DRE",
    },

    "EBIT": {
        "codes": ("3.05",),
        "descriptions": (
            "RESULTADO ANTES DO RESULTADO FINANCEIRO E DOS TRIBUTOS",
            "RESULTADO OPERACIONAL",
        ),
        "statement": "DRE",
    },

    "LUCRO_LIQUIDO": {
        "codes": ("3.11",),
        "descriptions": (
            "LUCRO/PREJUIZO CONSOLIDADO DO PERIODO",
            "LUCRO OU PREJUIZO LIQUIDO DO PERIODO",
            "LUCRO/PREJUIZO DO PERIODO",
        ),
        "statement": "DRE",
    },

    "FCO": {
        "codes": ("6.01",),
        "descriptions": (
            "CAIXA LIQUIDO ATIVIDADES OPERACIONAIS",
            "CAIXA LIQUIDO GERADO PELAS ATIVIDADES OPERACIONAIS",
        ),
        "statement": "DFC",
    },
}


# Dívida bruta conforme metodologia histórica já utilizada no estudo:
#
# curto prazo = 2.01.04
# longo prazo = 2.02.01
#
# Não substituímos silenciosamente por "passivo total".

DEBT_RULES = {
    "DIVIDA_CP": {
        "codes": ("2.01.04",),
        "statement": "BPP",
    },

    "DIVIDA_LP": {
        "codes": ("2.02.01",),
        "statement": "BPP",
    },
}


CASH_RULE = {
    "codes": ("1.01.01",),
    "descriptions": (
        "CAIXA E EQUIVALENTES DE CAIXA",
        "CAIXA",
    ),
    "statement": "BPA",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class AccountingContext:
    formation_date: pd.Timestamp
    accounting_cutoff: pd.Timestamp
    year: int


# =============================================================================
# UTILIDADES
# =============================================================================

def _normalize_text(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().upper()

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        ch
        for ch in text
        if not unicodedata.combining(ch)
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _normalize_column(value: str) -> str:
    return _normalize_text(value).replace(" ", "_")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result.columns = [
        _normalize_column(col)
        for col in result.columns
    ]

    return result


def _to_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)

    if pd.isna(ts):
        raise AccountingPITError(
            "Data inválida."
        )

    return ts.normalize()


def _parse_dates(
    series: pd.Series,
) -> pd.Series:

    parsed = pd.to_datetime(
        series,
        errors="coerce",
        dayfirst=False,
    )

    if parsed.notna().sum() == 0:
        parsed = pd.to_datetime(
            series,
            errors="coerce",
            dayfirst=True,
        )

    return parsed


def _numeric(
    series: pd.Series,
) -> pd.Series:

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(
            series,
            errors="coerce",
        )

    values = (
        series.astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    values = values.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
        }
    )

    return pd.to_numeric(
        values,
        errors="coerce",
    )


def _issuer_id(value) -> Optional[str]:
    if pd.isna(value):
        return None

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    digits = re.sub(
        r"\D",
        "",
        text,
    )

    if not digits:
        return None

    return str(int(digits))


def _cnpj(value) -> Optional[str]:
    if pd.isna(value):
        return None

    digits = re.sub(
        r"\D",
        "",
        str(value),
    )

    if not digits:
        return None

    return digits.zfill(14)


def _ticker(value) -> Optional[str]:
    if pd.isna(value):
        return None

    text = (
        str(value)
        .strip()
        .upper()
        .replace(" ", "")
    )

    if not TICKER_PATTERN.fullmatch(text):
        return None

    return text


def _find_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    required: bool = True,
) -> Optional[str]:

    normalized = {
        _normalize_column(col): col
        for col in df.columns
    }

    for candidate in candidates:
        key = _normalize_column(candidate)

        if key in normalized:
            return normalized[key]

    if required:
        raise AccountingDataError(
            "Nenhuma coluna encontrada entre: "
            f"{list(candidates)}"
        )

    return None


def assert_no_future_information(
    df: pd.DataFrame,
    *,
    stage: str,
) -> None:

    bad = []

    for column in df.columns:
        name = str(column).upper()

        if name.startswith("FUTURE_RETURN_USED"):
            values = df[column]

            if values.fillna(False).astype(bool).any():
                bad.append(column)

            continue

        if any(
            pattern in name
            for pattern in FORBIDDEN_FUTURE_PATTERNS
        ):
            bad.append(column)

    if bad:
        raise AccountingDataError(
            "FAIL-SAFE: informação futura detectada "
            f"em {stage}: {bad}"
        )


# =============================================================================
# CONTEXTO PIT
# =============================================================================

def create_accounting_context(
    formation_date,
    accounting_cutoff,
) -> AccountingContext:

    formation = _to_timestamp(
        formation_date
    )

    cutoff = _to_timestamp(
        accounting_cutoff
    )

    if cutoff > formation:
        raise AccountingPITError(
            "FAIL-SAFE: accounting_cutoff posterior "
            "à formation_date."
        )

    return AccountingContext(
        formation_date=formation,
        accounting_cutoff=cutoff,
        year=int(cutoff.year),
    )


# =============================================================================
# ZIP / ARQUIVOS INTERNOS
# =============================================================================

def _statement_type(
    filename: str,
) -> Optional[str]:

    name = _normalize_text(
        Path(filename).name
    )

    if "_BPA_" in name:
        return "BPA"

    if "_BPP_" in name:
        return "BPP"

    if "_DRE_" in name:
        return "DRE"

    if "_DFC_MD_" in name:
        return "DFC"

    if "_DFC_MI_" in name:
        return "DFC"

    return None


def _is_consolidated(
    filename: str,
) -> bool:

    name = _normalize_text(
        Path(filename).name
    )

    return "_CON_" in name


def _read_statement_files(
    dataset: str,
    year: int,
) -> dict[str, pd.DataFrame]:

    zip_path = download_dataset(
        dataset,
        year,
    )

    files = list_zip_files(
        dataset,
        int(year),
    )

    result: dict[str, list[pd.DataFrame]] = {
        "BPA": [],
        "BPP": [],
        "DRE": [],
        "DFC": [],
    }

    for filename in files:

        statement = _statement_type(
            filename
        )

        if statement is None:
            continue

        # Preferência metodológica:
        # demonstrações consolidadas.
        if not _is_consolidated(
            filename
        ):
            continue

        frame = read_csv_from_zip(
            dataset,
            int(year),
            filename,
        )

        frame = _normalize_columns(
            frame
        )

        frame["SOURCE_DATASET"] = dataset
        frame["SOURCE_YEAR"] = year
        frame["SOURCE_FILE"] = filename
        frame["STATEMENT"] = statement

        result[statement].append(
            frame
        )

    consolidated: dict[str, pd.DataFrame] = {}

    for statement, frames in result.items():

        if not frames:
            consolidated[statement] = (
                pd.DataFrame()
            )

            continue

        consolidated[statement] = (
            pd.concat(
                frames,
                ignore_index=True,
                sort=False,
            )
        )

    return consolidated


# =============================================================================
# DFP / ITR — COMPOSIÇÃO DO CAPITAL
# =============================================================================

def _find_capital_composition_file(
    dataset: str,
    year: int,
) -> Optional[str]:
    """
    Localiza, dentro do ZIP oficial DFP/ITR, o CSV da seção
    Dados da Empresa / Composição do Capital.

    A identificação é deliberadamente conservadora:
    - aceita apenas arquivos cujo nome contenha COMPOSICAO e CAPITAL;
    - exige identificação única;
    - retorna None quando a seção não existir no ano/dataset.
    """
    files = list_zip_files(
        dataset,
        int(year),
    )

    candidates = []

    for filename in files:
        normalized = _normalize_text(
            Path(filename).name
        )

        if (
            "COMPOSICAO" in normalized
            and "CAPITAL" in normalized
            and normalized.endswith(".CSV")
        ):
            candidates.append(filename)

    candidates = sorted(
        set(candidates)
    )

    if not candidates:
        return None

    if len(candidates) != 1:
        raise AccountingDataError(
            "FAIL-SAFE: arquivo de Composição do Capital "
            f"não identificado de forma única em {dataset}/{year}. "
            f"Encontrados: {candidates}"
        )

    return candidates[0]


def _read_capital_composition(
    dataset: str,
    year: int,
    context: AccountingContext,
) -> pd.DataFrame:
    """
    Lê e normaliza a Composição do Capital oficial da CVM.

    Saída canônica por observação:
        ISSUER_ID
        CAPITAL_REFERENCE_DATE
        SHARES_ON_ISSUED
        SHARES_PN_ISSUED
        SHARES_TOTAL_ISSUED
        SHARES_ON_TREASURY
        SHARES_PN_TREASURY
        SHARES_TOTAL_TREASURY
        SHARES_ON_OUTSTANDING
        SHARES_PN_OUTSTANDING
        SHARES_OUTSTANDING
        CAPITAL_SOURCE_DATASET
        CAPITAL_SOURCE_YEAR
        CAPITAL_SOURCE_FILE

    "Outstanding" é calculado como ações integralizadas/emitidas
    menos ações mantidas em tesouraria. Ausência de informação
    não é transformada silenciosamente em zero, salvo quando o
    total pode ser reconstruído de ON + PN.
    """
    filename = _find_capital_composition_file(
        dataset,
        int(year),
    )

    if filename is None:
        return pd.DataFrame()

    raw = read_csv_from_zip(
        dataset,
        int(year),
        filename,
    )

    raw = _normalize_columns(
        raw
    )

    if raw.empty:
        return pd.DataFrame()

    cnpj_col = _find_column(
        raw,
        (
            "CNPJ_CIA",
            "CNPJ_COMPANHIA",
            "CNPJ",
        ),
        required=False,
    )

    cvm_col = _find_column(
        raw,
        (
            "CD_CVM",
            "CODIGO_CVM",
        ),
        required=False,
    )

    if cnpj_col is None and cvm_col is None:
        raise AccountingDataError(
            "FAIL-SAFE: Composição do Capital sem "
            "identificador de emissor."
        )

    reference_col = _find_column(
        raw,
        (
            "DT_REFER",
            "DATA_REFERENCIA",
        ),
    )

    version_col = _find_column(
        raw,
        (
            "VERSAO",
            "VERSAO_DOCUMENTO",
        ),
        required=False,
    )

    on_issued_col = _find_column(
        raw,
        (
            "QT_ACAO_ORDIN_CAP_INTEGR",
            "QT_ACAO_ORDINARIA_CAP_INTEGR",
            "QT_ACOES_ORDINARIAS_CAPITAL_INTEGRALIZADO",
        ),
        required=False,
    )

    pn_issued_col = _find_column(
        raw,
        (
            "QT_ACAO_PREF_CAP_INTEGR",
            "QT_ACAO_PREFERENCIAL_CAP_INTEGR",
            "QT_ACOES_PREFERENCIAIS_CAPITAL_INTEGRALIZADO",
        ),
        required=False,
    )

    total_issued_col = _find_column(
        raw,
        (
            "QT_ACAO_TOTAL_CAP_INTEGR",
            "QT_ACOES_TOTAL_CAPITAL_INTEGRALIZADO",
        ),
        required=False,
    )

    on_treasury_col = _find_column(
        raw,
        (
            "QT_ACAO_ORDIN_TESOURO",
            "QT_ACAO_ORDINARIA_TESOURO",
            "QT_ACOES_ORDINARIAS_TESOURARIA",
        ),
        required=False,
    )

    pn_treasury_col = _find_column(
        raw,
        (
            "QT_ACAO_PREF_TESOURO",
            "QT_ACAO_PREFERENCIAL_TESOURO",
            "QT_ACOES_PREFERENCIAIS_TESOURARIA",
        ),
        required=False,
    )

    total_treasury_col = _find_column(
        raw,
        (
            "QT_ACAO_TOTAL_TESOURO",
            "QT_ACOES_TOTAL_TESOURARIA",
        ),
        required=False,
    )

    if (
        on_issued_col is None
        and pn_issued_col is None
        and total_issued_col is None
    ):
        raise AccountingDataError(
            "FAIL-SAFE: Composição do Capital localizada, "
            "mas nenhuma coluna de quantidade de ações "
            "foi reconhecida. Colunas disponíveis: "
            f"{sorted(raw.columns.tolist())}"
        )

    result = pd.DataFrame(
        index=raw.index
    )

    if cnpj_col is not None:
        result["ISSUER_ID"] = (
            raw[cnpj_col]
            .map(_cnpj)
        )
    else:
        result["ISSUER_ID"] = (
            raw[cvm_col]
            .map(_issuer_id)
        )

    result[
        "CAPITAL_REFERENCE_DATE"
    ] = _parse_dates(
        raw[reference_col]
    )

    if version_col is not None:
        result[
            "CAPITAL_VERSION"
        ] = pd.to_numeric(
            raw[version_col],
            errors="coerce",
        )
    else:
        result[
            "CAPITAL_VERSION"
        ] = np.nan

    def numeric_or_nan(
        column: Optional[str],
    ) -> pd.Series:
        if column is None:
            return pd.Series(
                np.nan,
                index=raw.index,
                dtype="float64",
            )

        return _numeric(
            raw[column]
        )

    result[
        "SHARES_ON_ISSUED"
    ] = numeric_or_nan(
        on_issued_col
    )

    result[
        "SHARES_PN_ISSUED"
    ] = numeric_or_nan(
        pn_issued_col
    )

    result[
        "SHARES_TOTAL_ISSUED"
    ] = numeric_or_nan(
        total_issued_col
    )

    result[
        "SHARES_ON_TREASURY"
    ] = numeric_or_nan(
        on_treasury_col
    )

    result[
        "SHARES_PN_TREASURY"
    ] = numeric_or_nan(
        pn_treasury_col
    )

    result[
        "SHARES_TOTAL_TREASURY"
    ] = numeric_or_nan(
        total_treasury_col
    )

    # Reconstruções apenas quando matematicamente determinadas.
    missing_total_issued = (
        result[
            "SHARES_TOTAL_ISSUED"
        ].isna()
        & result[
            "SHARES_ON_ISSUED"
        ].notna()
        & result[
            "SHARES_PN_ISSUED"
        ].notna()
    )

    result.loc[
        missing_total_issued,
        "SHARES_TOTAL_ISSUED",
    ] = (
        result.loc[
            missing_total_issued,
            "SHARES_ON_ISSUED",
        ]
        + result.loc[
            missing_total_issued,
            "SHARES_PN_ISSUED",
        ]
    )

    missing_total_treasury = (
        result[
            "SHARES_TOTAL_TREASURY"
        ].isna()
        & result[
            "SHARES_ON_TREASURY"
        ].notna()
        & result[
            "SHARES_PN_TREASURY"
        ].notna()
    )

    result.loc[
        missing_total_treasury,
        "SHARES_TOTAL_TREASURY",
    ] = (
        result.loc[
            missing_total_treasury,
            "SHARES_ON_TREASURY",
        ]
        + result.loc[
            missing_total_treasury,
            "SHARES_PN_TREASURY",
        ]
    )

    def outstanding(
        issued: pd.Series,
        treasury: pd.Series,
    ) -> pd.Series:
        value = issued.copy()

        both_known = (
            issued.notna()
            & treasury.notna()
        )

        value.loc[
            both_known
        ] = (
            issued.loc[both_known]
            - treasury.loc[both_known]
        )

        # Se tesouraria estiver ausente, não presumimos zero.
        value.loc[
            issued.notna()
            & treasury.isna()
        ] = np.nan

        value.loc[
            issued.isna()
        ] = np.nan

        return value

    result[
        "SHARES_ON_OUTSTANDING"
    ] = outstanding(
        result["SHARES_ON_ISSUED"],
        result["SHARES_ON_TREASURY"],
    )

    result[
        "SHARES_PN_OUTSTANDING"
    ] = outstanding(
        result["SHARES_PN_ISSUED"],
        result["SHARES_PN_TREASURY"],
    )

    result[
        "SHARES_OUTSTANDING"
    ] = outstanding(
        result["SHARES_TOTAL_ISSUED"],
        result["SHARES_TOTAL_TREASURY"],
    )

    # Quando o total outstanding não puder ser obtido diretamente,
    # ON + PN pode reconstruí-lo de forma exata.
    reconstruct_outstanding = (
        result[
            "SHARES_OUTSTANDING"
        ].isna()
        & result[
            "SHARES_ON_OUTSTANDING"
        ].notna()
        & result[
            "SHARES_PN_OUTSTANDING"
        ].notna()
    )

    result.loc[
        reconstruct_outstanding,
        "SHARES_OUTSTANDING",
    ] = (
        result.loc[
            reconstruct_outstanding,
            "SHARES_ON_OUTSTANDING",
        ]
        + result.loc[
            reconstruct_outstanding,
            "SHARES_PN_OUTSTANDING",
        ]
    )

    result[
        "CAPITAL_SOURCE_DATASET"
    ] = dataset

    result[
        "CAPITAL_SOURCE_YEAR"
    ] = int(year)

    result[
        "CAPITAL_SOURCE_FILE"
    ] = filename

    result = result.loc[
        result["ISSUER_ID"].notna()
        & result[
            "CAPITAL_REFERENCE_DATE"
        ].notna()
    ].copy()

    # PIT por data de referência. A limitação de disponibilidade/publicação
    # histórica permanece explícita no projeto: o ZIP anual é reapresentável.
    result = result.loc[
        result[
            "CAPITAL_REFERENCE_DATE"
        ]
        <= context.accounting_cutoff
    ].copy()

    if result.empty:
        return result

    numeric_share_columns = (
        "SHARES_ON_ISSUED",
        "SHARES_PN_ISSUED",
        "SHARES_TOTAL_ISSUED",
        "SHARES_ON_TREASURY",
        "SHARES_PN_TREASURY",
        "SHARES_TOTAL_TREASURY",
        "SHARES_ON_OUTSTANDING",
        "SHARES_PN_OUTSTANDING",
        "SHARES_OUTSTANDING",
    )

    for column in numeric_share_columns:
        invalid = (
            result[column].notna()
            & (result[column] < 0)
        )

        if invalid.any():
            raise AccountingDataError(
                "FAIL-SAFE: quantidade negativa de ações "
                f"em {dataset}/{year}, coluna {column}."
            )

    result = result.sort_values(
        [
            "ISSUER_ID",
            "CAPITAL_REFERENCE_DATE",
            "CAPITAL_VERSION",
        ],
        na_position="first",
    )

    result = result.drop_duplicates(
        subset=["ISSUER_ID"],
        keep="last",
    )

    assert_no_future_information(
        result,
        stage=(
            f"{dataset} capital composition"
        ),
    )

    return result.reset_index(
        drop=True
    )


def _build_capital_composition(
    context: AccountingContext,
) -> pd.DataFrame:
    """
    Monta o snapshot PIT de capital.

    Prioridade:
    1. ITR do ano do cutoff;
    2. DFP do ano anterior como fallback.

    Para o mesmo emissor, vence a observação com data de referência
    mais recente; em empate, ITR recebe prioridade sobre DFP.
    """
    current_year = int(
        context.accounting_cutoff.year
    )

    frames = []

    itr = _read_capital_composition(
        "ITR",
        current_year,
        context,
    )

    if not itr.empty:
        itr = itr.copy()
        itr[
            "_CAPITAL_SOURCE_PRIORITY"
        ] = 2
        frames.append(itr)

    dfp = _read_capital_composition(
        "DFP",
        current_year - 1,
        context,
    )

    if not dfp.empty:
        dfp = dfp.copy()
        dfp[
            "_CAPITAL_SOURCE_PRIORITY"
        ] = 1
        frames.append(dfp)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    result = result.sort_values(
        [
            "ISSUER_ID",
            "CAPITAL_REFERENCE_DATE",
            "_CAPITAL_SOURCE_PRIORITY",
            "CAPITAL_VERSION",
        ],
        na_position="first",
    )

    result = result.drop_duplicates(
        subset=["ISSUER_ID"],
        keep="last",
    )

    result = result.drop(
        columns=[
            "_CAPITAL_SOURCE_PRIORITY",
        ],
        errors="ignore",
    )

    return result.reset_index(
        drop=True
    )



# =============================================================================
# FRE — CAPITAL SOCIAL POR CLASSE DE AÇÃO
# =============================================================================
#
# Esta camada NÃO substitui a Composição do Capital DFP/ITR.
#
# Objetivo:
# - obter do FRE a quantidade emitida por espécie/classe;
# - combinar essa informação com a tesouraria agregada já obtida de DFP/ITR;
# - produzir SHARES_CLASS_OUTSTANDING somente quando o cálculo for
#   matematicamente determinado;
# - nunca ratear tesouraria entre PNA/PNB ou outras classes por hipótese.
#
# Saída longa:
#     ISSUER_ID
#     SHARE_CLASS
#     SHARE_SPECIES
#     SHARE_SUBCLASS
#     SHARES_CLASS_ISSUED
#     SHARES_CLASS_TREASURY
#     SHARES_CLASS_OUTSTANDING
#     SHARE_CLASS_EXACT
#     FRE_REFERENCE_DATE
#     FRE_VERSION
#     FRE_SOURCE_YEAR
#     FRE_SOURCE_FILE
#
# A descoberta do esquema é fail-safe: se o arquivo oficial mudar e as colunas
# necessárias não puderem ser identificadas de forma inequívoca, a execução
# interrompe com a lista real de colunas do CSV. Nenhum dado é inventado.
# =============================================================================

def _find_fre_capital_class_file(
    year: int,
) -> str:

    download_dataset(
        "FRE",
        int(year),
    )

    files = list_zip_files(
        "FRE",
        int(year),
    )

    candidates = []

    for filename in files:

        normalized = (
            _normalize_text(
                Path(filename).name
            )
            .replace(" ", "_")
        )

        if (
            "CAPITAL_SOCIAL_CLASSE_ACAO"
            in normalized
            and normalized.endswith(".CSV")
            and "AUMENTO" not in normalized
            and "REDUCAO" not in normalized
            and "DESDOBRAMENTO" not in normalized
        ):
            candidates.append(filename)

    candidates = sorted(
        set(candidates)
    )

    if len(candidates) != 1:
        raise AccountingDataError(
            "FAIL-SAFE: arquivo FRE "
            "Capital Social por Classe de Ação "
            f"não identificado de forma única em FRE/{year}. "
            f"Encontrados: {candidates}"
        )

    return candidates[0]


def _find_column_by_tokens(
    df: pd.DataFrame,
    *,
    required_all: Iterable[str] = (),
    required_any: Iterable[str] = (),
    forbidden: Iterable[str] = (),
) -> Optional[str]:
    """
    Descoberta estrutural conservadora de coluna.

    Retorna a coluna apenas quando existe exatamente uma candidata.
    Não escolhe arbitrariamente entre colunas semanticamente ambíguas.
    """

    all_tokens = tuple(
        _normalize_column(value)
        for value in required_all
    )

    any_tokens = tuple(
        _normalize_column(value)
        for value in required_any
    )

    forbidden_tokens = tuple(
        _normalize_column(value)
        for value in forbidden
    )

    candidates = []

    for column in df.columns:

        normalized = _normalize_column(
            column
        )

        if any(
            token
            and token in normalized
            for token in forbidden_tokens
        ):
            continue

        if any(
            token
            and token not in normalized
            for token in all_tokens
        ):
            continue

        if (
            any_tokens
            and not any(
                token
                and token in normalized
                for token in any_tokens
            )
        ):
            continue

        candidates.append(column)

    candidates = list(
        dict.fromkeys(candidates)
    )

    if len(candidates) == 1:
        return candidates[0]

    return None


def _normalize_share_class(
    species,
    share_class=None,
) -> tuple[
    Optional[str],
    Optional[str],
    Optional[str],
]:
    """
    Normaliza espécie/classe sem inferir pelo ticker.

    Exemplos aceitos:
        Ordinária / ON
        Preferencial / PN
        Preferencial Classe A / PNA
        Preferencial Classe B / PNB

    Classes preferenciais diferentes permanecem diferentes.
    """

    species_text = _normalize_text(
        species
    )

    class_text = _normalize_text(
        share_class
    )

    combined = " ".join(
        value
        for value in (
            species_text,
            class_text,
        )
        if value
    )

    if not combined:
        return None, None, None

    is_on = (
        "ORDIN" in combined
        or re.search(
            r"(^|[^A-Z])ON([^A-Z]|$)",
            combined,
        )
        is not None
    )

    is_pn = (
        "PREFER" in combined
        or re.search(
            r"(^|[^A-Z])PN[A-Z]?([^A-Z]|$)",
            combined,
        )
        is not None
    )

    if is_on and is_pn:
        return None, None, None

    if is_on:
        return "ON", "ON", None

    if not is_pn:
        return None, None, None

    subclass = None

    explicit = re.search(
        r"(^|[^A-Z])PN([A-Z])([^A-Z]|$)",
        combined,
    )

    if explicit is not None:
        subclass = explicit.group(2)

    if subclass is None:

        match = re.search(
            r"CLASSE\s+([A-Z])([^A-Z]|$)",
            combined,
        )

        if match is not None:
            subclass = match.group(1)

    if subclass:
        return (
            f"PN{subclass}",
            "PN",
            subclass,
        )

    return "PN", "PN", None


def _read_fre_capital_classes(
    year: int,
    context: AccountingContext,
) -> pd.DataFrame:
    """
    Lê o arquivo oficial FRE Capital Social por Classe de Ação.

    O parser usa primeiro nomes conhecidos/compatíveis e, apenas quando
    inequívoco, descoberta estrutural por tokens. Mudança de esquema ambígua
    causa fail-safe.
    """

    filename = _find_fre_capital_class_file(
        int(year)
    )

    raw = read_csv_from_zip(
        "FRE",
        int(year),
        filename,
    )

    raw = _normalize_columns(
        raw
    )

    if raw.empty:
        return pd.DataFrame()

    cnpj_col = _find_column(
        raw,
        (
            "CNPJ_CIA",
            "CNPJ_COMPANHIA",
            "CNPJ_EMISSOR",
            "CNPJ",
        ),
        required=False,
    )

    cvm_col = _find_column(
        raw,
        (
            "CD_CVM",
            "CODIGO_CVM",
        ),
        required=False,
    )

    if cnpj_col is None and cvm_col is None:
        raise AccountingDataError(
            "FAIL-SAFE: FRE Capital Social por Classe de Ação "
            "sem identificador de emissor. "
            f"Colunas disponíveis: {sorted(raw.columns.tolist())}"
        )

    reference_col = _find_column(
        raw,
        (
            "DT_REFER",
            "DATA_REFERENCIA",
            "DT_REFERENCIA",
            "DATA_ULTIMA_ALTERACAO",
            "DT_ULTIMA_ALTERACAO",
        ),
        required=False,
    )

    if reference_col is None:
        reference_col = _find_column_by_tokens(
            raw,
            required_any=(
                "DT_REFER",
                "DATA_REFER",
                "ULTIMA_ALTERACAO",
            ),
        )

    if reference_col is None:
        raise AccountingDataError(
            "FAIL-SAFE: FRE Capital Social por Classe de Ação "
            "sem data de referência identificável de forma inequívoca. "
            f"Colunas disponíveis: {sorted(raw.columns.tolist())}"
        )

    version_col = _find_column(
        raw,
        (
            "VERSAO",
            "VERSAO_DOCUMENTO",
            "VERSAO_FRE",
        ),
        required=False,
    )

    species_col = _find_column(
        raw,
        (
            "ESPECIE_ACAO",
            "ESPECIE",
            "TIPO_ACAO",
            "TIPO_ESPECIE_ACAO",
        ),
        required=False,
    )

    if species_col is None:
        species_col = _find_column_by_tokens(
            raw,
            required_any=(
                "ESPECIE",
                "TIPO_ACAO",
            ),
            forbidden=(
                "QUANT",
                "QT_",
                "PERCENT",
                "VALOR",
            ),
        )

    class_col = _find_column(
        raw,
        (
            "CLASSE_ACAO",
            "CLASSE",
            "TIPO_CLASSE_ACAO",
        ),
        required=False,
    )

    if class_col is None:
        class_col = _find_column_by_tokens(
            raw,
            required_all=(
                "CLASSE",
            ),
            required_any=(
                "ACAO",
                "CLASSE",
            ),
            forbidden=(
                "QUANT",
                "QT_",
                "PERCENT",
                "VALOR",
            ),
        )

    # Alguns esquemas podem trazer espécie e classe numa única coluna.
    if species_col is None and class_col is None:
        raise AccountingDataError(
            "FAIL-SAFE: FRE Capital Social por Classe de Ação "
            "sem espécie/classe identificável de forma inequívoca. "
            f"Colunas disponíveis: {sorted(raw.columns.tolist())}"
        )

    quantity_col = _find_column(
        raw,
        (
            "QT_ACOES",
            "QT_ACAO",
            "QUANTIDADE_ACOES",
            "QUANTIDADE_ACAO",
            "QTD_ACOES",
            "QTD_ACAO",
            "QT_ACOES_EMITIDAS",
            "QUANTIDADE_ACOES_EMITIDAS",
            "QT_ACAO_CAPITAL_SOCIAL",
            "QT_ACOES_CAPITAL_SOCIAL",
        ),
        required=False,
    )

    if quantity_col is None:

        quantity_candidates = []

        for column in raw.columns:

            normalized = _normalize_column(
                column
            )

            has_quantity = (
                "QUANT" in normalized
                or normalized.startswith("QT_")
                or normalized.startswith("QTD_")
            )

            has_share = (
                "ACAO" in normalized
                or "ACOES" in normalized
            )

            forbidden_quantity = any(
                token in normalized
                for token in (
                    "PERCENT",
                    "VALOR",
                    "PRECO",
                    "CAPITAL_AUTORIZ",
                    "TESOUR",
                    "CIRCUL",
                )
            )

            if (
                has_quantity
                and has_share
                and not forbidden_quantity
            ):
                quantity_candidates.append(
                    column
                )

        quantity_candidates = list(
            dict.fromkeys(
                quantity_candidates
            )
        )

        if len(quantity_candidates) == 1:
            quantity_col = (
                quantity_candidates[0]
            )

    if quantity_col is None:
        raise AccountingDataError(
            "FAIL-SAFE: FRE Capital Social por Classe de Ação "
            "sem coluna única de quantidade de ações emitidas. "
            f"Colunas disponíveis: {sorted(raw.columns.tolist())}"
        )

    result = pd.DataFrame(
        index=raw.index
    )

    if cnpj_col is not None:
        result["ISSUER_ID"] = (
            raw[cnpj_col]
            .map(_cnpj)
        )
    else:
        result["ISSUER_ID"] = (
            raw[cvm_col]
            .map(_issuer_id)
        )

    result[
        "FRE_REFERENCE_DATE"
    ] = _parse_dates(
        raw[reference_col]
    )

    if version_col is not None:
        result[
            "FRE_VERSION"
        ] = pd.to_numeric(
            raw[version_col],
            errors="coerce",
        )
    else:
        result[
            "FRE_VERSION"
        ] = np.nan

    species_values = (
        raw[species_col]
        if species_col is not None
        else pd.Series(
            "",
            index=raw.index,
        )
    )

    class_values = (
        raw[class_col]
        if class_col is not None
        else pd.Series(
            "",
            index=raw.index,
        )
    )

    normalized_classes = [
        _normalize_share_class(
            species,
            share_class,
        )
        for species, share_class
        in zip(
            species_values,
            class_values,
        )
    ]

    result[
        "SHARE_CLASS"
    ] = [
        value[0]
        for value in normalized_classes
    ]

    result[
        "SHARE_SPECIES"
    ] = [
        value[1]
        for value in normalized_classes
    ]

    result[
        "SHARE_SUBCLASS"
    ] = [
        value[2]
        for value in normalized_classes
    ]

    result[
        "SHARES_CLASS_ISSUED"
    ] = _numeric(
        raw[quantity_col]
    )

    result[
        "FRE_SOURCE_YEAR"
    ] = int(year)

    result[
        "FRE_SOURCE_FILE"
    ] = filename

    result = result.loc[
        result["ISSUER_ID"].notna()
        & result[
            "FRE_REFERENCE_DATE"
        ].notna()
        & result[
            "SHARE_CLASS"
        ].notna()
        & result[
            "SHARES_CLASS_ISSUED"
        ].notna()
    ].copy()

    result = result.loc[
        result[
            "FRE_REFERENCE_DATE"
        ]
        <= context.accounting_cutoff
    ].copy()

    if result.empty:
        return result

    invalid = (
        result[
            "SHARES_CLASS_ISSUED"
        ] < 0
    )

    if invalid.any():
        raise AccountingDataError(
            "FAIL-SAFE: FRE contém quantidade negativa "
            "de ações por classe."
        )

    # Mantém a versão/data mais recente por emissor e classe.
    result = result.sort_values(
        [
            "ISSUER_ID",
            "SHARE_CLASS",
            "FRE_REFERENCE_DATE",
            "FRE_VERSION",
        ],
        na_position="first",
    )

    result = result.drop_duplicates(
        subset=[
            "ISSUER_ID",
            "SHARE_CLASS",
        ],
        keep="last",
    )

    # Uma linha final por emissor/classe é condição obrigatória.
    if result.duplicated(
        subset=[
            "ISSUER_ID",
            "SHARE_CLASS",
        ]
    ).any():
        raise AccountingDataError(
            "FAIL-SAFE: FRE produziu classe duplicada "
            "por emissor."
        )

    assert_no_future_information(
        result,
        stage="FRE capital social por classe",
    )

    return result.reset_index(
        drop=True
    )


def build_share_class_data(
    formation_date,
    accounting_cutoff,
) -> pd.DataFrame:
    """
    Produz quantidades por classe para a camada de Market Cap.

    Regra de tesouraria:
    - ON: usa tesouraria ON agregada DFP/ITR;
    - PN com uma única classe: usa tesouraria PN agregada;
    - PN com múltiplas classes:
        * se tesouraria PN == 0, outstanding == issued;
        * se tesouraria PN > 0 ou desconhecida, NÃO há rateio:
          SHARES_CLASS_OUTSTANDING permanece NaN.
    """

    context = create_accounting_context(
        formation_date,
        accounting_cutoff,
    )

    current_year = int(
        context.accounting_cutoff.year
    )

    # FRE é periódico/eventual. O ano corrente é a primeira fonte;
    # ano anterior é fallback por emissor/classe.
    frames = []

    for year, priority in (
        (current_year, 2),
        (current_year - 1, 1),
    ):

        try:
            frame = _read_fre_capital_classes(
                year,
                context,
            )
        except AccountingDataError:
            # Erro de esquema no ano corrente não pode ser mascarado.
            if year == current_year:
                raise
            frame = pd.DataFrame()

        if not frame.empty:
            frame = frame.copy()
            frame[
                "_FRE_SOURCE_PRIORITY"
            ] = priority
            frames.append(frame)

    if not frames:
        raise AccountingDataError(
            "FAIL-SAFE: nenhum Capital Social por Classe "
            "de Ação válido encontrado no FRE."
        )

    classes = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    classes = classes.sort_values(
        [
            "ISSUER_ID",
            "SHARE_CLASS",
            "FRE_REFERENCE_DATE",
            "_FRE_SOURCE_PRIORITY",
            "FRE_VERSION",
        ],
        na_position="first",
    )

    classes = classes.drop_duplicates(
        subset=[
            "ISSUER_ID",
            "SHARE_CLASS",
        ],
        keep="last",
    )

    classes = classes.drop(
        columns=[
            "_FRE_SOURCE_PRIORITY",
        ],
        errors="ignore",
    )

    capital = _build_capital_composition(
        context
    )

    if capital.empty:
        raise AccountingDataError(
            "FAIL-SAFE: composição DFP/ITR ausente; "
            "não é possível determinar ações em tesouraria."
        )

    treasury_columns = [
        column
        for column in (
            "ISSUER_ID",
            "SHARES_ON_TREASURY",
            "SHARES_PN_TREASURY",
        )
        if column in capital.columns
    ]

    if "ISSUER_ID" not in treasury_columns:
        raise AccountingDataError(
            "FAIL-SAFE: composição de capital sem ISSUER_ID."
        )

    classes = classes.merge(
        capital[
            treasury_columns
        ],
        on="ISSUER_ID",
        how="left",
        validate="many_to_one",
    )

    classes[
        "SHARES_CLASS_TREASURY"
    ] = np.nan

    classes[
        "SHARES_CLASS_OUTSTANDING"
    ] = np.nan

    classes[
        "SHARE_CLASS_EXACT"
    ] = False

    # ON é uma espécie sem subclasses econômicas no modelo de tickers
    # admitido pelo projeto. Só calculamos se a tesouraria ON é conhecida.
    on_mask = (
        classes[
            "SHARE_SPECIES"
        ] == "ON"
    )

    on_known = (
        on_mask
        & classes[
            "SHARES_ON_TREASURY"
        ].notna()
    )

    classes.loc[
        on_known,
        "SHARES_CLASS_TREASURY",
    ] = classes.loc[
        on_known,
        "SHARES_ON_TREASURY",
    ]

    classes.loc[
        on_known,
        "SHARES_CLASS_OUTSTANDING",
    ] = (
        classes.loc[
            on_known,
            "SHARES_CLASS_ISSUED",
        ]
        - classes.loc[
            on_known,
            "SHARES_CLASS_TREASURY",
        ]
    )

    classes.loc[
        on_known,
        "SHARE_CLASS_EXACT",
    ] = True

    # Para PN, a tesouraria DFP/ITR é agregada. Portanto só pode ser
    # atribuída a uma classe quando há exatamente uma classe PN, ou quando
    # a tesouraria agregada é zero.
    pn_mask = (
        classes[
            "SHARE_SPECIES"
        ] == "PN"
    )

    pn_counts = (
        classes.loc[
            pn_mask
        ]
        .groupby(
            "ISSUER_ID"
        )[
            "SHARE_CLASS"
        ]
        .nunique()
    )

    classes[
        "_PN_CLASS_COUNT"
    ] = (
        classes[
            "ISSUER_ID"
        ]
        .map(pn_counts)
    )

    single_pn = (
        pn_mask
        & (
            classes[
                "_PN_CLASS_COUNT"
            ] == 1
        )
        & classes[
            "SHARES_PN_TREASURY"
        ].notna()
    )

    zero_treasury_multi_pn = (
        pn_mask
        & (
            classes[
                "_PN_CLASS_COUNT"
            ] > 1
        )
        & (
            classes[
                "SHARES_PN_TREASURY"
            ] == 0
        )
    )

    pn_exact = (
        single_pn
        | zero_treasury_multi_pn
    )

    classes.loc[
        single_pn,
        "SHARES_CLASS_TREASURY",
    ] = classes.loc[
        single_pn,
        "SHARES_PN_TREASURY",
    ]

    classes.loc[
        zero_treasury_multi_pn,
        "SHARES_CLASS_TREASURY",
    ] = 0.0

    classes.loc[
        pn_exact,
        "SHARES_CLASS_OUTSTANDING",
    ] = (
        classes.loc[
            pn_exact,
            "SHARES_CLASS_ISSUED",
        ]
        - classes.loc[
            pn_exact,
            "SHARES_CLASS_TREASURY",
        ]
    )

    classes.loc[
        pn_exact,
        "SHARE_CLASS_EXACT",
    ] = True

    negative = (
        classes[
            "SHARES_CLASS_OUTSTANDING"
        ].notna()
        & (
            classes[
                "SHARES_CLASS_OUTSTANDING"
            ] < 0
        )
    )

    if negative.any():
        bad = (
            classes.loc[
                negative,
                [
                    "ISSUER_ID",
                    "SHARE_CLASS",
                    "SHARES_CLASS_ISSUED",
                    "SHARES_CLASS_TREASURY",
                ],
            ]
            .to_dict(
                orient="records"
            )
        )

        raise AccountingDataError(
            "FAIL-SAFE: outstanding por classe negativo. "
            f"Casos: {bad[:10]}"
        )

    classes[
        "FORMATION_DATE"
    ] = context.formation_date

    classes[
        "ACCOUNTING_CUTOFF"
    ] = context.accounting_cutoff

    classes[
        "PIT_VALID"
    ] = True

    classes[
        "FUTURE_RETURN_USED"
    ] = False

    classes = classes.drop(
        columns=[
            "_PN_CLASS_COUNT",
        ],
        errors="ignore",
    )

    output_columns = [
        column
        for column in (
            "ISSUER_ID",
            "SHARE_CLASS",
            "SHARE_SPECIES",
            "SHARE_SUBCLASS",
            "SHARES_CLASS_ISSUED",
            "SHARES_CLASS_TREASURY",
            "SHARES_CLASS_OUTSTANDING",
            "SHARE_CLASS_EXACT",
            "FRE_REFERENCE_DATE",
            "FRE_VERSION",
            "FRE_SOURCE_YEAR",
            "FRE_SOURCE_FILE",
            "FORMATION_DATE",
            "ACCOUNTING_CUTOFF",
            "PIT_VALID",
            "FUTURE_RETURN_USED",
        )
        if column in classes.columns
    ]

    classes = classes[
        output_columns
    ].copy()

    classes = classes.sort_values(
        [
            "ISSUER_ID",
            "SHARE_CLASS",
        ]
    ).reset_index(
        drop=True
    )

    assert_no_future_information(
        classes,
        stage="share_class_data final",
    )

    return classes



# =============================================================================
# FCA — IDENTIDADE TICKER → EMISSOR
# =============================================================================

def _find_fca_security_file(
    year: int,
) -> str:

    files = list_zip_files(
        "FCA",
        int(year),
    )

    candidates = []

    for filename in files:

        normalized = _normalize_text(
            Path(filename).name
        )

        if (
            "VALOR_MOBILIARIO"
            in normalized
        ):
            candidates.append(
                filename
            )

    if len(candidates) != 1:
        raise AccountingIdentityError(
            "FAIL-SAFE: arquivo FCA de valores "
            "mobiliários não identificado de forma "
            f"única. Encontrados: {candidates}"
        )

    return candidates[0]


def build_identity_map(
    year: int,
    *,
    allowed_tickers: Optional[
        Iterable[str]
    ] = None,
) -> pd.DataFrame:
    """
    Constrói:
        TICKER
        ISSUER_ID
        CNPJ
        ISSUER_NAME

    diretamente do FCA oficial.
    """

    zip_path = download_dataset(
        "FCA",
        int(year),
    )

    filename = _find_fca_security_file(
        int(year)
    )

    raw = read_csv_from_zip(
        "FCA",
        int(year),
        filename,
    )

    raw = _normalize_columns(
        raw
    )

    ticker_col = _find_column(
        raw,
        (
            "CODIGO_NEGOCIACAO",
            "CODIGO_DE_NEGOCIACAO",
        ),
    )

    cvm_col = _find_column(
        raw,
        (
            "CODIGO_CVM",
            "CD_CVM",
        ),
        required=False,
    )

    cnpj_col = _find_column(
        raw,
        (
            "CNPJ_COMPANHIA",
            "CNPJ_CIA",
            "CNPJ",
        ),
        required=False,
    )

    if cnpj_col is None and cvm_col is None:
        raise AccountingIdentityError(
            "FAIL-SAFE: FCA sem identificador de emissor. "
            "Esperado CNPJ_COMPANHIA/CNPJ_CIA/CNPJ ou CODIGO_CVM/CD_CVM."
        )

    name_col = _find_column(
        raw,
        (
            "NOME_EMPRESARIAL",
            "DENOMINACAO_SOCIAL",
        ),
        required=False,
    )

    result = pd.DataFrame()

    result["TICKER"] = (
        raw[ticker_col]
        .map(_ticker)
    )

    if cnpj_col is not None:
        result["CNPJ"] = (
            raw[cnpj_col]
            .map(_cnpj)
        )
        result["ISSUER_ID"] = result["CNPJ"]
    else:
        result["CNPJ"] = None
        result["ISSUER_ID"] = (
            raw[cvm_col]
            .map(_issuer_id)
        )

    if cvm_col is not None:
        result["CD_CVM"] = (
            raw[cvm_col]
            .map(_issuer_id)
        )
    else:
        result["CD_CVM"] = None

    if name_col is not None:
        result["ISSUER_NAME"] = (
            raw[name_col]
            .astype(str)
            .str.strip()
        )
    else:
        result["ISSUER_NAME"] = None

    result = result.loc[
        result["TICKER"].notna()
        & result["ISSUER_ID"].notna()
    ].copy()

    if allowed_tickers is not None:

        allowed = {
            str(t).strip().upper()
            for t in allowed_tickers
        }

        result = result.loc[
            result["TICKER"].isin(
                allowed
            )
        ].copy()

    # Um ticker não pode apontar para múltiplos
    # emissores dentro do snapshot utilizado.

    conflicts = (
        result.groupby("TICKER")[
            "ISSUER_ID"
        ]
        .nunique()
    )

    conflicts = conflicts[
        conflicts > 1
    ]

    if not conflicts.empty:
        raise AccountingIdentityError(
            "FAIL-SAFE: ticker associado a "
            "múltiplos emissores no FCA: "
            f"{conflicts.index.tolist()}"
        )

    result = (
        result.sort_values(
            [
                "TICKER",
                "ISSUER_ID",
            ]
        )
        .drop_duplicates(
            subset=["TICKER"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if result.empty:
        raise AccountingIdentityError(
            "FAIL-SAFE: identity_map FCA vazio."
        )

    assert_no_future_information(
        result,
        stage="FCA identity map",
    )

    return result


# =============================================================================
# NORMALIZAÇÃO CVM CONTÁBIL
# =============================================================================

def _prepare_statement(
    df: pd.DataFrame,
    context: AccountingContext,
) -> pd.DataFrame:

    if df.empty:
        return df.copy()

    result = df.copy()

    cnpj_col = _find_column(
        result,
        (
            "CNPJ_CIA",
            "CNPJ_COMPANHIA",
            "CNPJ",
        ),
        required=False,
    )

    cvm_col = _find_column(
        result,
        (
            "CD_CVM",
            "CODIGO_CVM",
        ),
        required=False,
    )

    if cnpj_col is None and cvm_col is None:
        raise AccountingDataError(
            "FAIL-SAFE: demonstração CVM sem identificador de emissor. "
            "Esperado CNPJ_CIA/CNPJ_COMPANHIA/CNPJ ou CD_CVM/CODIGO_CVM."
        )

    account_col = _find_column(
        result,
        (
            "CD_CONTA",
            "CODIGO_CONTA",
        ),
    )

    desc_col = _find_column(
        result,
        (
            "DS_CONTA",
            "DESCRICAO_CONTA",
        ),
        required=False,
    )

    value_col = _find_column(
        result,
        (
            "VL_CONTA",
            "VALOR_CONTA",
        ),
    )

    reference_col = _find_column(
        result,
        (
            "DT_REFER",
            "DATA_REFERENCIA",
        ),
    )

    start_col = _find_column(
        result,
        (
            "DT_INI_EXERC",
            "DATA_INICIO_EXERCICIO",
        ),
        required=False,
    )

    end_col = _find_column(
        result,
        (
            "DT_FIM_EXERC",
            "DATA_FIM_EXERCICIO",
        ),
        required=False,
    )

    order_col = _find_column(
        result,
        (
            "ORDEM_EXERC",
            "ORDEM_EXERCICIO",
        ),
        required=False,
    )

    if cnpj_col is not None:
        result["ISSUER_ID"] = (
            result[cnpj_col]
            .map(_cnpj)
        )
    else:
        result["ISSUER_ID"] = (
            result[cvm_col]
            .map(_issuer_id)
        )

    result["CD_CONTA_CANONICAL"] = (
        result[account_col]
        .astype(str)
        .str.strip()
    )

    if desc_col is not None:
        result["DS_CONTA_CANONICAL"] = (
            result[desc_col]
            .map(_normalize_text)
        )
    else:
        result["DS_CONTA_CANONICAL"] = ""

    result["VL_CONTA_CANONICAL"] = (
        _numeric(
            result[value_col]
        )
    )

    result["DT_REFER_CANONICAL"] = (
        _parse_dates(
            result[reference_col]
        )
    )

    if start_col is not None:
        result["DT_INI_CANONICAL"] = (
            _parse_dates(
                result[start_col]
            )
        )
    else:
        result["DT_INI_CANONICAL"] = (
            pd.NaT
        )

    if end_col is not None:
        result["DT_FIM_CANONICAL"] = (
            _parse_dates(
                result[end_col]
            )
        )
    else:
        result["DT_FIM_CANONICAL"] = (
            result[
                "DT_REFER_CANONICAL"
            ]
        )

    if order_col is not None:
        result["ORDEM_CANONICAL"] = (
            result[order_col]
            .map(_normalize_text)
        )
    else:
        result["ORDEM_CANONICAL"] = ""

    result = result.loc[
        result["ISSUER_ID"].notna()
        & result[
            "DT_REFER_CANONICAL"
        ].notna()
        & result[
            "VL_CONTA_CANONICAL"
        ].notna()
    ].copy()

    # PIT:
    # nenhuma demonstração com referência posterior
    # ao cutoff contábil pode entrar.

    result = result.loc[
        result[
            "DT_REFER_CANONICAL"
        ]
        <= context.accounting_cutoff
    ].copy()

    if result.empty:
        return result

    # Preferência pela coluna "ÚLTIMO", quando presente.
    if (
        result["ORDEM_CANONICAL"]
        .str.contains(
            "ULTIMO",
            na=False,
        )
        .any()
    ):
        result = result.loc[
            result["ORDEM_CANONICAL"]
            .str.contains(
                "ULTIMO",
                na=False,
            )
        ].copy()

    assert_no_future_information(
        result,
        stage="CVM accounting statement",
    )

    return result


# =============================================================================
# SELEÇÃO DE CONTAS
# =============================================================================

def _account_match(
    df: pd.DataFrame,
    *,
    codes: Iterable[str],
    descriptions: Iterable[str] = (),
) -> pd.DataFrame:

    if df.empty:
        return df.copy()

    code_set = {
        str(code).strip()
        for code in codes
    }

    mask_code = (
        df["CD_CONTA_CANONICAL"]
        .isin(code_set)
    )

    matched = df.loc[
        mask_code
    ].copy()

    if not matched.empty:
        return matched

    description_set = {
        _normalize_text(desc)
        for desc in descriptions
    }

    if not description_set:
        return matched

    mask_desc = (
        df["DS_CONTA_CANONICAL"]
        .isin(description_set)
    )

    return df.loc[
        mask_desc
    ].copy()


def _latest_stock_value(
    df: pd.DataFrame,
    rule: dict,
) -> pd.DataFrame:

    matched = _account_match(
        df,
        codes=rule.get(
            "codes",
            (),
        ),
        descriptions=rule.get(
            "descriptions",
            (),
        ),
    )

    if matched.empty:
        return pd.DataFrame(
            columns=[
                "ISSUER_ID",
                "VALUE",
                "REFERENCE_DATE",
            ]
        )

    matched = matched.sort_values(
        [
            "ISSUER_ID",
            "DT_REFER_CANONICAL",
        ]
    )

    matched = matched.drop_duplicates(
        subset=["ISSUER_ID"],
        keep="last",
    )

    return pd.DataFrame(
        {
            "ISSUER_ID":
                matched["ISSUER_ID"],
            "VALUE":
                matched[
                    "VL_CONTA_CANONICAL"
                ],
            "REFERENCE_DATE":
                matched[
                    "DT_REFER_CANONICAL"
                ],
        }
    ).reset_index(drop=True)


def _latest_flow_value(
    df: pd.DataFrame,
    rule: dict,
) -> pd.DataFrame:

    matched = _account_match(
        df,
        codes=rule.get(
            "codes",
            (),
        ),
        descriptions=rule.get(
            "descriptions",
            (),
        ),
    )

    if matched.empty:
        return pd.DataFrame(
            columns=[
                "ISSUER_ID",
                "VALUE",
                "REFERENCE_DATE",
            ]
        )

    # Para DRE/DFC escolhemos o registro mais recente
    # disponível no PIT e, na mesma data, o período
    # acumulado mais longo.

    matched["PERIOD_DAYS"] = (
        matched["DT_FIM_CANONICAL"]
        - matched["DT_INI_CANONICAL"]
    ).dt.days

    matched["PERIOD_DAYS"] = (
        matched["PERIOD_DAYS"]
        .fillna(-1)
    )

    matched = matched.sort_values(
        [
            "ISSUER_ID",
            "DT_REFER_CANONICAL",
            "PERIOD_DAYS",
        ]
    )

    matched = matched.drop_duplicates(
        subset=["ISSUER_ID"],
        keep="last",
    )

    return pd.DataFrame(
        {
            "ISSUER_ID":
                matched["ISSUER_ID"],
            "VALUE":
                matched[
                    "VL_CONTA_CANONICAL"
                ],
            "REFERENCE_DATE":
                matched[
                    "DT_REFER_CANONICAL"
                ],
        }
    ).reset_index(drop=True)


def _merge_value(
    base: pd.DataFrame,
    values: pd.DataFrame,
    column: str,
) -> pd.DataFrame:

    temp = values[
        [
            "ISSUER_ID",
            "VALUE",
        ]
    ].rename(
        columns={
            "VALUE": column,
        }
    )

    return base.merge(
        temp,
        on="ISSUER_ID",
        how="left",
        validate="one_to_one",
    )


# =============================================================================
# DÍVIDA
# =============================================================================

def _build_debt(
    bpp: pd.DataFrame,
) -> pd.DataFrame:

    cp = _latest_stock_value(
        bpp,
        DEBT_RULES["DIVIDA_CP"],
    )

    lp = _latest_stock_value(
        bpp,
        DEBT_RULES["DIVIDA_LP"],
    )

    cp = cp[
        [
            "ISSUER_ID",
            "VALUE",
        ]
    ].rename(
        columns={
            "VALUE": "DIVIDA_CP",
        }
    )

    lp = lp[
        [
            "ISSUER_ID",
            "VALUE",
        ]
    ].rename(
        columns={
            "VALUE": "DIVIDA_LP",
        }
    )

    result = cp.merge(
        lp,
        on="ISSUER_ID",
        how="outer",
        validate="one_to_one",
    )

    # Não transformamos ausência completa em zero.
    both_missing = (
        result["DIVIDA_CP"].isna()
        & result["DIVIDA_LP"].isna()
    )

    result["DIVIDA_BRUTA"] = (
        result[
            [
                "DIVIDA_CP",
                "DIVIDA_LP",
            ]
        ]
        .fillna(0.0)
        .sum(axis=1)
    )

    result.loc[
        both_missing,
        "DIVIDA_BRUTA",
    ] = np.nan

    return result[
        [
            "ISSUER_ID",
            "DIVIDA_BRUTA",
        ]
    ]


# =============================================================================
# PL ANTERIOR
# =============================================================================

def _previous_equity(
    bpp_history: pd.DataFrame,
    current_reference_date: pd.Timestamp,
) -> pd.DataFrame:

    if bpp_history.empty:
        return pd.DataFrame(
            columns=[
                "ISSUER_ID",
                "PL_ANTERIOR",
            ]
        )

    rule = ACCOUNT_RULES["PL"]

    matched = _account_match(
        bpp_history,
        codes=rule["codes"],
        descriptions=rule[
            "descriptions"
        ],
    )

    if matched.empty:
        return pd.DataFrame(
            columns=[
                "ISSUER_ID",
                "PL_ANTERIOR",
            ]
        )

    matched = matched.loc[
        matched[
            "DT_REFER_CANONICAL"
        ]
        < current_reference_date
    ].copy()

    if matched.empty:
        return pd.DataFrame(
            columns=[
                "ISSUER_ID",
                "PL_ANTERIOR",
            ]
        )

    matched = matched.sort_values(
        [
            "ISSUER_ID",
            "DT_REFER_CANONICAL",
        ]
    )

    matched = matched.drop_duplicates(
        subset=["ISSUER_ID"],
        keep="last",
    )

    return (
        matched[
            [
                "ISSUER_ID",
                "VL_CONTA_CANONICAL",
            ]
        ]
        .rename(
            columns={
                "VL_CONTA_CANONICAL":
                    "PL_ANTERIOR",
            }
        )
        .reset_index(drop=True)
    )


# =============================================================================
# CONSTRUÇÃO CONTÁBIL
# =============================================================================

def build_accounting_data(
    formation_date,
    accounting_cutoff,
    *,
    identity_map: Optional[
        pd.DataFrame
    ] = None,
) -> pd.DataFrame:
    """
    Produz uma linha por emissor.

    O output é compatível com engines/fundamentals.py.
    """

    context = create_accounting_context(
        formation_date,
        accounting_cutoff,
    )

    current_year = (
        context.accounting_cutoff.year
    )

    previous_year = current_year - 1

    # -------------------------------------------------------------------------
    # ITR do ano corrente
    # -------------------------------------------------------------------------

    itr_raw = _read_statement_files(
        "ITR",
        current_year,
    )

    itr = {
        key: _prepare_statement(
            value,
            context,
        )
        for key, value
        in itr_raw.items()
    }

    # -------------------------------------------------------------------------
    # DFP anterior
    #
    # Serve principalmente para PL_ANTERIOR e como histórico disponível
    # antes do cutoff.
    # -------------------------------------------------------------------------

    dfp_previous_raw = (
        _read_statement_files(
            "DFP",
            previous_year,
        )
    )

    dfp_previous = {
        key: _prepare_statement(
            value,
            context,
        )
        for key, value
        in dfp_previous_raw.items()
    }

    # -------------------------------------------------------------------------
    # Base de emissores
    # -------------------------------------------------------------------------

    issuer_sets = []

    for statements in (
        itr,
        dfp_previous,
    ):
        for frame in statements.values():

            if (
                not frame.empty
                and "ISSUER_ID"
                in frame.columns
            ):
                issuer_sets.extend(
                    frame["ISSUER_ID"]
                    .dropna()
                    .tolist()
                )

    accounting_issuers = set(
        issuer_sets
    )

    # Em produção, quando existe identity_map filtrado pelos tickers
    # efetivamente negociáveis do Market Engine, a base contábil deve
    # permanecer no mesmo universo econômico. Isso impede que emissores
    # presentes nos ZIPs CVM, mas ausentes do universo B3 formado,
    # avancem para Fundamentals/Quality/Valuation/Ranking.
    if identity_map is not None:

        identity_for_scope = (
            identity_map.copy()
        )

        if "ISSUER_ID" not in identity_for_scope.columns:
            raise AccountingIdentityError(
                "FAIL-SAFE: identity_map sem ISSUER_ID "
                "para restringir a base contábil."
            )

        market_issuers = {
            _issuer_id(value)
            for value in identity_for_scope[
                "ISSUER_ID"
            ]
            .dropna()
            .tolist()
        }

        market_issuers.discard(None)

        issuers = sorted(
            accounting_issuers
            & market_issuers
        )

    else:
        issuers = sorted(
            accounting_issuers
        )

    if not issuers:
        raise AccountingDataError(
            "FAIL-SAFE: nenhum emissor "
            "contábil encontrado no universo elegível."
        )

    base = pd.DataFrame(
        {
            "ISSUER_ID": issuers,
        }
    )

    # -------------------------------------------------------------------------
    # BALANÇO
    # -------------------------------------------------------------------------

    for column, rule in (
        ACCOUNT_RULES.items()
    ):

        statement = rule[
            "statement"
        ]

        source = itr.get(
            statement,
            pd.DataFrame(),
        )

        values = _latest_stock_value(
            source,
            rule,
        )

        base = _merge_value(
            base,
            values,
            column,
        )

    # -------------------------------------------------------------------------
    # FLUXOS
    # -------------------------------------------------------------------------

    for column, rule in (
        FLOW_RULES.items()
    ):

        statement = rule[
            "statement"
        ]

        source = itr.get(
            statement,
            pd.DataFrame(),
        )

        values = _latest_flow_value(
            source,
            rule,
        )

        base = _merge_value(
            base,
            values,
            column,
        )

    # -------------------------------------------------------------------------
    # DÍVIDA
    # -------------------------------------------------------------------------

    debt = _build_debt(
        itr.get(
            "BPP",
            pd.DataFrame(),
        )
    )

    base = base.merge(
        debt,
        on="ISSUER_ID",
        how="left",
        validate="one_to_one",
    )

    # -------------------------------------------------------------------------
    # CAIXA
    # -------------------------------------------------------------------------

    cash = _latest_stock_value(
        itr.get(
            "BPA",
            pd.DataFrame(),
        ),
        CASH_RULE,
    )

    base = _merge_value(
        base,
        cash,
        "CAIXA",
    )

    # -------------------------------------------------------------------------
    # PL ANTERIOR
    # -------------------------------------------------------------------------

    previous_bpp = (
        dfp_previous.get(
            "BPP",
            pd.DataFrame(),
        )
    )

    current_reference = (
        context.accounting_cutoff
    )

    previous_equity = (
        _previous_equity(
            previous_bpp,
            current_reference,
        )
    )

    base = base.merge(
        previous_equity,
        on="ISSUER_ID",
        how="left",
        validate="one_to_one",
    )

    # -------------------------------------------------------------------------
    # COMPOSIÇÃO DO CAPITAL
    # -------------------------------------------------------------------------

    capital = _build_capital_composition(
        context
    )

    if not capital.empty:

        capital_columns = [
            column
            for column in (
                "ISSUER_ID",
                "CAPITAL_REFERENCE_DATE",
                "CAPITAL_VERSION",
                "SHARES_ON_ISSUED",
                "SHARES_PN_ISSUED",
                "SHARES_TOTAL_ISSUED",
                "SHARES_ON_TREASURY",
                "SHARES_PN_TREASURY",
                "SHARES_TOTAL_TREASURY",
                "SHARES_ON_OUTSTANDING",
                "SHARES_PN_OUTSTANDING",
                "SHARES_OUTSTANDING",
                "CAPITAL_SOURCE_DATASET",
                "CAPITAL_SOURCE_YEAR",
                "CAPITAL_SOURCE_FILE",
            )
            if column in capital.columns
        ]

        base = base.merge(
            capital[
                capital_columns
            ],
            on="ISSUER_ID",
            how="left",
            validate="one_to_one",
        )

    else:
        # Ausência da seção não é convertida em quantidade artificial.
        base[
            "SHARES_ON_OUTSTANDING"
        ] = np.nan

        base[
            "SHARES_PN_OUTSTANDING"
        ] = np.nan

        base[
            "SHARES_OUTSTANDING"
        ] = np.nan

    # -------------------------------------------------------------------------
    # IDENTIDADE
    # -------------------------------------------------------------------------

    if identity_map is not None:

        identity = (
            identity_map.copy()
        )

        required = {
            "TICKER",
            "ISSUER_ID",
        }

        missing = (
            required
            - set(identity.columns)
        )

        if missing:
            raise AccountingIdentityError(
                "identity_map sem colunas: "
                f"{sorted(missing)}"
            )

        identity[
            "ISSUER_ID"
        ] = identity[
            "ISSUER_ID"
        ].map(
            _issuer_id
        )

        issuer_identity = (
            identity.sort_values(
                [
                    "ISSUER_ID",
                    "TICKER",
                ]
            )
            .groupby(
                "ISSUER_ID",
                as_index=False,
            )
            .agg(
                TICKERS=(
                    "TICKER",
                    lambda x:
                    ",".join(
                        sorted(
                            set(x)
                        )
                    ),
                )
            )
        )

        base = base.merge(
            issuer_identity,
            on="ISSUER_ID",
            how="left",
            validate="one_to_one",
        )

    # -------------------------------------------------------------------------
    # METADADOS PIT
    # -------------------------------------------------------------------------

    base[
        "FORMATION_DATE"
    ] = context.formation_date

    base[
        "ACCOUNTING_CUTOFF"
    ] = context.accounting_cutoff

    base[
        "PIT_VALID"
    ] = True

    base[
        "FUTURE_RETURN_USED"
    ] = False

    base[
        "ACCOUNTING_ENGINE_OK"
    ] = True

    # -------------------------------------------------------------------------
    # SANIDADE
    # -------------------------------------------------------------------------

    if base[
        "ISSUER_ID"
    ].duplicated().any():

        raise AccountingDataError(
            "FAIL-SAFE: emissor duplicado "
            "no accounting_data."
        )

    assert_no_future_information(
        base,
        stage="accounting_data final",
    )

    base = base.sort_values(
        "ISSUER_ID"
    ).reset_index(
        drop=True
    )

    return base


# =============================================================================
# PIPELINE DE PRODUÇÃO
# =============================================================================

def build_production_accounting(
    formation_date,
    accounting_cutoff,
    *,
    market_tickers: Optional[
        Iterable[str]
    ] = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Constrói automaticamente:

        identity_map
        accounting_data

    para o main.py.
    """

    context = create_accounting_context(
        formation_date,
        accounting_cutoff,
    )

    identity = build_identity_map(
        context.formation_date.year,
        allowed_tickers=market_tickers,
    )

    accounting = (
        build_accounting_data(
            formation_date=
                context.formation_date,
            accounting_cutoff=
                context.accounting_cutoff,
            identity_map=identity,
        )
    )

    return identity, accounting


# =============================================================================
# AUDITORIA
# =============================================================================

def audit_accounting_data(
    accounting_data: pd.DataFrame,
) -> dict:

    if accounting_data is None:
        raise AccountingDataError(
            "accounting_data ausente."
        )

    total = int(
        len(accounting_data)
    )

    if total == 0:
        raise AccountingDataError(
            "accounting_data vazio."
        )

    core = (
        "ATIVO_TOTAL",
        "PL",
        "RECEITA",
    )

    coverage = {}

    for column in core:

        if column in accounting_data:
            coverage[column] = float(
                accounting_data[
                    column
                ]
                .notna()
                .mean()
            )
        else:
            coverage[column] = 0.0

    return {
        "n_issuers": total,
        "coverage": coverage,
        "formation_date":
            str(
                accounting_data[
                    "FORMATION_DATE"
                ].iloc[0]
            )
            if "FORMATION_DATE"
            in accounting_data
            else None,
        "accounting_cutoff":
            str(
                accounting_data[
                    "ACCOUNTING_CUTOFF"
                ].iloc[0]
            )
            if "ACCOUNTING_CUTOFF"
            in accounting_data
            else None,
        "future_return_used": False,
    }


# =============================================================================
# SAVE
# =============================================================================

def save_accounting_data(
    accounting_data: pd.DataFrame,
    identity_map: Optional[
        pd.DataFrame
    ] = None,
    *,
    prefix: str = "accounting",
) -> dict[str, Path]:

    ACCOUNTING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {}

    accounting_path = (
        ACCOUNTING_DIR
        / f"{prefix}_data.csv"
    )

    accounting_data.to_csv(
        accounting_path,
        index=False,
    )

    output[
        "accounting_data"
    ] = accounting_path

    if identity_map is not None:

        identity_path = (
            ACCOUNTING_DIR
            / f"{prefix}_identity_map.csv"
        )

        identity_map.to_csv(
            identity_path,
            index=False,
        )

        output[
            "identity_map"
        ] = identity_path

    return output


# =============================================================================
# SELF TEST
# =============================================================================

def _self_test() -> None:

    print("=" * 72)
    print(
        "B3 STOCK SELECTOR — "
        "ACCOUNTING ENGINE"
    )
    print("=" * 72)

    context = (
        create_accounting_context(
            formation_date=
                "2026-10-05",
            accounting_cutoff=
                "2026-06-30",
        )
    )

    assert (
        context.accounting_cutoff
        <= context.formation_date
    )

    sample = pd.DataFrame(
        {
            "CNPJ_CIA": [
                "12.345.678/0001-90",
                "12.345.678/0001-90",
            ],
            "CD_CVM": [
                "1234",
                "1234",
            ],
            "CD_CONTA": [
                "1",
                "1",
            ],
            "DS_CONTA": [
                "Ativo Total",
                "Ativo Total",
            ],
            "VL_CONTA": [
                "1000",
                "2000",
            ],
            "DT_REFER": [
                "2026-06-30",
                "2026-12-31",
            ],
            "ORDEM_EXERC": [
                "ÚLTIMO",
                "ÚLTIMO",
            ],
            "STATEMENT": [
                "BPA",
                "BPA",
            ],
        }
    )

    sample = _normalize_columns(
        sample
    )

    prepared = _prepare_statement(
        sample,
        context,
    )

    if len(prepared) != 1:
        raise AccountingPITError(
            "SELF-TEST: filtro PIT falhou."
        )

    value = _latest_stock_value(
        prepared,
        ACCOUNT_RULES[
            "ATIVO_TOTAL"
        ],
    )

    if value.empty:
        raise AccountingDataError(
            "SELF-TEST: conta não encontrada."
        )

    obtained = float(
        value["VALUE"].iloc[0]
    )

    if obtained != 1000.0:
        raise AccountingDataError(
            "SELF-TEST: valor incorreto."
        )

    print("PIT: OK")
    print("Normalização: OK")
    print("Seleção de conta: OK")
    print("Future return: BLOQUEADO")
    print("Status: OK")
    print("=" * 72)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    _self_test()
