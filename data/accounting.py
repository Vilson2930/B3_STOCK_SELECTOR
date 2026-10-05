"""
B3 STOCK SELECTOR — ACCOUNTING ENGINE

Responsabilidade
----------------
Transformar os arquivos oficiais brutos da CVM (FCA / ITR / DFP)
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
    )

    cnpj_col = _find_column(
        raw,
        (
            "CNPJ_COMPANHIA",
            "CNPJ",
        ),
        required=False,
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

    result["ISSUER_ID"] = (
        raw[cvm_col]
        .map(_issuer_id)
    )

    if cnpj_col is not None:
        result["CNPJ"] = (
            raw[cnpj_col]
            .map(_cnpj)
        )
    else:
        result["CNPJ"] = None

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

    cvm_col = _find_column(
        result,
        (
            "CD_CVM",
            "CODIGO_CVM",
        ),
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

    issuers = sorted(
        set(issuer_sets)
    )

    if not issuers:
        raise AccountingDataError(
            "FAIL-SAFE: nenhum emissor "
            "contábil encontrado."
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
