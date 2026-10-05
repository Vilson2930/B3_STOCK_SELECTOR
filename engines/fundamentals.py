# =============================================================================
# B3 STOCK SELECTOR
# Arquivo: engines/fundamentals.py
#
# FUNDAMENTAL ENGINE
#
# OBJETIVO:
# Transformar dados contábeis Point-in-Time em indicadores econômicos
# padronizados por emissor.
#
# RESPONSABILIDADE:
# - Validar base fundamental
# - Trabalhar em nível de ISSUER
# - Calcular margens
# - Calcular rentabilidade
# - Calcular estrutura de capital
# - Calcular eficiência
# - Calcular geração de caixa
# - Criar flags econômicos
# - Registrar cobertura e qualidade dos dados
#
# NÃO FAZ:
# - Quality Score
# - Turnaround Score
# - Valuation Score
# - Ranking
# - Otimização de fatores
# - Retorno futuro
#
# REGRA:
# Este módulo CALCULA indicadores.
# Ele NÃO decide se alto ou baixo é melhor.
# =============================================================================

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

try:
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


# =============================================================================
# DIRETÓRIOS
# =============================================================================

FUNDAMENTAL_DIR = (
    OUTPUT_DIR
    / "fundamentals"
)

FUNDAMENTAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# EXCEÇÕES
# =============================================================================

class FundamentalError(RuntimeError):
    """Erro geral do Fundamental Engine."""


class FundamentalIntegrityError(FundamentalError):
    """Erro de integridade da base fundamental."""


class FundamentalPITError(FundamentalError):
    """Erro relacionado ao Point-in-Time."""


# =============================================================================
# NOMES CANÔNICOS
#
# O pipeline anterior deve entregar, quando disponível, estas contas.
#
# Nem todas são obrigatórias para todas as empresas.
# O engine preserva missing data e registra cobertura.
# =============================================================================

BASE_FIELDS = (
    "ATIVO_TOTAL",
    "ATIVO_CIRCULANTE",
    "PASSIVO_CIRCULANTE",
    "PL",
    "PL_ANTERIOR",
    "RECEITA",
    "CUSTO",
    "LUCRO_BRUTO",
    "EBIT",
    "LUCRO_LIQUIDO",
    "FCO",
    "DIVIDA_BRUTA",
    "CAIXA",
    "ESTOQUES",
    "IMOBILIZADO",
    "INTANGIVEL",
)


# =============================================================================
# INDICADORES PRODUZIDOS
# =============================================================================

OUTPUT_FACTORS = (
    "MARGEM_BRUTA",
    "MARGEM_EBIT",
    "MARGEM_LIQUIDA",
    "ROA",
    "ROE",
    "GIRO_ATIVO",
    "FCO_RECEITA",
    "FCO_ATIVO",
    "DIVIDA_BRUTA_ATIVO",
    "DIVIDA_BRUTA_PL",
    "DIVIDA_LIQUIDA",
    "DIVIDA_LIQUIDA_ATIVO",
    "CAPITAL_GIRO",
    "CAPITAL_GIRO_ATIVO",
    "ESTOQUES_ATIVO",
    "IMOBILIZADO_ATIVO",
    "INTANGIVEL_ATIVO",
)


# =============================================================================
# FLAGS
# =============================================================================

OUTPUT_FLAGS = (
    "LUCRO_POSITIVO",
    "EBIT_POSITIVO",
    "FCO_POSITIVO",
    "CAPITAL_GIRO_POSITIVO",
    "PL_POSITIVO",
)


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
        raise FundamentalIntegrityError(
            f"{name} não é DataFrame."
        )

    if df.empty:
        raise FundamentalIntegrityError(
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
        raise FundamentalIntegrityError(
            f"FAIL-SAFE: {name} sem colunas obrigatórias: "
            f"{sorted(missing)}"
        )


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
# CONVERSÃO NUMÉRICA
# =============================================================================

def _numeric(
    series: pd.Series,
) -> pd.Series:
    """
    Converte valores para float sem preencher missing data.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    ).astype(float)


# =============================================================================
# DIVISÃO SEGURA
# =============================================================================

def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Divisão segura.

    Denominador:
        0    -> NaN
        NaN  -> NaN

    Valores infinitos nunca são permitidos.
    """

    numerator = _numeric(
        numerator
    )

    denominator = _numeric(
        denominator
    )

    valid = (
        numerator.notna()
        &
        denominator.notna()
        &
        denominator.ne(0)
    )

    result = pd.Series(
        np.nan,
        index=numerator.index,
        dtype=float,
    )

    result.loc[
        valid
    ] = (
        numerator.loc[
            valid
        ]
        /
        denominator.loc[
            valid
        ]
    )

    result = result.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return result


# =============================================================================
# COLUNA OPCIONAL
# =============================================================================

def _get_numeric_column(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:

    if column not in df.columns:

        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return _numeric(
        df[column]
    )


# =============================================================================
# PREPARAÇÃO
# =============================================================================

def prepare_fundamental_base(
    df: pd.DataFrame,
    context: PITContext,
) -> pd.DataFrame:
    """
    Prepara uma linha por emissor.

    A base recebida deve já representar um snapshot PIT.
    Este módulo valida novamente o cutoff.
    """

    _require_columns(
        df,
        ["ISSUER_ID"],
        "fundamental_base",
    )

    result = (
        df
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
        raise FundamentalIntegrityError(
            "FAIL-SAFE: ISSUER_ID ausente."
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

        raise FundamentalIntegrityError(
            "FAIL-SAFE: Fundamental Engine exige "
            "uma linha por emissor. "
            f"Duplicados={examples}"
        )

    # -------------------------------------------------------------------------
    # VALIDAÇÃO PIT
    # -------------------------------------------------------------------------

    date_columns = (
        "ACCOUNTING_CUTOFF",
        "PIT_CUTOFF",
        "DATA_CORTE",
    )

    pit_column = None

    for column in date_columns:

        if column in result.columns:
            pit_column = column
            break

    if pit_column is None:

        if "PIT_VALID" not in result.columns:

            raise FundamentalPITError(
                "FAIL-SAFE: não existe evidência "
                "de validação Point-in-Time."
            )

        pit_valid = (
            result[
                "PIT_VALID"
            ]
        )

        if pd.api.types.is_bool_dtype(
            pit_valid
        ):

            valid = (
                pit_valid
                .fillna(False)
                .astype(bool)
            )

        else:

            valid = (
                pit_valid
                .fillna("")
                .astype(str)
                .str.upper()
                .str.strip()
                .isin(
                    {
                        "TRUE",
                        "1",
                        "YES",
                        "SIM",
                        "OK",
                        "VALID",
                    }
                )
            )

        if not valid.all():

            raise FundamentalPITError(
                "FAIL-SAFE: existem emissores "
                "sem validação PIT."
            )

    else:

        dates = pd.to_datetime(
            result[
                pit_column
            ],
            errors="coerce",
        )

        invalid = (
            dates.isna()
            |
            (
                dates.dt.normalize()
                >
                context.accounting_cutoff
            )
        )

        if invalid.any():

            examples = (
                result.loc[
                    invalid,
                    [
                        "ISSUER_ID",
                        pit_column,
                    ],
                ]
                .head(20)
                .to_dict(
                    orient="records"
                )
            )

            raise FundamentalPITError(
                "FAIL-SAFE: dado fundamental posterior "
                "ao accounting cutoff. "
                f"Exemplos={examples}"
            )

    return result


# =============================================================================
# LUCRO BRUTO
# =============================================================================

def calculate_gross_profit(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Prioridade:
        1. LUCRO_BRUTO informado
        2. RECEITA + CUSTO

    Na CVM, custo frequentemente é registrado com sinal negativo.
    Nesse caso:
        lucro bruto = receita + custo

    Se custo vier positivo:
        lucro bruto = receita - custo

    O sinal é tratado linha a linha.
    """

    reported = _get_numeric_column(
        df,
        "LUCRO_BRUTO",
    )

    revenue = _get_numeric_column(
        df,
        "RECEITA",
    )

    cost = _get_numeric_column(
        df,
        "CUSTO",
    )

    calculated = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    valid = (
        revenue.notna()
        &
        cost.notna()
    )

    negative_cost = (
        valid
        &
        cost.lt(0)
    )

    positive_cost = (
        valid
        &
        cost.ge(0)
    )

    calculated.loc[
        negative_cost
    ] = (
        revenue.loc[
            negative_cost
        ]
        +
        cost.loc[
            negative_cost
        ]
    )

    calculated.loc[
        positive_cost
    ] = (
        revenue.loc[
            positive_cost
        ]
        -
        cost.loc[
            positive_cost
        ]
    )

    return reported.combine_first(
        calculated
    )


# =============================================================================
# ROE COM PL MÉDIO
# =============================================================================

def calculate_roe(
    df: pd.DataFrame,
) -> pd.Series:
    """
    ROE = Lucro Líquido / Patrimônio Líquido Médio.

    Metodologia preservada do estudo:

        PL_MEDIO = (PL atual + PL anterior) / 2

    Não substitui silenciosamente pelo PL final quando
    PL_ANTERIOR estiver ausente.
    """

    profit = _get_numeric_column(
        df,
        "LUCRO_LIQUIDO",
    )

    equity = _get_numeric_column(
        df,
        "PL",
    )

    previous_equity = (
        _get_numeric_column(
            df,
            "PL_ANTERIOR",
        )
    )

    average_equity = (
        equity
        +
        previous_equity
    ) / 2.0

    return safe_divide(
        profit,
        average_equity,
    )


# =============================================================================
# INDICADORES
# =============================================================================

def calculate_factors(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        df
        .copy()
    )

    # -------------------------------------------------------------------------
    # CONTAS BASE
    # -------------------------------------------------------------------------

    ativo = _get_numeric_column(
        result,
        "ATIVO_TOTAL",
    )

    ativo_circ = _get_numeric_column(
        result,
        "ATIVO_CIRCULANTE",
    )

    passivo_circ = _get_numeric_column(
        result,
        "PASSIVO_CIRCULANTE",
    )

    pl = _get_numeric_column(
        result,
        "PL",
    )

    receita = _get_numeric_column(
        result,
        "RECEITA",
    )

    ebit = _get_numeric_column(
        result,
        "EBIT",
    )

    lucro = _get_numeric_column(
        result,
        "LUCRO_LIQUIDO",
    )

    fco = _get_numeric_column(
        result,
        "FCO",
    )

    divida = _get_numeric_column(
        result,
        "DIVIDA_BRUTA",
    )

    caixa = _get_numeric_column(
        result,
        "CAIXA",
    )

    estoques = _get_numeric_column(
        result,
        "ESTOQUES",
    )

    imobilizado = _get_numeric_column(
        result,
        "IMOBILIZADO",
    )

    intangivel = _get_numeric_column(
        result,
        "INTANGIVEL",
    )

    # -------------------------------------------------------------------------
    # LUCRO BRUTO
    # -------------------------------------------------------------------------

    lucro_bruto = (
        calculate_gross_profit(
            result
        )
    )

    result[
        "LUCRO_BRUTO_CALCULADO"
    ] = lucro_bruto

    # -------------------------------------------------------------------------
    # MARGENS
    # -------------------------------------------------------------------------

    result[
        "MARGEM_BRUTA"
    ] = safe_divide(
        lucro_bruto,
        receita,
    )

    result[
        "MARGEM_EBIT"
    ] = safe_divide(
        ebit,
        receita,
    )

    result[
        "MARGEM_LIQUIDA"
    ] = safe_divide(
        lucro,
        receita,
    )

    # -------------------------------------------------------------------------
    # RENTABILIDADE
    # -------------------------------------------------------------------------

    result[
        "ROA"
    ] = safe_divide(
        lucro,
        ativo,
    )

    result[
        "ROE"
    ] = calculate_roe(
        result
    )

    # -------------------------------------------------------------------------
    # EFICIÊNCIA
    # -------------------------------------------------------------------------

    result[
        "GIRO_ATIVO"
    ] = safe_divide(
        receita,
        ativo,
    )

    # -------------------------------------------------------------------------
    # CAIXA
    # -------------------------------------------------------------------------

    result[
        "FCO_RECEITA"
    ] = safe_divide(
        fco,
        receita,
    )

    result[
        "FCO_ATIVO"
    ] = safe_divide(
        fco,
        ativo,
    )

    # -------------------------------------------------------------------------
    # ENDIVIDAMENTO
    # -------------------------------------------------------------------------

    result[
        "DIVIDA_BRUTA_ATIVO"
    ] = safe_divide(
        divida,
        ativo,
    )

    result[
        "DIVIDA_BRUTA_PL"
    ] = safe_divide(
        divida,
        pl,
    )

    result[
        "DIVIDA_LIQUIDA"
    ] = (
        divida
        -
        caixa
    )

    result[
        "DIVIDA_LIQUIDA_ATIVO"
    ] = safe_divide(
        result[
            "DIVIDA_LIQUIDA"
        ],
        ativo,
    )

    # -------------------------------------------------------------------------
    # CAPITAL DE GIRO
    # -------------------------------------------------------------------------

    result[
        "CAPITAL_GIRO"
    ] = (
        ativo_circ
        -
        passivo_circ
    )

    result[
        "CAPITAL_GIRO_ATIVO"
    ] = safe_divide(
        result[
            "CAPITAL_GIRO"
        ],
        ativo,
    )

    # -------------------------------------------------------------------------
    # COMPOSIÇÃO DO ATIVO
    # -------------------------------------------------------------------------

    result[
        "ESTOQUES_ATIVO"
    ] = safe_divide(
        estoques,
        ativo,
    )

    result[
        "IMOBILIZADO_ATIVO"
    ] = safe_divide(
        imobilizado,
        ativo,
    )

    result[
        "INTANGIVEL_ATIVO"
    ] = safe_divide(
        intangivel,
        ativo,
    )

    # -------------------------------------------------------------------------
    # FLAGS
    #
    # Missing permanece missing.
    # Não transformamos ausência de dado em "False".
    # -------------------------------------------------------------------------

    def positive_flag(
        series: pd.Series,
    ) -> pd.Series:

        output = pd.Series(
            pd.NA,
            index=series.index,
            dtype="boolean",
        )

        valid = series.notna()

        output.loc[
            valid
        ] = (
            series.loc[
                valid
            ]
            > 0
        )

        return output

    result[
        "LUCRO_POSITIVO"
    ] = positive_flag(
        lucro
    )

    result[
        "EBIT_POSITIVO"
    ] = positive_flag(
        ebit
    )

    result[
        "FCO_POSITIVO"
    ] = positive_flag(
        fco
    )

    result[
        "CAPITAL_GIRO_POSITIVO"
    ] = positive_flag(
        result[
            "CAPITAL_GIRO"
        ]
    )

    result[
        "PL_POSITIVO"
    ] = positive_flag(
        pl
    )

    return result


# =============================================================================
# SANIDADE DOS INDICADORES
# =============================================================================

def sanitize_factors(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove apenas valores matematicamente inválidos.

    NÃO winsoriza.
    NÃO corta outliers.
    NÃO muda direção.
    NÃO preenche missing com mediana.

    Essas decisões pertencem aos motores posteriores.
    """

    result = (
        df
        .copy()
    )

    for factor in OUTPUT_FACTORS:

        if factor not in result.columns:
            continue

        result[
            factor
        ] = (
            pd.to_numeric(
                result[
                    factor
                ],
                errors="coerce",
            )
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
        )

    return result


# =============================================================================
# COBERTURA
# =============================================================================

def calculate_factor_coverage(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    total = len(
        df
    )

    for factor in (
        *OUTPUT_FACTORS,
        *OUTPUT_FLAGS,
    ):

        if factor not in df.columns:

            available = 0

        else:

            available = int(
                df[
                    factor
                ]
                .notna()
                .sum()
            )

        rows.append(
            {
                "FACTOR":
                    factor,

                "TOTAL_ISSUERS":
                    int(
                        total
                    ),

                "AVAILABLE":
                    available,

                "MISSING":
                    int(
                        total
                        -
                        available
                    ),

                "COVERAGE":
                    (
                        available / total
                        if total
                        else 0.0
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MOTOR PRINCIPAL
# =============================================================================

def build_fundamentals(
    fundamental_base: pd.DataFrame,
    context: PITContext,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Pipeline fundamental:

        base PIT
            ↓
        validação
            ↓
        indicadores
            ↓
        sanitização
            ↓
        cobertura

    Retorna:
        fundamentals
        coverage
    """

    base = (
        prepare_fundamental_base(
            fundamental_base,
            context,
        )
    )

    fundamentals = (
        calculate_factors(
            base
        )
    )

    fundamentals = (
        sanitize_factors(
            fundamentals
        )
    )

    coverage = (
        calculate_factor_coverage(
            fundamentals
        )
    )

    # -------------------------------------------------------------------------
    # METADADOS
    # -------------------------------------------------------------------------

    fundamentals[
        "FORMATION_DATE"
    ] = context.formation_date

    fundamentals[
        "ACCOUNTING_CUTOFF_ENGINE"
    ] = context.accounting_cutoff

    fundamentals[
        "FUTURE_RETURN_USED"
    ] = False

    fundamentals[
        "FUNDAMENTAL_ENGINE_OK"
    ] = True

    # -------------------------------------------------------------------------
    # FAIL-SAFE
    # -------------------------------------------------------------------------

    if (
        fundamentals[
            "ISSUER_ID"
        ]
        .duplicated()
        .any()
    ):

        raise FundamentalIntegrityError(
            "FAIL-SAFE: emissor duplicado "
            "após Fundamental Engine."
        )

    forbidden = [
        column
        for column in fundamentals.columns
        if (
            "FUTURE_RETURN"
            in str(column).upper()
            or
            "RETORNO_FUTURO"
            in str(column).upper()
        )
        and column != "FUTURE_RETURN_USED"
    ]

    if forbidden:

        raise FundamentalIntegrityError(
            "FAIL-SAFE: variável de retorno futuro "
            "detectada no Fundamental Engine: "
            f"{forbidden}"
        )

    return (
        fundamentals,
        coverage,
    )


# =============================================================================
# AUDITORIA
# =============================================================================

def audit_fundamentals(
    fundamentals: pd.DataFrame,
    coverage: pd.DataFrame,
) -> dict:

    _require_dataframe(
        fundamentals,
        "fundamentals",
    )

    _require_dataframe(
        coverage,
        "coverage",
    )

    factor_coverage = {}

    for factor in OUTPUT_FACTORS:

        row = coverage.loc[
            coverage[
                "FACTOR"
            ]
            == factor
        ]

        if row.empty:
            continue

        factor_coverage[
            factor
        ] = float(
            row.iloc[0][
                "COVERAGE"
            ]
        )

    return {
        "status":
            "OK",

        "issuers":
            int(
                len(
                    fundamentals
                )
            ),

        "factors_created":
            int(
                sum(
                    factor
                    in fundamentals.columns
                    for factor
                    in OUTPUT_FACTORS
                )
            ),

        "flags_created":
            int(
                sum(
                    flag
                    in fundamentals.columns
                    for flag
                    in OUTPUT_FLAGS
                )
            ),

        "factor_coverage":
            factor_coverage,

        "future_return_used":
            False,
    }


# =============================================================================
# SALVAR RESULTADOS
# =============================================================================

def save_fundamentals(
    fundamentals: pd.DataFrame,
    coverage: pd.DataFrame,
    context: PITContext,
) -> dict:

    audit = audit_fundamentals(
        fundamentals,
        coverage,
    )

    date_tag = (
        context
        .formation_date
        .strftime(
            "%Y%m%d"
        )
    )

    fundamentals_path = (
        FUNDAMENTAL_DIR
        / f"fundamentals_{date_tag}.csv"
    )

    coverage_path = (
        FUNDAMENTAL_DIR
        / f"fundamental_coverage_{date_tag}.csv"
    )

    manifest_path = (
        FUNDAMENTAL_DIR
        / f"fundamental_manifest_{date_tag}.json"
    )

    fundamentals.to_csv(
        fundamentals_path,
        index=False,
        encoding="utf-8-sig",
    )

    coverage.to_csv(
        coverage_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "engine":
            "FUNDAMENTAL_ENGINE",

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

        **audit,

        "files": {
            "fundamentals":
                str(
                    fundamentals_path
                ),

            "coverage":
                str(
                    coverage_path
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
        "fundamentals":
            fundamentals_path,

        "coverage":
            coverage_path,

        "manifest":
            manifest_path,
    }


# =============================================================================
# SELF-TEST
# =============================================================================

def _self_test():

    sample = pd.DataFrame(
        {
            "ISSUER_ID": [
                "1",
                "2",
            ],

            "ATIVO_TOTAL": [
                1000.0,
                2000.0,
            ],

            "ATIVO_CIRCULANTE": [
                400.0,
                700.0,
            ],

            "PASSIVO_CIRCULANTE": [
                250.0,
                800.0,
            ],

            "PL": [
                500.0,
                900.0,
            ],

            "PL_ANTERIOR": [
                450.0,
                850.0,
            ],

            "RECEITA": [
                800.0,
                1200.0,
            ],

            # Empresa 1: custo negativo estilo CVM.
            # Empresa 2: custo positivo.
            "CUSTO": [
                -500.0,
                700.0,
            ],

            "EBIT": [
                100.0,
                -20.0,
            ],

            "LUCRO_LIQUIDO": [
                70.0,
                -50.0,
            ],

            "FCO": [
                90.0,
                -10.0,
            ],

            "DIVIDA_BRUTA": [
                200.0,
                600.0,
            ],

            "CAIXA": [
                50.0,
                100.0,
            ],

            "ESTOQUES": [
                100.0,
                300.0,
            ],

            "IMOBILIZADO": [
                300.0,
                800.0,
            ],

            "INTANGIVEL": [
                20.0,
                150.0,
            ],

            "ACCOUNTING_CUTOFF": [
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

    fundamentals, coverage = (
        build_fundamentals(
            sample,
            FakeContext(),
        )
    )

    # -------------------------------------------------------------------------
    # TESTE MARGEM BRUTA EMPRESA 1
    #
    # Receita = 800
    # Custo = -500
    # Lucro bruto = 300
    # Margem = 37,5%
    # -------------------------------------------------------------------------

    company_1 = (
        fundamentals.loc[
            fundamentals[
                "ISSUER_ID"
            ]
            == "1"
        ]
        .iloc[0]
    )

    if not np.isclose(
        company_1[
            "MARGEM_BRUTA"
        ],
        0.375,
    ):

        raise FundamentalError(
            "SELF-TEST: MARGEM_BRUTA incorreta."
        )

    # -------------------------------------------------------------------------
    # TESTE ROE
    #
    # PL médio = (500 + 450) / 2 = 475
    # ROE = 70 / 475
    # -------------------------------------------------------------------------

    expected_roe = (
        70.0 / 475.0
    )

    if not np.isclose(
        company_1[
            "ROE"
        ],
        expected_roe,
    ):

        raise FundamentalError(
            "SELF-TEST: ROE com PL médio incorreto."
        )

    # -------------------------------------------------------------------------
    # TESTE CAPITAL DE GIRO
    # -------------------------------------------------------------------------

    if not np.isclose(
        company_1[
            "CAPITAL_GIRO"
        ],
        150.0,
    ):

        raise FundamentalError(
            "SELF-TEST: CAPITAL_GIRO incorreto."
        )

    # -------------------------------------------------------------------------
    # TESTE FLAGS
    # -------------------------------------------------------------------------

    if not bool(
        company_1[
            "LUCRO_POSITIVO"
        ]
    ):

        raise FundamentalError(
            "SELF-TEST: flag LUCRO_POSITIVO incorreta."
        )

    company_2 = (
        fundamentals.loc[
            fundamentals[
                "ISSUER_ID"
            ]
            == "2"
        ]
        .iloc[0]
    )

    if bool(
        company_2[
            "LUCRO_POSITIVO"
        ]
    ):

        raise FundamentalError(
            "SELF-TEST: empresa com prejuízo "
            "marcada como lucro positivo."
        )

    if coverage.empty:

        raise FundamentalError(
            "SELF-TEST: cobertura não calculada."
        )

    return True


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("B3 STOCK SELECTOR — FUNDAMENTAL ENGINE")
    print("=" * 72)

    _self_test()

    print("Self-test: OK")
    print("Unidade fundamental: ISSUER")
    print("Point-in-Time: OBRIGATÓRIO")
    print("ROE: PL MÉDIO")
    print("Margem bruta: CALCULADA")
    print("Margem líquida: CALCULADA")
    print("Geração de caixa: CALCULADA")
    print("Endividamento: CALCULADO")
    print("Capital de giro: CALCULADO")
    print("Missing data: PRESERVADO")
    print("Outliers: NÃO OTIMIZADOS")
    print("Direção dos fatores: NÃO DEFINIDA")
    print("Retorno futuro: NÃO UTILIZADO")
    print("Ranking: NÃO EXECUTADO")

    print("=" * 72)
