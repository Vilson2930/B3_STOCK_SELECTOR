# =============================================================================
# B3 STOCK SELECTOR
# Arquivo: engines/turnaround.py
#
# TURNAROUND / ASYMMETRY ENGINE
#
# OBJETIVO:
# Identificar empresas em uma condição econômica compatível com possível
# recuperação operacional / assimetria, SEM transformar baixa rentabilidade
# isolada em recomendação de compra.
#
# EVIDÊNCIA DO ESTUDO:
#
# MARGEM_BRUTA:
# - principal sinal candidato;
# - direção observada: LOW;
# - AUC 2018 ~ 0.7217;
# - gradiente por quintis:
#       Q1 = 6.42% winners
#       Q2 = 4.59%
#       Q3 = 3.67%
#       Q4 = 0.92%
#       Q5 = 0.00%
# - Spearman quintis = -1.0;
# - 15/17 winners estavam na metade inferior da distribuição.
#
# MARGEM_LIQUIDA:
# - sinal secundário recorrente;
# - direção LOW.
#
# ROE:
# - sinal recorrente mais fraco;
# - direção LOW.
#
# IMPORTANTE:
# A combinação desses sinais NÃO foi validada como regra final.
#
# O full model 2022 NÃO replicou robustamente.
# O multivariado 2018 NÃO sobreviveu à permutação global.
#
# CONSEQUÊNCIA:
# Este engine NÃO está autorizado a gerar compra.
#
# Ele apenas:
# - mede posição relativa;
# - identifica zona de margem deprimida;
# - mede contexto econômico;
# - registra possíveis sinais;
# - exige proteção financeira;
# - prepara dados para validação futura.
#
# PRODUÇÃO:
# TURNAROUND_ENGINE["production_authorized"] deve permanecer False
# até existir validação temporal independente suficiente.
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
    from config import (
        OUTPUT_DIR,
        TURNAROUND_ENGINE,
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
# DIRETÓRIO
# =============================================================================

TURNAROUND_DIR = (
    OUTPUT_DIR
    / "turnaround"
)

TURNAROUND_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# EXCEÇÕES
# =============================================================================

class TurnaroundError(RuntimeError):
    """Erro geral do Turnaround Engine."""


class TurnaroundIntegrityError(TurnaroundError):
    """Erro de integridade."""


class TurnaroundProductionError(TurnaroundError):
    """Tentativa de usar sinal experimental como produção."""


# =============================================================================
# FATORES CANDIDATOS
#
# Estes fatores são derivados do estudo científico.
#
# NÃO representam regra final.
# =============================================================================

RESEARCH_FACTORS = {

    "MARGEM_BRUTA": {
        "direction": "LOW",
        "role": "PRIMARY",
        "status": "CANDIDATE",
    },

    "MARGEM_LIQUIDA": {
        "direction": "LOW",
        "role": "SECONDARY",
        "status": "RECURRENT_SIGNAL",
    },

    "ROE": {
        "direction": "LOW",
        "role": "CONTEXT",
        "status": "WEAK_RECURRENT_SIGNAL",
    },
}


# =============================================================================
# FATORES QUE NÃO PODEM SER REINTRODUZIDOS COMO SINAL ESTRUTURAL
# =============================================================================

BLOCKED_RESEARCH_FACTORS = (
    "ESTOQUES",
    "DIVIDA_BRUTA_ATIVO_AS_OPPORTUNITY",
    "INTANGIVEL_ATIVO_AS_OPPORTUNITY",
)


# =============================================================================
# PROTEÇÕES
#
# Não são sinais de oportunidade.
#
# Servem para reduzir o risco de interpretar empresa estruturalmente
# deteriorada como turnaround.
#
# Neste estágio utilizamos principalmente flags econômicos objetivos.
# =============================================================================

SAFETY_FLAGS = (
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

        raise TurnaroundIntegrityError(
            f"{name} não é DataFrame."
        )

    if df.empty:

        raise TurnaroundIntegrityError(
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

        raise TurnaroundIntegrityError(
            f"FAIL-SAFE: {name} sem colunas obrigatórias: "
            f"{sorted(missing)}"
        )


# =============================================================================
# PROTEÇÃO CONTRA RETORNO FUTURO
# =============================================================================

def assert_no_future_return(
    df: pd.DataFrame,
) -> None:

    forbidden_terms = (
        "FUTURE_RETURN",
        "RETORNO_FUTURO",
        "RETURN_FUTURE",
        "WINNER_LABEL",
        "FUTURE_WINNER",
    )

    forbidden = []

    for column in df.columns:

        normalized = (
            str(column)
            .upper()
        )

        if normalized in {
            "FUTURE_RETURN_USED",
            "FUTURE_RETURN_USED_QUALITY",
        }:
            continue

        if any(
            term in normalized
            for term in forbidden_terms
        ):

            forbidden.append(
                column
            )

    if forbidden:

        raise TurnaroundIntegrityError(
            "FAIL-SAFE: retorno futuro ou label "
            "detectado no Turnaround Engine: "
            f"{forbidden}"
        )


# =============================================================================
# BASE
# =============================================================================

def prepare_turnaround_base(
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        "ISSUER_ID",
        "FUNDAMENTAL_ENGINE_OK",
        "MARGEM_BRUTA",
        "MARGEM_LIQUIDA",
        "ROE",
    }

    _require_columns(
        fundamentals,
        required,
        "fundamentals",
    )

    assert_no_future_return(
        fundamentals
    )

    result = (
        fundamentals
        .copy()
    )

    # -------------------------------------------------------------------------
    # SOMENTE EMPRESAS QUE PASSARAM PELO INVESTABILITY GATE
    # -------------------------------------------------------------------------

    if "INVESTABLE" in result.columns:

        result = result.loc[
            result[
                "INVESTABLE"
            ]
            .fillna(False)
            .astype(bool)
        ].copy()

    # -------------------------------------------------------------------------
    # FUNDAMENTAL ENGINE DEVE ESTAR VÁLIDO
    # -------------------------------------------------------------------------

    result = result.loc[
        result[
            "FUNDAMENTAL_ENGINE_OK"
        ]
        .fillna(False)
        .astype(bool)
    ].copy()

    if result.empty:

        raise TurnaroundIntegrityError(
            "FAIL-SAFE: nenhuma empresa elegível "
            "para Turnaround."
        )

    if (
        result[
            "ISSUER_ID"
        ]
        .duplicated()
        .any()
    ):

        raise TurnaroundIntegrityError(
            "FAIL-SAFE: ISSUER_ID duplicado."
        )

    return result


# =============================================================================
# PERCENTIL
# =============================================================================

def percentile_position(
    series: pd.Series,
) -> pd.Series:
    """
    Retorna posição percentual crescente.

    Quanto menor o valor econômico,
    menor o percentil.

    Exemplo:
        margem bruta muito baixa -> percentil próximo de 0.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    valid = (
        numeric.notna()
    )

    if valid.sum() == 0:
        return result

    result.loc[
        valid
    ] = (
        numeric.loc[
            valid
        ]
        .rank(
            pct=True,
            method="average",
        )
    )

    return result.clip(
        0.0,
        1.0,
    )


# =============================================================================
# POSIÇÃO DOS FATORES
# =============================================================================

def calculate_research_positions(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        df
        .copy()
    )

    for factor in RESEARCH_FACTORS:

        result[
            f"{factor}_PERCENTILE"
        ] = percentile_position(
            result[
                factor
            ]
        )

    return result


# =============================================================================
# MARGEM BRUTA — SINAL PRINCIPAL CANDIDATO
# =============================================================================

def classify_gross_margin_zone(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classifica a empresa pela posição relativa da margem bruta.

    IMPORTANTE:
    São zonas descritivas.

    NÃO são thresholds de compra.

    Os cortes representam divisões naturais da distribuição:
        <=20%
        20-40%
        40-60%
        60-80%
        >80%

    Isso preserva a lógica do estudo por quintis sem afirmar
    que o primeiro quintil seja uma regra final.
    """

    result = (
        df
        .copy()
    )

    percentile = pd.to_numeric(
        result[
            "MARGEM_BRUTA_PERCENTILE"
        ],
        errors="coerce",
    )

    zone = pd.Series(
        "UNKNOWN",
        index=result.index,
        dtype="object",
    )

    zone.loc[
        percentile.le(0.20)
    ] = "VERY_LOW"

    zone.loc[
        percentile.gt(0.20)
        &
        percentile.le(0.40)
    ] = "LOW"

    zone.loc[
        percentile.gt(0.40)
        &
        percentile.le(0.60)
    ] = "MID"

    zone.loc[
        percentile.gt(0.60)
        &
        percentile.le(0.80)
    ] = "HIGH"

    zone.loc[
        percentile.gt(0.80)
    ] = "VERY_HIGH"

    result[
        "GROSS_MARGIN_ZONE"
    ] = zone

    return result


# =============================================================================
# INTENSIDADE DO SINAL DE MARGEM BRUTA
# =============================================================================

def calculate_gross_margin_research_signal(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transforma posição de margem bruta em intensidade [0,1].

    Quanto menor a margem bruta relativa,
    maior o sinal de PESQUISA.

    Isso NÃO é probabilidade calibrada.
    Isso NÃO é score de compra.
    """

    result = (
        df
        .copy()
    )

    percentile = pd.to_numeric(
        result[
            "MARGEM_BRUTA_PERCENTILE"
        ],
        errors="coerce",
    )

    result[
        "GROSS_MARGIN_RESEARCH_SIGNAL"
    ] = (
        1.0
        -
        percentile
    )

    result[
        "GROSS_MARGIN_RESEARCH_SIGNAL"
    ] = (
        result[
            "GROSS_MARGIN_RESEARCH_SIGNAL"
        ]
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    return result


# =============================================================================
# SINAIS SECUNDÁRIOS
# =============================================================================

def calculate_secondary_context(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        df
        .copy()
    )

    for factor in (
        "MARGEM_LIQUIDA",
        "ROE",
    ):

        percentile_column = (
            f"{factor}_PERCENTILE"
        )

        signal_column = (
            f"{factor}_DEPRESSED_SIGNAL"
        )

        result[
            signal_column
        ] = (
            1.0
            -
            pd.to_numeric(
                result[
                    percentile_column
                ],
                errors="coerce",
            )
        ).clip(
            lower=0.0,
            upper=1.0,
        )

    return result


# =============================================================================
# PROTEÇÃO CONTRA VALUE TRAP
# =============================================================================

def calculate_safety_context(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Camada conservadora.

    Não atribui pontos de oportunidade.
    Apenas registra se existem condições mínimas observáveis.

    Neste estágio:
        patrimônio líquido positivo.

    Investability já deve ter sido aplicado anteriormente.
    """

    result = (
        df
        .copy()
    )

    if "PL_POSITIVO" in result.columns:

        pl_positive = (
            result[
                "PL_POSITIVO"
            ]
            .astype("boolean")
        )

    else:

        pl_positive = pd.Series(
            pd.NA,
            index=result.index,
            dtype="boolean",
        )

    result[
        "TURNAROUND_PL_POSITIVE"
    ] = pl_positive

    # -------------------------------------------------------------------------
    # CAIXA OPERACIONAL
    #
    # Não exigimos FCO positivo como regra definitiva,
    # porque empresas em recuperação podem ainda apresentar FCO negativo.
    # Apenas registramos o contexto.
    # -------------------------------------------------------------------------

    if "FCO_POSITIVO" in result.columns:

        result[
            "TURNAROUND_FCO_POSITIVE"
        ] = (
            result[
                "FCO_POSITIVO"
            ]
            .astype(
                "boolean"
            )
        )

    else:

        result[
            "TURNAROUND_FCO_POSITIVE"
        ] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="boolean",
        )

    # -------------------------------------------------------------------------
    # EBIT
    # -------------------------------------------------------------------------

    if "EBIT_POSITIVO" in result.columns:

        result[
            "TURNAROUND_EBIT_POSITIVE"
        ] = (
            result[
                "EBIT_POSITIVO"
            ]
            .astype(
                "boolean"
            )
        )

    else:

        result[
            "TURNAROUND_EBIT_POSITIVE"
        ] = pd.Series(
            pd.NA,
            index=result.index,
            dtype="boolean",
        )

    return result


# =============================================================================
# CANDIDATO DE PESQUISA
# =============================================================================

def classify_research_candidate(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classificação exploratória.

    CANDIDATE:
        margem bruta na metade inferior da distribuição.

    HIGH_INTEREST:
        margem bruta no quintil inferior.

    Isso serve para auditoria e pesquisa.

    NÃO significa:
        BUY
        ENTRY
        STRONG_ENTRY
        recomendação
    """

    result = (
        df
        .copy()
    )

    percentile = pd.to_numeric(
        result[
            "MARGEM_BRUTA_PERCENTILE"
        ],
        errors="coerce",
    )

    status = pd.Series(
        "NO_SIGNAL",
        index=result.index,
        dtype="object",
    )

    status.loc[
        percentile.le(0.50)
    ] = "RESEARCH_CANDIDATE"

    status.loc[
        percentile.le(0.20)
    ] = "HIGH_RESEARCH_INTEREST"

    status.loc[
        percentile.isna()
    ] = "INSUFFICIENT_DATA"

    result[
        "TURNAROUND_RESEARCH_STATUS"
    ] = status

    return result


# =============================================================================
# SCORE EXPERIMENTAL
# =============================================================================

def calculate_experimental_score(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score exclusivamente descritivo.

    IMPORTANTE:
    Não combina os três fatores em um modelo preditivo.

    Isso é intencional.

    O estudo mostrou que a combinação dos sinais não teve significância
    global suficiente para virar regra.

    Portanto o score experimental principal permanece baseado apenas
    na posição relativa da MARGEM_BRUTA.

    Margem líquida e ROE ficam como contexto separado.
    """

    result = (
        df
        .copy()
    )

    result[
        "TURNAROUND_RESEARCH_SCORE"
    ] = (
        result[
            "GROSS_MARGIN_RESEARCH_SIGNAL"
        ]
    )

    result[
        "TURNAROUND_RESEARCH_SCORE_VALID"
    ] = (
        result[
            "TURNAROUND_RESEARCH_SCORE"
        ]
        .notna()
    )

    return result


# =============================================================================
# PRODUÇÃO
# =============================================================================

def enforce_production_lock() -> None:
    """
    Impede que o motor seja promovido silenciosamente para produção.
    """

    production_authorized = bool(
        TURNAROUND_ENGINE.get(
            "production_authorized",
            False,
        )
    )

    validated_rule = (
        TURNAROUND_ENGINE.get(
            "validated_rule"
        )
    )

    if (
        production_authorized
        and validated_rule is None
    ):

        raise TurnaroundProductionError(
            "FAIL-SAFE: Turnaround marcado como produção "
            "sem regra temporalmente validada."
        )


# =============================================================================
# MOTOR PRINCIPAL
# =============================================================================

def run_turnaround_engine(
    fundamentals: pd.DataFrame,
    context: PITContext,
) -> pd.DataFrame:

    if not TURNAROUND_ENGINE.get(
        "enabled",
        True,
    ):

        raise TurnaroundError(
            "Turnaround Engine está desativado."
        )

    enforce_production_lock()

    base = prepare_turnaround_base(
        fundamentals
    )

    result = calculate_research_positions(
        base
    )

    result = classify_gross_margin_zone(
        result
    )

    result = (
        calculate_gross_margin_research_signal(
            result
        )
    )

    result = calculate_secondary_context(
        result
    )

    result = calculate_safety_context(
        result
    )

    result = classify_research_candidate(
        result
    )

    result = calculate_experimental_score(
        result
    )

    # -------------------------------------------------------------------------
    # STATUS DE PRODUÇÃO
    # -------------------------------------------------------------------------

    production_authorized = bool(
        TURNAROUND_ENGINE.get(
            "production_authorized",
            False,
        )
    )

    result[
        "TURNAROUND_PRODUCTION_AUTHORIZED"
    ] = production_authorized

    # -------------------------------------------------------------------------
    # DECISÃO OPERACIONAL
    #
    # Enquanto não houver validação suficiente:
    # NENHUMA empresa recebe sinal operacional.
    # -------------------------------------------------------------------------

    if not production_authorized:

        result[
            "TURNAROUND_OPERATIONAL_SIGNAL"
        ] = "BLOCKED_RESEARCH_ONLY"

    else:

        validated_rule = (
            TURNAROUND_ENGINE.get(
                "validated_rule"
            )
        )

        if validated_rule is None:

            raise TurnaroundProductionError(
                "FAIL-SAFE: regra validada ausente."
            )

        # A execução de uma futura regra de produção deve ser
        # implementada explicitamente quando a validação científica
        # estiver concluída.
        raise TurnaroundProductionError(
            "FAIL-SAFE: execução de produção ainda "
            "não implementada. Pesquisa preservada."
        )

    # -------------------------------------------------------------------------
    # METADADOS
    # -------------------------------------------------------------------------

    result[
        "TURNAROUND_ENGINE_VERSION"
    ] = "0.1.0"

    result[
        "TURNAROUND_ENGINE_STATUS"
    ] = TURNAROUND_ENGINE.get(
        "status",
        "RESEARCH_CANDIDATE",
    )

    result[
        "FORMATION_DATE_TURNAROUND"
    ] = context.formation_date

    result[
        "ACCOUNTING_CUTOFF_TURNAROUND"
    ] = context.accounting_cutoff

    result[
        "FUTURE_RETURN_USED_TURNAROUND"
    ] = False

    # -------------------------------------------------------------------------
    # RANK DE PESQUISA
    #
    # Não é ranking de compra.
    # -------------------------------------------------------------------------

    result[
        "TURNAROUND_RESEARCH_RANK"
    ] = (
        result[
            "TURNAROUND_RESEARCH_SCORE"
        ]
        .rank(
            ascending=False,
            method="min",
            na_option="bottom",
        )
    )

    result = (
        result
        .sort_values(
            [
                "TURNAROUND_RESEARCH_SCORE",
                "ISSUER_ID",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    assert_no_future_return(
        result
    )

    return result


# =============================================================================
# AUDITORIA
# =============================================================================

def audit_turnaround(
    result: pd.DataFrame,
) -> dict:

    _require_dataframe(
        result,
        "turnaround_result",
    )

    status_counts = (
        result[
            "TURNAROUND_RESEARCH_STATUS"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    zone_counts = (
        result[
            "GROSS_MARGIN_ZONE"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    return {
        "status":
            "OK",

        "mode":
            "RESEARCH_ONLY",

        "issuers":
            int(
                len(
                    result
                )
            ),

        "primary_factor":
            "MARGEM_BRUTA",

        "primary_direction":
            "LOW",

        "research_status_distribution":
            status_counts,

        "gross_margin_zone_distribution":
            zone_counts,

        "production_authorized":
            bool(
                TURNAROUND_ENGINE.get(
                    "production_authorized",
                    False,
                )
            ),

        "future_return_used":
            False,

        "buy_signal_generated":
            False,
    }


# =============================================================================
# SALVAR
# =============================================================================

def save_turnaround(
    result: pd.DataFrame,
    context: PITContext,
) -> dict:

    audit = audit_turnaround(
        result
    )

    date_tag = (
        context
        .formation_date
        .strftime(
            "%Y%m%d"
        )
    )

    result_path = (
        TURNAROUND_DIR
        / f"turnaround_research_{date_tag}.csv"
    )

    manifest_path = (
        TURNAROUND_DIR
        / f"turnaround_manifest_{date_tag}.json"
    )

    result.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "engine":
            "TURNAROUND_ENGINE",

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

        "research_factors":
            RESEARCH_FACTORS,

        "blocked_research_factors":
            list(
                BLOCKED_RESEARCH_FACTORS
            ),

        **audit,

        "files": {
            "turnaround_research":
                str(
                    result_path
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
        "turnaround":
            result_path,

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
                "3",
                "4",
                "5",
            ],

            "FUNDAMENTAL_ENGINE_OK": [
                True,
                True,
                True,
                True,
                True,
            ],

            "INVESTABLE": [
                True,
                True,
                True,
                True,
                True,
            ],

            "MARGEM_BRUTA": [
                0.05,
                0.15,
                0.30,
                0.50,
                0.80,
            ],

            "MARGEM_LIQUIDA": [
                -0.10,
                -0.02,
                0.05,
                0.10,
                0.20,
            ],

            "ROE": [
                -0.10,
                0.01,
                0.08,
                0.15,
                0.25,
            ],

            "PL_POSITIVO": [
                True,
                True,
                True,
                True,
                True,
            ],

            "FCO_POSITIVO": [
                False,
                True,
                True,
                True,
                True,
            ],

            "EBIT_POSITIVO": [
                False,
                True,
                True,
                True,
                True,
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

    result = run_turnaround_engine(
        sample,
        FakeContext(),
    )

    indexed = (
        result
        .set_index(
            "ISSUER_ID"
        )
    )

    # Empresa com menor margem deve ter maior sinal de pesquisa.
    if not (
        indexed.loc[
            "1",
            "TURNAROUND_RESEARCH_SCORE",
        ]
        >
        indexed.loc[
            "5",
            "TURNAROUND_RESEARCH_SCORE",
        ]
    ):

        raise TurnaroundError(
            "SELF-TEST: direção da margem bruta incorreta."
        )

    # Nenhuma empresa pode receber sinal de compra.
    if not (
        result[
            "TURNAROUND_OPERATIONAL_SIGNAL"
        ]
        ==
        "BLOCKED_RESEARCH_ONLY"
    ).all():

        raise TurnaroundError(
            "SELF-TEST: sinal operacional foi "
            "liberado indevidamente."
        )

    # Retorno futuro jamais pode ser usado.
    if (
        result[
            "FUTURE_RETURN_USED_TURNAROUND"
        ]
        .any()
    ):

        raise TurnaroundError(
            "SELF-TEST: retorno futuro utilizado."
        )

    return True


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("B3 STOCK SELECTOR — TURNAROUND ENGINE")
    print("=" * 72)

    _self_test()

    print("Self-test: OK")
    print("Modo: RESEARCH ONLY")
    print("Principal candidato: MARGEM_BRUTA")
    print("Direção observada: LOW")
    print("Margem líquida: CONTEXTO SECUNDÁRIO")
    print("ROE: CONTEXTO SECUNDÁRIO")
    print("Baixa margem = compra: NÃO")
    print("Combinação 3 fatores validada: NÃO")
    print("Full rule 2022 utilizada: NÃO")
    print("Multivariado 2018 utilizado: NÃO")
    print("Retorno futuro: NÃO UTILIZADO")
    print("Sinal operacional: BLOQUEADO")
    print("Validação temporal adicional: NECESSÁRIA")

    print("=" * 72)
