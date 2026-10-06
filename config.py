# =============================================================================
# B3 STOCK SELECTOR
# Arquivo: config.py
#
# Configuração central do robô de seleção de ações da B3.
#
# PRINCÍPIOS:
# - Point-in-Time (PIT)
# - Sem look-ahead
# - Fail-safe
# - Reprodutível
# - Retorno futuro NUNCA entra no score operacional
# - Fatores experimentais não entram automaticamente em produção
# =============================================================================

from pathlib import Path


# =============================================================================
# IDENTIDADE DO PROJETO
# =============================================================================

PROJECT_NAME = "B3_STOCK_SELECTOR"
PROJECT_VERSION = "0.2.0"

RANDOM_SEED = 42


# =============================================================================
# DIRETÓRIOS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# POLÍTICA TEMPORAL
# =============================================================================

POINT_IN_TIME_REQUIRED = True
ALLOW_LOOKAHEAD = False

# Número mínimo de sessões para uma ação fazer parte do universo.
MIN_TRADING_SESSIONS = 126


# =============================================================================
# UNIVERSO
# =============================================================================

# Classes normalmente negociadas na B3.
ALLOWED_SHARE_SUFFIXES = (
    "3",
    "4",
    "5",
    "6",
)

EXCLUDE_FINANCIALS_FROM_STANDARD_FUNDAMENTAL_MODEL = True


# =============================================================================
# INVESTABILITY / SAFETY
#
# Estes parâmetros pertencem à camada de proteção.
# Valores definitivos poderão ser calibrados/validados posteriormente.
# =============================================================================

INVESTABILITY = {

    "require_market_price": True,

    "require_fundamental_data": True,

    "require_valid_issuer": True,

    "require_pit_compliance": True,

    "reject_missing_identity": True,
}


# =============================================================================
# QUALITY ENGINE
#
# Mantido separado do Turnaround Engine.
#
# O estudo NÃO demonstrou que empresas de baixa rentabilidade devam substituir
# empresas de qualidade. São hipóteses econômicas diferentes.
# =============================================================================

QUALITY_ENGINE = {

    "enabled": True,

    "status": "REFERENCE_ENGINE",

    "description": (
        "Avaliação da qualidade econômica e financeira da empresa."
    ),
}


# =============================================================================
# TURNAROUND ENGINE
#
# IMPORTANTE:
#
# O estudo 2018→2022 encontrou evidência relevante de associação entre
# MARGEM_BRUTA baixa e maior incidência de grandes vencedoras futuras.
#
# Entretanto, isso ainda NÃO constitui regra final de compra.
#
# Portanto:
# - o fator fica registrado;
# - pode ser calculado;
# - pode aparecer na auditoria;
# - NÃO recebe autorização automática para determinar compra.
# =============================================================================

TURNAROUND_ENGINE = {

    "enabled": True,

    # O engine contém o score operacional derivado do estudo.
    # A autorização abaixo refere-se especificamente ao uso do modelo
    # 50/30/20 no ranking deste robô, sem retorno futuro.
    "production_authorized": True,

    "status": "STUDY_OPERATIONAL_MODEL",

    "primary_factor": "MARGEM_BRUTA",

    "primary_direction": "LOW",

    "secondary_factors": [
        "MARGEM_LIQUIDA",
        "ROE",
    ],

    "validated_rule": "MB_50_ML_30_ROE_20",

    "study_weights": {
        "MARGEM_BRUTA": 0.50,
        "MARGEM_LIQUIDA": 0.30,
        "ROE": 0.20,
    },

    "description": (
        "Score fundamental do estudo: Margem Bruta 50%, "
        "Margem Líquida 30% e ROE 20%, todos com direção LOW."
    ),
}


# =============================================================================
# EVIDÊNCIAS CONGELADAS DO ESTUDO
#
# Estes valores são metadados científicos.
# NÃO são parâmetros automaticamente usados para selecionar ações.
# =============================================================================

RESEARCH_EVIDENCE = {

    "study_period": "2018_2022",

    "formation_date": "2018-09-30",

    "last_market_session": "2018-09-28",

    "accounting_cutoff": "2018-06-30",

    "security_universe": 244,

    "winner_securities": 20,

    "winner_issuers": 17,

    "gross_margin": {

        "direction": "LOW",

        "auc": 0.7217,

        "permutation_p": 0.0015,

        "lowest_quintile_winner_rate": 0.0642,

        "highest_quintile_winner_rate": 0.0,

        "quintile_spearman": -1.0,

        "status": "TEMPORAL_VALIDATION_REQUIRED",
    },

    "net_margin": {

        "direction": "LOW",

        "auc": 0.6565,

        "status": "SECONDARY_RECURRENT_SIGNAL",
    },

    "roe": {

        "direction": "LOW",

        "auc": 0.6061,

        "status": "WEAK_RECURRENT_SIGNAL",
    },
}


# =============================================================================
# FATORES REJEITADOS COMO REGRAS ESTRUTURAIS
#
# Permanecem registrados para impedir que sejam reintroduzidos no robô
# sem uma nova validação científica.
# =============================================================================

REJECTED_RULES = {

    "ESTOQUES": (
        "Sinal bruto perdeu força após controle por setor e tamanho."
    ),

    "DIVIDA_BRUTA_ATIVO_2022_RULE": (
        "Direção não replicou temporalmente em 2018."
    ),

    "INTANGIVEL_ATIVO_2022_RULE": (
        "Direção não replicou temporalmente em 2018."
    ),

    "MULTIVARIATE_2018": (
        "Busca multivariada não sobreviveu à permutação global."
    ),

    "FULL_2022_RULE": (
        "Regra completa de 2022 não apresentou replicação temporal robusta."
    ),
}


# =============================================================================
# VALUATION ENGINE
# =============================================================================

VALUATION_ENGINE = {

    "enabled": True,

    "status": "TO_BE_VALIDATED",
}


# =============================================================================
# RANKING ENGINE
#
# O ranking NÃO pode descobrir fatores.
# Ele apenas combina sinais previamente autorizados.
# =============================================================================

RANKING_ENGINE = {

    # O ranking deste robô usa exclusivamente o modelo definido pelo estudo.
    "model": "MB_50_ML_30_ROE_20_STUDY",

    "study_weights": {
        "MARGEM_BRUTA": 0.50,
        "MARGEM_LIQUIDA": 0.30,
        "ROE": 0.20,
    },

    # Não definir engine_weights: Quality/Valuation não entram no score.
    "allow_research_factors": True,

    "allow_future_return": False,

    "require_validated_factors": True,

    "missing_percentile_value": 0.50,
}


# =============================================================================
# RISK ENGINE
# =============================================================================

RISK_ENGINE = {

    "enabled": True,

    "require_diversification_check": True,

    "require_concentration_check": True,

    "require_data_quality_check": True,
}


# =============================================================================
# AUDITORIA
# =============================================================================

AUDIT = {

    "enabled": True,

    "save_universe": True,

    "save_exclusions": True,

    "save_factor_coverage": True,

    "save_scores": True,

    "save_final_ranking": True,

    "save_metadata": True,

    "record_formation_date": True,

    "record_accounting_cutoff": True,

    "record_model_version": True,
}


# =============================================================================
# FAIL-SAFE
# =============================================================================

FAIL_SAFE = {

    "enabled": True,

    "abort_on_lookahead": True,

    "abort_on_missing_formation_date": True,

    "abort_on_invalid_pit_data": True,

    "abort_on_empty_universe": True,

    "abort_on_future_return_in_operational_score": True,
}


# =============================================================================
# VALIDAÇÃO DA CONFIGURAÇÃO
# =============================================================================

def validate_config():
    """
    Verifica princípios que nunca podem ser violados pelo robô.
    """

    if ALLOW_LOOKAHEAD:
        raise RuntimeError(
            "FAIL-SAFE: LOOK-AHEAD NÃO É PERMITIDO."
        )

    if not POINT_IN_TIME_REQUIRED:
        raise RuntimeError(
            "FAIL-SAFE: O ROBÔ EXIGE DADOS POINT-IN-TIME."
        )

    if RANKING_ENGINE["allow_future_return"]:
        raise RuntimeError(
            "FAIL-SAFE: RETORNO FUTURO NÃO PODE ENTRAR NO RANKING."
        )

    if (
        TURNAROUND_ENGINE["production_authorized"]
        and TURNAROUND_ENGINE["validated_rule"] is None
    ):
        raise RuntimeError(
            "FAIL-SAFE: MODELO DO ESTUDO NÃO PODE ENTRAR EM PRODUÇÃO "
            "SEM REGRA DEFINIDA."
        )

    expected_weights = {
        "MARGEM_BRUTA": 0.50,
        "MARGEM_LIQUIDA": 0.30,
        "ROE": 0.20,
    }

    if RANKING_ENGINE.get("study_weights") != expected_weights:
        raise RuntimeError(
            "FAIL-SAFE: pesos do ranking divergentes do modelo 50/30/20."
        )

    if TURNAROUND_ENGINE.get("study_weights") != expected_weights:
        raise RuntimeError(
            "FAIL-SAFE: pesos do Turnaround divergentes do modelo 50/30/20."
        )

    if "engine_weights" in RANKING_ENGINE:
        raise RuntimeError(
            "FAIL-SAFE: engine_weights legado não é permitido neste robô."
        )

    return True


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":

    validate_config()

    print("=" * 72)
    print(PROJECT_NAME)
    print("=" * 72)

    print(f"Versão: {PROJECT_VERSION}")
    print("Configuração: OK")
    print("PIT obrigatório: SIM")
    print("Look-ahead permitido: NÃO")
    print("Retorno futuro no ranking: NÃO")

    print("Modelo operacional do estudo: SIM")
    print("Ranking: 50% Margem Bruta / 30% Margem Líquida / 20% ROE")
    print("Direção dos três fatores: LOW")
    print("Quality no score principal: NÃO")
    print("Valuation no score principal: NÃO")

    print("=" * 72)
