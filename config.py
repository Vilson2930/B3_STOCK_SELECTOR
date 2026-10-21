# -*- coding: utf-8 -*-
"""Configuração central do B3 Stock Selector — metodologia das 34 etapas.

Esta configuração NÃO implementa uma estratégia de investimento por si só.
Os motores antigos de MB_100_STUDY precisam ser substituídos antes da produção.
Não há pesos de ranking, preços justos ou aprovações presumidos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_NAME = "B3_STOCK_SELECTOR"
PROJECT_VERSION = "0.3.0-34-STAGES-TRANSITION"
VERSION = PROJECT_VERSION
RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Princípios técnicos inegociáveis.
POINT_IN_TIME_REQUIRED = True
ALLOW_LOOKAHEAD = False
ALLOW_FUTURE_RETURN = False
REQUIRE_OFFICIAL_SOURCE_PROVENANCE = True
REQUIRE_ACTUAL_PUBLICATION_DATE = True
REJECT_UNKNOWN_AVAILABILITY_DATE = True
REQUIRE_ISSUER_CONTINUITY = True
REQUIRE_INDEPENDENT_ACCOUNTING_RECONCILIATION = True

# Compatibilidade temporária com leitores antigos. NÃO é o filtro final de
# liquidez/presença das 34 etapas; sua aplicação deve ser auditada e revista.
MIN_TRADING_SESSIONS = 126
ALLOWED_SHARE_SUFFIXES = ("3", "4", "5", "6")

# Nunca inferir setor desconhecido como setor não financeiro. Não há exclusão
# automática de bancos/seguradoras sem regra expressa da metodologia.
EXCLUDE_FINANCIALS_FROM_STANDARD_FUNDAMENTAL_MODEL = False

INVESTABILITY = {
    "require_market_price": True,
    "require_fundamental_data": True,
    "require_valid_issuer": True,
    "require_pit_compliance": True,
    "reject_missing_identity": True,
}

# A estrutura técnica pode ser reaproveitada, mas seus scores legados não
# conferem aprovação de investimento.
QUALITY_ENGINE = {
    "enabled": False,
    "status": "LEGACY_SCORE_DISABLED_PENDING_REPLACEMENT",
    "description": "Indicadores financeiros serão avaliados pelos gates das 34 etapas.",
}

# Desativação explícita da hipótese de margem bruta baixa do robô anterior.
# Chaves legadas são mantidas somente para identificação de dependências.
TURNAROUND_ENGINE = {
    "enabled": False,
    "production_authorized": False,
    "status": "RETIRED_LEGACY_MODEL",
    "primary_factor": None,
    "primary_direction": None,
    "secondary_factors": [],
    "validated_rule": None,
    "study_weights": {},
    "description": "Modelo antigo excluído da seleção; não executar.",
}

# Nenhuma evidência do estudo antigo MB_100 é usada na seleção atual.
RESEARCH_EVIDENCE = {}
REJECTED_RULES = {
    "MB_100_STUDY": "Modelo do robô antigo; proibido na seleção das 34 etapas.",
    "MB_50_ML_30_ROE_20_STUDY": "Modelo antigo do e-mail; proibido.",
}

# Etapas 32 a 34: lacunas documentais impedem aprovação e preço justo.
# Não confundir ausência de evidência com reprovação econômica.
STUDY_34 = {
    "source": "ETAPAS_01_A_34",
    "status": "EVIDENCE_COMPLETION_PENDING",
    "approved_issuers_from_validated_study": 0,
    "unresolved_gates": (
        "CAPEX_VERIFIED",
        "FREE_CASH_FLOW_VERIFIED",
        "NET_DEBT_VERIFIED",
        "ROIC_WACC_VERIFIED",
        "FAIR_VALUE_VERIFIED",
        "HISTORICAL_LIQUIDITY_VERIFIED",
    ),
    "allowed_decisions": ("APROVADA", "REPROVADA", "PENDENTE"),
    "missing_evidence_decision": "PENDENTE",
    "allow_unverified_assumptions": False,
    "allow_automatic_buy_recommendation": False,
    "criteria_matrix_validated": False,
}

# Limiares numéricos e pesos NÃO são inventados aqui. A matriz de critérios
# precisa vincular cada regra à etapa/documento que a fundamenta.
SELECTION_CRITERIA = {
    "source": "STUDY_34",
    "require_evidence_per_gate": True,
    "require_source_document": True,
    "require_period": True,
    "require_available_at": True,
    "require_calculation_trace": True,
    "numeric_thresholds": {},
    "ranking_weights": {},
    "methodology_ready": False,
}

VALUATION_ENGINE = {
    "enabled": False,
    "status": "BLOCKED_UNTIL_FCF_DEBT_CAPITAL_COST_AND_SHARE_COUNT_VALIDATED",
    "require_verified_fair_value": True,
    "require_verified_margin_of_safety": True,
}

RANKING_ENGINE = {
    "enabled": False,
    "model": "STUDY_34_GATES_PENDING_VALIDATION",
    "study_weights": {},
    "allow_research_factors": False,
    "allow_future_return": False,
    "require_validated_factors": True,
    "missing_percentile_value": None,
    "require_all_mandatory_gates": True,
    "allow_ranking_without_evidence": False,
}

RISK_ENGINE = {
    "enabled": False,
    "require_diversification_check": True,
    "require_concentration_check": True,
    "require_data_quality_check": True,
    "require_approved_issuer_before_portfolio": True,
}

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
    "record_source_document": True,
    "record_publication_date": True,
    "record_gate_evidence": True,
    "record_missing_evidence": True,
}

FAIL_SAFE = {
    "enabled": True,
    "abort_on_lookahead": True,
    "abort_on_missing_formation_date": True,
    "abort_on_invalid_pit_data": True,
    "abort_on_empty_universe": True,
    "abort_on_future_return_in_operational_score": True,
    "abort_on_unknown_publication_date": True,
    "abort_on_legacy_ranking": True,
    "abort_on_unvalidated_methodology": True,
    "allow_zero_approved_issuers": True,
}


def _require_exact_true(value: Any, name: str) -> None:
    if value is not True:
        raise RuntimeError(f"FAIL-SAFE: {name} deve ser exatamente True.")


def _require_exact_false(value: Any, name: str) -> None:
    if value is not False:
        raise RuntimeError(f"FAIL-SAFE: {name} deve ser exatamente False.")


def validate_config(*, require_production_ready: bool = False) -> bool:
    """Valida a configuração; produção exige validação metodológica adicional.

    O retorno True significa apenas que a configuração de TRANSIÇÃO é segura;
    não significa que os motores antigos estão autorizados a operar.
    """
    _require_exact_true(POINT_IN_TIME_REQUIRED, "POINT_IN_TIME_REQUIRED")
    _require_exact_false(ALLOW_LOOKAHEAD, "ALLOW_LOOKAHEAD")
    _require_exact_false(ALLOW_FUTURE_RETURN, "ALLOW_FUTURE_RETURN")
    _require_exact_true(REQUIRE_ACTUAL_PUBLICATION_DATE, "REQUIRE_ACTUAL_PUBLICATION_DATE")
    _require_exact_true(REJECT_UNKNOWN_AVAILABILITY_DATE, "REJECT_UNKNOWN_AVAILABILITY_DATE")
    _require_exact_true(AUDIT["enabled"], "AUDIT.enabled")
    _require_exact_true(FAIL_SAFE["enabled"], "FAIL_SAFE.enabled")
    _require_exact_false(TURNAROUND_ENGINE["enabled"], "TURNAROUND_ENGINE.enabled")
    _require_exact_false(
        TURNAROUND_ENGINE["production_authorized"],
        "TURNAROUND_ENGINE.production_authorized",
    )
    _require_exact_false(RANKING_ENGINE["allow_future_return"], "RANKING_ENGINE.allow_future_return")
    _require_exact_false(RANKING_ENGINE["allow_research_factors"], "RANKING_ENGINE.allow_research_factors")
    _require_exact_false(STUDY_34["allow_unverified_assumptions"], "STUDY_34.allow_unverified_assumptions")
    _require_exact_false(
        STUDY_34["allow_automatic_buy_recommendation"],
        "STUDY_34.allow_automatic_buy_recommendation",
    )
    if RANKING_ENGINE["model"] in REJECTED_RULES:
        raise RuntimeError("FAIL-SAFE: ranking legado não autorizado.")
    if RANKING_ENGINE["study_weights"] or SELECTION_CRITERIA["ranking_weights"]:
        raise RuntimeError("FAIL-SAFE: pesos não validados não são permitidos.")
    if require_production_ready:
        if not SELECTION_CRITERIA["methodology_ready"]:
            raise RuntimeError("FAIL-SAFE: matriz das 34 etapas não está validada.")
        if not STUDY_34["criteria_matrix_validated"]:
            raise RuntimeError("FAIL-SAFE: critérios das 34 etapas não estão homologados.")
        if not RANKING_ENGINE["enabled"] or not VALUATION_ENGINE["enabled"]:
            raise RuntimeError("FAIL-SAFE: motores de seleção/valuation ainda bloqueados.")
    return True


if __name__ == "__main__":
    validate_config()
    print(f"{PROJECT_NAME} | {PROJECT_VERSION}")
    print("Configuração de transição: VALIDADA")
    print("Modelo antigo MB_100: DESATIVADO")
    print("Seleção e recomendação automáticas: BLOQUEADAS")
    print("Gates sem evidência: PENDENTE")
    print("Produção: NÃO AUTORIZADA até integração e testes")
