# -*- coding: utf-8 -*-
"""
B3 STOCK SELECTOR — reports/email_report.py

Canal de comunicação da auditoria das 34 etapas.
NÃO constitui motor de recomendação de investimento.

Contratos preservados do módulo original:
    _required_env, _latest_file, _find_portfolio_file, _read_csv,
    _first_present, _to_float, _rank_value, _normalize_rows,
    _validate_portfolio, _format_weight, _format_score,
    _find_attachment, _plain_body, _html_body, _attach_file,
    send_email, main.

Princípios:
- Só comunica dados de um manifesto comprovadamente pendente.
- Nunca infere aprovação pelo nome de um CSV ou HTML.
- Nunca envia Top 10, score, pesos de carteira ou recomendação antiga.
- Nunca usa retorno futuro para classificar empresas.
- Não modifica nem recalcula os motores do estudo.
- Não apresenta uma execução incompleta como conclusão científica.
- O arquivo original permanece como referência de compatibilidade.
"""
from __future__ import annotations

import csv
import html
import json
import math
import os
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Optional, Sequence


OUTPUTS_DIR = Path("outputs")
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

EXPECTED_MODEL = "STUDY_34_PENDING_EVIDENCE"
EXPECTED_WEIGHTS = {}

PENDING_STATUSES = frozenset({"PENDENTE", "PENDING_EVIDENCE"})
FORBIDDEN_PORTFOLIO_PATTERNS = (
    "ranking/selected_top10_*.csv",
    "risk/portfolio_*.csv",
    "reports/portfolio_report_*.csv",
)
FORBIDDEN_AUDIT_TRUE_FIELDS = (
    "future_return_used",
    "turnaround_research_used_in_ranking",
)
MANDATORY_ZERO_FIELDS = (
    "selected_positions",
    "ranking_eligible",
)
METHODOLOGY_NOTES = (
    "Pesquisa científica das 34 etapas: comprovação integral pendente.",
    "Nenhuma empresa possui aprovação integral confirmada neste manifesto.",
    "Modelo histórico de margens com pesos arbitrários: desativado.",
    "Nenhuma carteira, Top 10 ou recomendação de compra autorizada.",
    "CAPEX, FCL, dívida líquida, ROIC/WACC e valor justo exigem evidência.",
    "Não confundir ausência de comprovação com reprovação econômica.",
)


class EmailReportIntegrityError(RuntimeError):
    """Falha de integridade antes da transmissão por e-mail."""


def _required_env(name: str) -> str:
    """Recupera credencial obrigatória sem registrar seu conteúdo."""
    value = os.getenv(name, "").strip()
    if not value:
        raise EmailReportIntegrityError(
            "FAIL-SAFE EMAIL: variável de ambiente ausente: "
            f"{name}"
        )
    return value


def _latest_file(patterns: Sequence[str]) -> Optional[Path]:
    """Localiza arquivo existente, sem declarar que ele é válido."""
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(OUTPUTS_DIR.glob(pattern))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _find_portfolio_file() -> Path:
    """
    Compatibilidade com o nome da função legada.

    A ausência de carteira é intencional: as 34 etapas ainda não
    autorizaram seleção. Não utilizar CSV antigo como fonte de verdade.
    """
    raise EmailReportIntegrityError(
        "FAIL-SAFE EMAIL: arquivo de carteira/Top 10 não autorizado. "
        "Use o manifesto de auditoria PENDENTE."
    )


def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Leitura técnica, sem converter linhas em ações selecionadas."""
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_file():
        raise EmailReportIntegrityError(
            f"FAIL-SAFE EMAIL: arquivo CSV ausente: {path}"
        )
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise EmailReportIntegrityError(
                "FAIL-SAFE EMAIL: CSV sem cabeçalho."
            )
        return list(reader)


def _first_present(row: Dict[str, str], names: Sequence[str]) -> str:
    """Acesso a campos legados somente para inspeção descritiva."""
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _to_float(value: str) -> Optional[float]:
    """Conversão numérica segura; NaN e infinito não são evidências."""
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        number = float(text.replace(",", "."))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rank_value(row: Dict[str, str]) -> float:
    """
    Função mantida para compatibilidade de importação.

    Não produz autorização de seleção ou prioridade de investimento.
    """
    value = _to_float(
        _first_present(row, ("FINAL_RANK", "RANK", "POSITION", "POSICAO"))
    )
    return value if value is not None else float("inf")


def _normalize_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    """
    Converte apenas metadados descritivos de pesquisa.

    Não converte ranking, score ou peso em recomendação de investimento.
    """
    normalized: List[Dict[str, object]] = []
    for row in rows:
        ticker = _first_present(row, ("TICKERS", "TICKER", "CODNEG"))
        issuer = _first_present(row, ("ISSUER_ID", "CD_CVM", "CNPJ"))
        status = _first_present(
            row,
            ("STUDY_34_STATUS", "RESEARCH_STATUS", "STATUS"),
        )
        if not issuer and not ticker:
            raise EmailReportIntegrityError(
                "FAIL-SAFE EMAIL: linha sem identidade de ativo/emissor."
            )
        if status.strip().upper() not in PENDING_STATUSES:
            raise EmailReportIntegrityError(
                "FAIL-SAFE EMAIL: status documental não comprovado como PENDENTE."
            )
        normalized.append(
            {
                "rank": None,
                "ticker": ticker,
                "issuer_id": issuer,
                "score": None,
                "weight": None,
                "sector": _first_present(
                    row,
                    ("FCA_SETOR_ATIVIDADE", "SECTOR", "SECTOR_BUCKET", "SETOR"),
                ) or "Não informado",
                "status": "PENDENTE",
            }
        )
    return normalized


def _validate_portfolio(rows: List[Dict[str, object]]) -> None:
    """
    Bloqueio explícito de carteiras, inclusive quando linhas vierem
    de funções antigas ou de um arquivo CSV externo.
    """
    if rows:
        raise EmailReportIntegrityError(
            "FAIL-SAFE EMAIL: carteira/Top 10 não autorizado "
            "pelo estudo das 34 etapas."
        )


def _format_weight(value: object) -> str:
    """Peso de carteira não autorizado; nunca exibir como sugestão."""
    return "NÃO AUTORIZADO"


def _format_score(value: object) -> str:
    """Score legado não validado; nunca exibir como indicador operacional."""
    return "NÃO VALIDADO"


def _find_attachment() -> Optional[Path]:
    """
    O único anexo permitido é o manifesto JSON que será validado.

    Evita anexar por engano relatórios HTML/CSV antigos e não auditados.
    """
    return _latest_file(("runs/run_manifest_*.json",))


def _plain_body(rows: List[Dict[str, object]], source: Path) -> str:
    """Corpo de texto da comunicação, sem recomendação de investimento."""
    _validate_portfolio(rows)
    lines = [
        "B3 STOCK SELECTOR",
        "",
        "RELATÓRIO DE PESQUISA — 34 ETAPAS",
        "STATUS: PENDENTE DE EVIDÊNCIAS",
        "",
        *METHODOLOGY_NOTES,
        "",
        f"Manifesto de origem: {source.as_posix()}",
        "",
        "Esta mensagem não contém carteira ou indicação de compra.",
        "Mensagem automática do B3 Stock Selector.",
    ]
    return "\n".join(lines)


def _html_body(rows: List[Dict[str, object]], source: Path) -> str:
    """Corpo HTML com conteúdo equivalente ao texto puro."""
    _validate_portfolio(rows)
    items = "".join(
        "<li>" + html.escape(note) + "</li>"
        for note in METHODOLOGY_NOTES
    )
    source_html = html.escape(source.as_posix(), quote=True)
    return (
        '<!doctype html>\n'
        '<html lang="pt-BR">\n'
        '<head><meta charset="utf-8"></head>\n'
        '<body>\n'
        '<h2>B3 Stock Selector — pesquisa das 34 etapas</h2>\n'
        '<p><strong>Status: PENDENTE DE EVIDÊNCIAS</strong></p>\n'
        f'<ul>{items}</ul>\n'
        f'<p>Manifesto de origem: <code>{source_html}</code></p>\n'
        '<p>Sem carteira, Top 10 ou recomendação de compra.</p>\n'
        '</body>\n'
        '</html>\n'
    )


def _attach_file(message: EmailMessage, path: Path) -> None:
    """
    Somente o manifesto JSON validado pode ser anexado.

    A função preserva a assinatura original e impede que anexos
    legados sejam enviados por acidente.
    """
    if path.suffix.lower() != ".json":
        raise EmailReportIntegrityError(
            "FAIL-SAFE EMAIL: somente manifesto JSON auditado é permitido."
        )
    if not path.is_file():
        raise EmailReportIntegrityError(
            "FAIL-SAFE EMAIL: manifesto para anexo não encontrado."
        )
    message.add_attachment(
        path.read_bytes(),
        maintype="application",
        subtype="json",
        filename=path.name,
    )


def _load_latest_manifest() -> tuple[Path, dict]:
    """
    Carrega somente o manifesto mais recente.

    Não recua para um manifesto antigo se o mais recente estiver inválido:
    isso evitaria mascarar uma execução que falhou.
    """
    path = _find_attachment()
    if path is None:
        raise EmailReportIntegrityError(
            "FAIL-SAFE EMAIL: manifesto de execução ausente."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EmailReportIntegrityError(
            "FAIL-SAFE EMAIL: manifesto ilegível ou JSON inválido."
        ) from exc
    if not isinstance(payload, dict):
        raise EmailReportIntegrityError(
            "FAIL-SAFE EMAIL: manifesto deve ser um objeto JSON."
        )
    return path, payload


def _validate_manifest(manifest: dict) -> dict:
    """
    Validação documental do manifesto.

    Este teste verifica consistência mínima e proíbe seleção indevida.
    Não substitui a auditoria econômica das 34 etapas.
    """
    status = str(manifest.get("status", "")).strip().upper()
    if status not in PENDING_STATUSES:
        raise EmailReportIntegrityError(
            "FAIL-SAFE EMAIL: manifesto sem status PENDENTE explícito."
        )

    audit = manifest.get("audit")
    if not isinstance(audit, dict):
        raise EmailReportIntegrityError(
            "FAIL-SAFE EMAIL: auditoria global ausente."
        )

    for key in MANDATORY_ZERO_FIELDS:
        value = audit.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise EmailReportIntegrityError(
                f"FAIL-SAFE EMAIL: {key} deve ser inteiro zero."
            )

    weight = audit.get("portfolio_weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise EmailReportIntegrityError(
            "FAIL-SAFE EMAIL: peso de carteira não auditado."
        )
    if not math.isfinite(weight) or weight != 0:
        raise EmailReportIntegrityError(
            "FAIL-SAFE EMAIL: carteira com peso diferente de zero."
        )

    for key in FORBIDDEN_AUDIT_TRUE_FIELDS:
        if audit.get(key) is not False:
            raise EmailReportIntegrityError(
                f"FAIL-SAFE EMAIL: {key} não está comprovado como falso."
            )

    # Impede a apresentação de aprovações sem documentação auditada.
    study = manifest.get("study_34", audit.get("study_34"))
    if study is not None:
        if not isinstance(study, dict):
            raise EmailReportIntegrityError(
                "FAIL-SAFE EMAIL: metadados das 34 etapas inválidos."
            )
        approved = study.get("fully_approved_issuers", 0)
        if isinstance(approved, bool) or approved != 0:
            raise EmailReportIntegrityError(
                "FAIL-SAFE EMAIL: emissores aprovados sem comprovação."
            )
        if study.get("investment_recommendations_authorized", False) is not False:
            raise EmailReportIntegrityError(
                "FAIL-SAFE EMAIL: recomendações indevidamente autorizadas."
            )

    return audit


def _build_message(manifest_path: Path, manifest: dict) -> EmailMessage:
    """
    Monta a mensagem após a validação, sem acessar SMTP.

    Essa separação permite testes offline antes de transmitir e-mail.
    """
    _validate_manifest(manifest)
    sender = _required_env("GMAIL_USER")
    recipient = _required_env("EMAIL_TO")

    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    date_text = datetime.now().astimezone().strftime("%d/%m/%Y")
    message["Subject"] = (
        f"B3 Stock Selector — Estudo 34 etapas — PENDENTE — {date_text}"
    )
    message.set_content(_plain_body([], manifest_path))
    message.add_alternative(
        _html_body([], manifest_path),
        subtype="html",
    )
    _attach_file(message, manifest_path)
    return message


def send_email() -> None:
    """
    Envia o manifesto pendente por SMTP_SSL somente após a auditoria.

    Nunca imprime credenciais. Nenhuma conexão é feita antes da
    validação do conteúdo e das variáveis de ambiente obrigatórias.
    """
    if not OUTPUTS_DIR.is_dir():
        raise EmailReportIntegrityError(
            "FAIL-SAFE EMAIL: diretório outputs/ não existe."
        )

    manifest_path, manifest = _load_latest_manifest()
    _validate_manifest(manifest)

    message = _build_message(manifest_path, manifest)
    sender = _required_env("GMAIL_USER")
    password = _required_env("GMAIL_APP_PASSWORD")

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(
        SMTP_HOST,
        SMTP_PORT,
        context=context,
        timeout=30,
    ) as smtp:
        smtp.login(sender, password)
        smtp.send_message(message)

    print("=" * 72)
    print("B3 STOCK SELECTOR — RELATÓRIO DE PESQUISA")
    print("Status: PENDENTE — comunicado enviado")
    print(f"Manifesto: {manifest_path}")
    print("Posições selecionadas: 0")
    print("Recomendações de compra: nenhuma")
    print("Credenciais: NÃO EXIBIDAS")
    print("=" * 72)


def main() -> None:
    """Ponto de entrada mantido por compatibilidade."""
    send_email()


def _self_test() -> bool:
    """Teste offline: nenhum SMTP é iniciado."""
    valid = {
        "status": "PENDING_EVIDENCE",
        "audit": {
            "selected_positions": 0,
            "ranking_eligible": 0,
            "portfolio_weight": 0.0,
            "future_return_used": False,
            "turnaround_research_used_in_ranking": False,
        },
        "study_34": {
            "fully_approved_issuers": 0,
            "investment_recommendations_authorized": False,
        },
    }
    _validate_manifest(valid)
    for change in (
        {"status": "SUCCESS"},
        {"audit": {**valid["audit"], "selected_positions": 1}},
        {"audit": {**valid["audit"], "portfolio_weight": float("nan")}},
        {"audit": {**valid["audit"], "future_return_used": True}},
        {"audit": {**valid["audit"], "ranking_eligible": 1}},
        {"audit": {**valid["audit"], "turnaround_research_used_in_ranking": True}},
    ):
        invalid = {**valid, **change}
        try:
            _validate_manifest(invalid)
        except EmailReportIntegrityError:
            pass
        else:
            raise AssertionError(f"Bloqueio falhou: {change!r}")

    for rows in ([{"ticker": "TEST3", "weight": 1.0}],):
        try:
            _validate_portfolio(rows)
        except EmailReportIntegrityError:
            pass
        else:
            raise AssertionError("Carteira antiga não foi bloqueada.")

    if "Rank | Ticker" in _plain_body([], Path("audit.json")):
        raise AssertionError("Conteúdo legado em relatório.")
    return True


if __name__ == "__main__":
    main()
