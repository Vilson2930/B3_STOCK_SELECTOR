# -*- coding: utf-8 -*-
"""
B3 STOCK SELECTOR
reports/email_report.py

Envio de auditoria pendente das 34 etapas por Gmail/SMTP.

Segurança:
- credenciais somente por variáveis de ambiente;
- nunca grava senha em arquivo ou log;
- falha se os outputs obrigatórios não forem encontrados;
- não recalcula ranking, score ou carteira;
- não envia Top 10 nem recomendações não aprovadas;
- usa somente os resultados já produzidos pelo pipeline.
"""

from __future__ import annotations

import csv
import html
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
EXPECTED_WEIGHTS = {}  # Nenhum peso de seleção aprovado.


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"FAIL-SAFE EMAIL: variável de ambiente obrigatória ausente: {name}"
        )
    return value


def _latest_file(patterns: Sequence[str]) -> Optional[Path]:
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(OUTPUTS_DIR.glob(pattern))

    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_portfolio_file() -> Path:
    """Legado bloqueado: nunca usar arquivos Top 10 como aprovação."""
    raise RuntimeError(
        "FAIL-SAFE EMAIL: seleção antiga desativada. "
        "Use exclusivamente um manifesto de auditoria das 34 etapas."
    )


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(
            f"FAIL-SAFE EMAIL: arquivo de carteira vazio: {path}"
        )

    return rows


def _first_present(row: Dict[str, str], names: Sequence[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _to_float(value: str) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def _rank_value(row: Dict[str, str]) -> float:
    raw = _first_present(row, ("FINAL_RANK", "RANK", "POSITION", "POSICAO"))
    value = _to_float(raw)
    return value if value is not None else float("inf")


def _normalize_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    normalized: List[Dict[str, object]] = []

    for row in rows:
        ticker = _first_present(row, ("TICKERS", "TICKER", "CODNEG"))
        rank_raw = _first_present(row, ("FINAL_RANK", "RANK", "POSITION", "POSICAO"))
        score_raw = _first_present(
            row,
            ("FINAL_SCORE", "TURNAROUND_RESEARCH_SCORE", "SCORE"),
        )
        weight_raw = _first_present(
            row,
            ("PORTFOLIO_WEIGHT", "WEIGHT", "PESO"),
        )
        sector = _first_present(
            row,
            (
                "FCA_SETOR_ATIVIDADE",
                "SECTOR",
                "SECTOR_BUCKET",
                "SETOR",
            ),
        )

        if not ticker:
            raise RuntimeError(
                "FAIL-SAFE EMAIL: carteira selecionada contém linha sem ticker."
            )

        score = _to_float(score_raw)
        weight = _to_float(weight_raw)

        normalized.append(
            {
                "rank": int(float(rank_raw)) if rank_raw else None,
                "ticker": ticker,
                "score": score,
                "weight": weight,
                "sector": sector or "Não informado",
            }
        )

    normalized.sort(
        key=lambda x: x["rank"] if x["rank"] is not None else 10**9
    )

    return normalized


def _validate_portfolio(rows: List[Dict[str, object]]) -> None:
    """API preservada; carteiras não são autorizadas neste estágio."""
    if rows:
        raise RuntimeError(
            "FAIL-SAFE EMAIL: não enviar carteira, Top 10 ou recomendações "
            "sem aprovação documental integral das 34 etapas."
        )


def _format_weight(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    numeric = float(value)
    if numeric <= 1.0:
        numeric *= 100.0
    return f"{numeric:.2f}%"


def _format_score(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{float(value):.6f}"


def _find_attachment() -> Optional[Path]:
    """Não anexa arquivos Top 10 ou relatórios antigos de carteira."""
    return None


def _plain_body(rows: List[Dict[str, object]], source: Path) -> str:
    _validate_portfolio(rows)
    return (
        "B3 STOCK SELECTOR — AUDITORIA DO ESTUDO DE 34 ETAPAS\n\n"
        "Situação: PENDENTE DE EVIDÊNCIAS.\n"
        "Empresas integralmente aprovadas: 0.\n"
        "Carteira e recomendações de compra: NÃO AUTORIZADAS.\n\n"
        "Permanecem pendentes verificações independentes de CAPEX, "
        "fluxo de caixa livre, dívida líquida, ROIC/WACC, valor justo "
        "e documentação Point-in-Time.\n\n"
        f"Fonte do manifesto: {source.name}\n"
        "Este e-mail é um registro de pesquisa, não uma recomendação."
    )


def _html_body(rows: List[Dict[str, object]], source: Path) -> str:
    _validate_portfolio(rows)
    body = html.escape(_plain_body(rows, source))
    return (
        '<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
        '</head><body><h2>B3 Stock Selector — Estudo 34 etapas</h2>'
        '<pre style="white-space:pre-wrap">' + body + '</pre></body></html>'
    )


def _attach_file(message: EmailMessage, path: Path) -> None:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        maintype, subtype = "application", "pdf"
    elif suffix == ".html":
        maintype, subtype = "text", "html"
    elif suffix == ".csv":
        maintype, subtype = "text", "csv"
    else:
        maintype, subtype = "application", "octet-stream"

    data = path.read_bytes()
    message.add_attachment(
        data,
        maintype=maintype,
        subtype=subtype,
        filename=path.name,
    )


def send_email() -> None:
    """Envia somente o estado documental pendente; nunca carteira antiga."""
    if not OUTPUTS_DIR.is_dir():
        raise RuntimeError("FAIL-SAFE EMAIL: outputs/ não existe.")

    # Manifesto explícito, com auditoria das 34 etapas. Nunca inferir aprovação
    # a partir de arquivos CSV antigos ou da presença de um relatório HTML.
    candidates = sorted(
        (p for p in OUTPUTS_DIR.glob("runs/run_manifest_*.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("FAIL-SAFE EMAIL: manifesto de execução ausente.")

    import json
    manifest_path = candidates[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("FAIL-SAFE EMAIL: manifesto inválido.")

    status = str(manifest.get("status", "")).strip().upper()
    if status not in {"PENDING_EVIDENCE", "PENDENTE"}:
        raise RuntimeError(
            "FAIL-SAFE EMAIL: manifesto sem status explícito PENDENTE. "
            "Nenhum relatório será enviado."
        )

    audit = manifest.get("audit")
    if not isinstance(audit, dict):
        raise RuntimeError("FAIL-SAFE EMAIL: auditoria global ausente.")
    for key in ("selected_positions", "ranking_eligible"):
        try:
            count = int(audit[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"FAIL-SAFE EMAIL: {key} não auditado.") from exc
        if count != 0:
            raise RuntimeError(f"FAIL-SAFE EMAIL: {key} não é zero.")
    try:
        portfolio_weight = float(audit["portfolio_weight"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("FAIL-SAFE EMAIL: peso não auditado.") from exc
    import math
    if not math.isfinite(portfolio_weight) or portfolio_weight != 0:
        raise RuntimeError("FAIL-SAFE EMAIL: peso de carteira diferente de zero.")
    if audit.get("future_return_used") is not False:
        raise RuntimeError("FAIL-SAFE EMAIL: uso futuro não auditado como falso.")
    if audit.get("turnaround_research_used_in_ranking") is not False:
        raise RuntimeError("FAIL-SAFE EMAIL: modelo antigo não bloqueado.")

    # Credenciais são lidas apenas após a validação documental do manifesto.
    gmail_user = _required_env("GMAIL_USER")
    gmail_app_password = _required_env("GMAIL_APP_PASSWORD")
    email_to = _required_env("EMAIL_TO")
    today = datetime.now().astimezone().strftime("%d/%m/%Y")
    message = EmailMessage()
    message["From"] = gmail_user
    message["To"] = email_to
    message["Subject"] = f"B3 Stock Selector — Auditoria PENDENTE — {today}"
    message.set_content(_plain_body([], manifest_path))
    message.add_alternative(_html_body([], manifest_path), subtype="html")
    _attach_file(message, manifest_path)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context, timeout=30) as smtp:
        smtp.login(gmail_user, gmail_app_password)
        smtp.send_message(message)
    print("EMAIL REPORT: auditoria PENDENTE enviada; nenhuma carteira anexada.")


def main() -> None:
    send_email()


if __name__ == "__main__":
    main()
