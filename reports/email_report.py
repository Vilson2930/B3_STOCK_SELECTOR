# -*- coding: utf-8 -*-
"""
B3 STOCK SELECTOR
reports/email_report.py

Envio do relatório de uma execução válida por Gmail/SMTP.

Segurança:
- credenciais somente por variáveis de ambiente;
- nunca grava senha em arquivo ou log;
- falha se os outputs obrigatórios não forem encontrados;
- não recalcula ranking, score ou carteira;
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

EXPECTED_MODEL = "MB_50_ML_30_ROE_20_STUDY"
EXPECTED_WEIGHTS = {
    "MARGEM_BRUTA": 0.50,
    "MARGEM_LIQUIDA": 0.30,
    "ROE": 0.20,
}


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
    path = _latest_file(
        (
            "reports/portfolio_report_*.csv",
            "risk/portfolio_*.csv",
            "ranking/selected_top10_*.csv",
        )
    )
    if path is None:
        raise RuntimeError(
            "FAIL-SAFE EMAIL: não foi encontrado relatório/portfolio/Top10 "
            "válido em outputs/."
        )
    return path


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
    if not rows:
        raise RuntimeError("FAIL-SAFE EMAIL: nenhuma posição selecionada.")

    if len(rows) > 10:
        raise RuntimeError(
            f"FAIL-SAFE EMAIL: relatório selecionado contém {len(rows)} linhas; "
            "esperado no máximo 10 posições."
        )

    tickers = [str(row["ticker"]) for row in rows]
    if len(tickers) != len(set(tickers)):
        raise RuntimeError(
            "FAIL-SAFE EMAIL: existem tickers duplicados na carteira."
        )

    weights = [
        row["weight"]
        for row in rows
        if isinstance(row["weight"], (int, float))
    ]
    if weights:
        total = float(sum(weights))
        if abs(total - 1.0) > 1e-6 and abs(total - 100.0) > 1e-6:
            raise RuntimeError(
                "FAIL-SAFE EMAIL: soma dos pesos da carteira não é 1.0 nem 100%. "
                f"Valor encontrado: {total}"
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
    # Prioridade para relatórios finais legíveis. Se não houver PDF/HTML,
    # anexa o CSV final usado para montar o e-mail.
    path = _latest_file(
        (
            "reports/*.pdf",
            "reports/*.html",
            "reports/portfolio_report_*.csv",
        )
    )
    return path


def _plain_body(rows: List[Dict[str, object]], source: Path) -> str:
    lines = [
        "B3 STOCK SELECTOR",
        "",
        "Execução concluída com sucesso.",
        "",
        "Modelo operacional:",
        "50% Margem Bruta + 30% Margem Líquida + 20% ROE",
        "Direção dos três fatores: LOW.",
        "Bancos/financeiras: excluídos do modelo fundamental padrão.",
        "",
        "TOP 10",
        "",
        "Rank | Ticker | Score | Peso | Setor",
        "-" * 72,
    ]

    for index, row in enumerate(rows, start=1):
        rank = row["rank"] if row["rank"] is not None else index
        lines.append(
            f"{rank} | {row['ticker']} | {_format_score(row['score'])} | "
            f"{_format_weight(row['weight'])} | {row['sector']}"
        )

    lines.extend(
        [
            "",
            f"Arquivo-fonte: {source.as_posix()}",
            f"Modelo: {EXPECTED_MODEL}",
            "",
            "Mensagem automática do B3 Stock Selector.",
        ]
    )

    return "\n".join(lines)


def _html_body(rows: List[Dict[str, object]], source: Path) -> str:
    table_rows = []

    for index, row in enumerate(rows, start=1):
        rank = row["rank"] if row["rank"] is not None else index
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(rank))}</td>"
            f"<td><strong>{html.escape(str(row['ticker']))}</strong></td>"
            f"<td>{html.escape(_format_score(row['score']))}</td>"
            f"<td>{html.escape(_format_weight(row['weight']))}</td>"
            f"<td>{html.escape(str(row['sector']))}</td>"
            "</tr>"
        )

    return f"""\
<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
</head>
<body>
<h2>B3 Stock Selector</h2>
<p><strong>Execução concluída com sucesso.</strong></p>

<p>
Modelo operacional:
<strong>50% Margem Bruta + 30% Margem Líquida + 20% ROE</strong><br>
Direção dos três fatores: <strong>LOW</strong><br>
Bancos/financeiras: <strong>excluídos do modelo fundamental padrão</strong>.
</p>

<h3>Top 10</h3>
<table border="1" cellpadding="6" cellspacing="0">
<thead>
<tr>
<th>Rank</th>
<th>Ticker</th>
<th>Score</th>
<th>Peso</th>
<th>Setor</th>
</tr>
</thead>
<tbody>
{''.join(table_rows)}
</tbody>
</table>

<p>
Arquivo-fonte: <code>{html.escape(source.as_posix())}</code><br>
Modelo: <code>{EXPECTED_MODEL}</code>
</p>

<p>Mensagem automática do B3 Stock Selector.</p>
</body>
</html>
"""


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
    if not OUTPUTS_DIR.is_dir():
        raise RuntimeError(
            "FAIL-SAFE EMAIL: diretório outputs/ não existe."
        )

    gmail_user = _required_env("GMAIL_USER")
    gmail_app_password = _required_env("GMAIL_APP_PASSWORD")
    email_to = _required_env("EMAIL_TO")

    portfolio_file = _find_portfolio_file()
    raw_rows = _read_csv(portfolio_file)
    raw_rows.sort(key=_rank_value)

    rows = _normalize_rows(raw_rows)
    _validate_portfolio(rows)

    today = datetime.now().astimezone().strftime("%d/%m/%Y")
    subject = f"B3 Stock Selector — Top 10 — {today}"

    message = EmailMessage()
    message["From"] = gmail_user
    message["To"] = email_to
    message["Subject"] = subject

    message.set_content(_plain_body(rows, portfolio_file))
    message.add_alternative(
        _html_body(rows, portfolio_file),
        subtype="html",
    )

    attachment = _find_attachment()
    if attachment is not None:
        _attach_file(message, attachment)

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL(
        SMTP_HOST,
        SMTP_PORT,
        context=context,
        timeout=30,
    ) as smtp:
        smtp.login(gmail_user, gmail_app_password)
        smtp.send_message(message)

    print("=" * 72)
    print("EMAIL REPORT")
    print("=" * 72)
    print("Status: ENVIADO")
    print(f"Remetente: {gmail_user}")
    print(f"Destinatário: {email_to}")
    print(f"Posições: {len(rows)}")
    print(f"Fonte: {portfolio_file}")
    print(f"Anexo: {attachment if attachment else 'nenhum'}")
    print("Credenciais: NÃO EXIBIDAS")
    print("=" * 72)


def main() -> None:
    send_email()


if __name__ == "__main__":
    main()
