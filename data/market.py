# =============================================================================
# B3 STOCK SELECTOR
# Arquivo: data/market.py
#
# MARKET DATA ENGINE
#
# RESPONSABILIDADE:
# - Baixar COTAHIST oficial da B3
# - Ler cotações históricas
# - Identificar ações ordinárias/preferenciais permitidas
# - Calcular número de sessões negociadas
# - Obter preço de formação
# - Criar snapshot de mercado compatível com PIT
# - Manter cache, SHA-256 e auditoria
#
# NÃO FAZ:
# - Retorno futuro
# - Seleção de winners
# - Fundamental Score
# - Quality Score
# - Turnaround Score
# - Valuation
# - Ranking
#
# PRINCÍPIOS:
# - Dados de mercado separados dos fundamentos
# - Sem look-ahead
# - Fail-safe
# - Determinístico
# =============================================================================

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

try:
    from config import (
        DATA_DIR,
        ALLOWED_SHARE_SUFFIXES,
        MIN_TRADING_SESSIONS,
        ALLOW_LOOKAHEAD,
    )

except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar config.py."
    ) from exc


try:
    from data.pit import (
        PITContext,
        normalize_date,
        market_snapshot,
    )

except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar data.pit."
    ) from exc


# =============================================================================
# DIRETÓRIOS
# =============================================================================

MARKET_DIR = DATA_DIR / "market"
RAW_DIR = MARKET_DIR / "raw"
MANIFEST_DIR = MARKET_DIR / "manifests"

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# CONFIGURAÇÃO COTAHIST
# =============================================================================

B3_COTAHIST_URL = (
    "https://bvmf.bmfbovespa.com.br/"
    "InstDados/SerHist/COTAHIST_A{year}.ZIP"
)

REQUEST_TIMEOUT = 180
CHUNK_SIZE = 1024 * 1024

USER_AGENT = (
    "B3-STOCK-SELECTOR/0.1 "
    "(research; public historical market data)"
)


# =============================================================================
# LAYOUT OFICIAL COTAHIST
#
# Posições Python são zero-based.
# O arquivo B3 é fixed-width.
# =============================================================================

COTAHIST_LAYOUT = {
    "TIPREG": (0, 2),
    "DATA_PREGAO": (2, 10),
    "CODBDI": (10, 12),
    "CODNEG": (12, 24),
    "TPMERC": (24, 27),
    "NOMRES": (27, 39),
    "ESPECI": (39, 49),
    "PRAZOT": (49, 52),
    "MODREF": (52, 56),
    "PREABE": (56, 69),
    "PREMAX": (69, 82),
    "PREMIN": (82, 95),
    "PREMED": (95, 108),
    "PREULT": (108, 121),
    "PREOFC": (121, 134),
    "PREOFV": (134, 147),
    "TOTNEG": (147, 152),
    "QUATOT": (152, 170),
    "VOLTOT": (170, 188),
    "PREEXE": (188, 201),
    "INDOPC": (201, 202),
    "DATVEN": (202, 210),
    "FATCOT": (210, 217),
    "PTOEXE": (217, 230),
    "CODISI": (230, 242),
    "DISMES": (242, 245),
}


PRICE_COLUMNS = (
    "PREABE",
    "PREMAX",
    "PREMIN",
    "PREMED",
    "PREULT",
    "PREOFC",
    "PREOFV",
    "PREEXE",
)


INTEGER_COLUMNS = (
    "TOTNEG",
    "QUATOT",
    "VOLTOT",
    "FATCOT",
)


# =============================================================================
# EXCEÇÕES
# =============================================================================

class MarketDataError(RuntimeError):
    """Erro geral do Market Data Engine."""


class MarketDownloadError(MarketDataError):
    """Erro no download."""


class MarketIntegrityError(MarketDataError):
    """Erro de integridade."""


class MarketParseError(MarketDataError):
    """Erro na interpretação do COTAHIST."""


# =============================================================================
# UTILIDADES
# =============================================================================

def _utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _validate_year(year: int) -> int:

    try:
        year = int(year)

    except Exception as exc:
        raise MarketDataError(
            f"Ano inválido: {year}"
        ) from exc

    current_year = datetime.now().year

    if year < 1986:
        raise MarketDataError(
            f"Ano fora do intervalo esperado: {year}"
        )

    if year > current_year:
        raise MarketDataError(
            "FAIL-SAFE: tentativa de acessar "
            f"ano futuro: {year}"
        )

    return year


def _sha256(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(CHUNK_SIZE),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# =============================================================================
# CAMINHO LOCAL
# =============================================================================

def local_cotahist_path(
    year: int,
) -> Path:

    year = _validate_year(year)

    return (
        RAW_DIR
        / f"COTAHIST_A{year}.ZIP"
    )


# =============================================================================
# URL
# =============================================================================

def build_cotahist_url(
    year: int,
) -> str:

    year = _validate_year(year)

    return B3_COTAHIST_URL.format(
        year=year
    )


# =============================================================================
# VALIDAÇÃO ZIP
# =============================================================================

def validate_cotahist_zip(
    path: Path,
) -> None:

    if not path.exists():

        raise MarketIntegrityError(
            f"COTAHIST não encontrado: {path}"
        )

    if path.stat().st_size <= 0:

        raise MarketIntegrityError(
            f"COTAHIST vazio: {path}"
        )

    if not zipfile.is_zipfile(path):

        raise MarketIntegrityError(
            f"Arquivo inválido: {path}"
        )

    try:

        with zipfile.ZipFile(
            path,
            "r",
        ) as archive:

            if not archive.namelist():

                raise MarketIntegrityError(
                    "ZIP COTAHIST sem conteúdo."
                )

            bad_file = archive.testzip()

            if bad_file is not None:

                raise MarketIntegrityError(
                    "ZIP COTAHIST corrompido: "
                    f"{bad_file}"
                )

    except MarketIntegrityError:
        raise

    except Exception as exc:

        raise MarketIntegrityError(
            f"Erro ao validar {path}"
        ) from exc


# =============================================================================
# MANIFESTO
# =============================================================================

def _save_manifest(
    year: int,
    path: Path,
    source_url: str,
) -> Path:

    manifest = {
        "dataset": "COTAHIST",
        "source": "B3",
        "year": year,
        "source_url": source_url,
        "local_file": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "downloaded_at_utc": _utc_now_iso(),
    }

    output = (
        MANIFEST_DIR
        / f"cotahist_{year}_manifest.json"
    )

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
# DOWNLOAD
# =============================================================================

def download_cotahist(
    year: int,
    force: bool = False,
) -> Path:

    if ALLOW_LOOKAHEAD:
        raise MarketDataError(
            "FAIL-SAFE: ALLOW_LOOKAHEAD=True."
        )

    year = _validate_year(year)

    destination = local_cotahist_path(
        year
    )

    url = build_cotahist_url(
        year
    )

    # -------------------------------------------------------------------------
    # CACHE
    # -------------------------------------------------------------------------

    if (
        destination.exists()
        and not force
    ):

        try:

            validate_cotahist_zip(
                destination
            )

            return destination

        except MarketIntegrityError:

            destination.unlink(
                missing_ok=True
            )

    # -------------------------------------------------------------------------
    # DOWNLOAD TEMPORÁRIO
    # -------------------------------------------------------------------------

    temp_path = (
        destination.parent
        / f"{destination.name}.part"
    )

    temp_path.unlink(
        missing_ok=True
    )

    try:

        with requests.get(
            url,
            stream=True,
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent": USER_AGENT
            },
        ) as response:

            if response.status_code != 200:

                raise MarketDownloadError(
                    "Falha ao baixar COTAHIST. "
                    f"HTTP {response.status_code}. "
                    f"URL={url}"
                )

            with temp_path.open(
                "wb"
            ) as file:

                for chunk in response.iter_content(
                    chunk_size=CHUNK_SIZE
                ):

                    if chunk:
                        file.write(chunk)

    except MarketDataError:

        temp_path.unlink(
            missing_ok=True
        )

        raise

    except Exception as exc:

        temp_path.unlink(
            missing_ok=True
        )

        raise MarketDownloadError(
            f"Erro no download COTAHIST {year}: {exc}"
        ) from exc

    # -------------------------------------------------------------------------
    # VALIDAÇÃO
    # -------------------------------------------------------------------------

    try:

        validate_cotahist_zip(
            temp_path
        )

    except Exception:

        temp_path.unlink(
            missing_ok=True
        )

        raise

    temp_path.replace(
        destination
    )

    validate_cotahist_zip(
        destination
    )

    _save_manifest(
        year=year,
        path=destination,
        source_url=url,
    )

    return destination


# =============================================================================
# ARQUIVO TXT INTERNO
# =============================================================================

def _get_internal_txt(
    archive: zipfile.ZipFile,
) -> str:

    candidates = [
        name
        for name in archive.namelist()
        if name.upper().endswith(
            ".TXT"
        )
    ]

    if len(candidates) != 1:

        raise MarketParseError(
            "FAIL-SAFE: era esperado exatamente "
            "um TXT dentro do COTAHIST. "
            f"Encontrados={candidates}"
        )

    return candidates[0]


# =============================================================================
# PARSE DE UMA LINHA
# =============================================================================

def _slice(
    line: str,
    field: str,
) -> str:

    start, end = (
        COTAHIST_LAYOUT[field]
    )

    return line[
        start:end
    ]


def _parse_price(
    value: str,
) -> Optional[float]:

    value = value.strip()

    if not value:
        return None

    try:
        return int(value) / 100.0

    except Exception:
        return None


def _parse_integer(
    value: str,
) -> Optional[int]:

    value = value.strip()

    if not value:
        return None

    try:
        return int(value)

    except Exception:
        return None


def _parse_record(
    line: str,
) -> Optional[dict]:

    # Somente registro de cotação.
    if _slice(
        line,
        "TIPREG",
    ) != "01":
        return None

    raw_date = _slice(
        line,
        "DATA_PREGAO",
    )

    try:

        trading_date = pd.to_datetime(
            raw_date,
            format="%Y%m%d",
            errors="raise",
        )

    except Exception as exc:

        raise MarketParseError(
            f"Data COTAHIST inválida: {raw_date}"
        ) from exc

    record = {
        "DATA_PREGAO": trading_date,
        "CODBDI": _slice(
            line,
            "CODBDI",
        ).strip(),
        "TICKER": _slice(
            line,
            "CODNEG",
        ).strip(),
        "TPMERC": _slice(
            line,
            "TPMERC",
        ).strip(),
        "NOMRES": _slice(
            line,
            "NOMRES",
        ).strip(),
        "ESPECI": _slice(
            line,
            "ESPECI",
        ).strip(),
        "CODISI": _slice(
            line,
            "CODISI",
        ).strip(),
    }

    for column in PRICE_COLUMNS:

        record[column] = _parse_price(
            _slice(
                line,
                column,
            )
        )

    for column in INTEGER_COLUMNS:

        record[column] = _parse_integer(
            _slice(
                line,
                column,
            )
        )

    return record


# =============================================================================
# LEITURA COTAHIST
# =============================================================================

def read_cotahist(
    year: int,
    *,
    auto_download: bool = True,
) -> pd.DataFrame:

    year = _validate_year(
        year
    )

    path = local_cotahist_path(
        year
    )

    if not path.exists():

        if not auto_download:

            raise MarketDataError(
                f"COTAHIST não encontrado: {path}"
            )

        path = download_cotahist(
            year
        )

    validate_cotahist_zip(
        path
    )

    records = []

    try:

        with zipfile.ZipFile(
            path,
            "r",
        ) as archive:

            internal_txt = (
                _get_internal_txt(
                    archive
                )
            )

            with archive.open(
                internal_txt,
                "r",
            ) as raw_file:

                text_file = io.TextIOWrapper(
                    raw_file,
                    encoding="latin1",
                    errors="strict",
                )

                for line in text_file:

                    record = _parse_record(
                        line.rstrip(
                            "\r\n"
                        )
                    )

                    if record is not None:

                        records.append(
                            record
                        )

    except MarketDataError:
        raise

    except Exception as exc:

        raise MarketParseError(
            f"Erro ao ler COTAHIST {year}."
        ) from exc

    if not records:

        raise MarketParseError(
            f"Nenhuma cotação encontrada em {year}."
        )

    df = pd.DataFrame(
        records
    )

    df = df.sort_values(
        [
            "TICKER",
            "DATA_PREGAO",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    return df


# =============================================================================
# LEITURA DE VÁRIOS ANOS
# =============================================================================

def read_cotahist_years(
    years: Iterable[int],
    *,
    auto_download: bool = True,
) -> pd.DataFrame:

    frames = []

    for year in years:

        frame = read_cotahist(
            year=year,
            auto_download=auto_download,
        )

        frames.append(
            frame
        )

    if not frames:

        raise MarketDataError(
            "Nenhum ano informado."
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result = result.sort_values(
        [
            "TICKER",
            "DATA_PREGAO",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    return result


# =============================================================================
# IDENTIFICAÇÃO DE AÇÕES
# =============================================================================

def is_allowed_share_ticker(
    ticker: str,
) -> bool:
    """
    Regra conservadora usada para identificar ações.

    Exemplos:
        PETR3
        PETR4
        UNIP5
        UNIP6

    ETFs, FIIs, units e derivativos não entram por esta regra.
    """

    ticker = str(
        ticker
    ).strip().upper()

    allowed_suffixes = "".join(
        re.escape(str(x))
        for x in ALLOWED_SHARE_SUFFIXES
    )

    pattern = (
        rf"^[A-Z]{{4}}"
        rf"[{allowed_suffixes}]$"
    )

    return bool(
        re.fullmatch(
            pattern,
            ticker,
        )
    )


# =============================================================================
# FILTRO DE AÇÕES
# =============================================================================

def filter_b3_shares(
    df: pd.DataFrame,
) -> pd.DataFrame:

    required = {
        "TICKER",
        "TPMERC",
        "PREULT",
        "DATA_PREGAO",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise MarketDataError(
            "FAIL-SAFE: colunas ausentes no "
            f"mercado: {sorted(missing)}"
        )

    result = df.copy()

    result["TICKER"] = (
        result["TICKER"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Mercado à vista.
    market_mask = (
        result["TPMERC"]
        .astype(str)
        .str.strip()
        == "010"
    )

    ticker_mask = (
        result["TICKER"]
        .map(
            is_allowed_share_ticker
        )
    )

    price_mask = (
        pd.to_numeric(
            result["PREULT"],
            errors="coerce",
        )
        > 0
    )

    result = result.loc[
        market_mask
        & ticker_mask
        & price_mask
    ].copy()

    return result


# =============================================================================
# DADOS SOMENTE ATÉ O MARKET CUTOFF
# =============================================================================

def market_history_pit(
    df: pd.DataFrame,
    context: PITContext,
) -> pd.DataFrame:

    if "DATA_PREGAO" not in df.columns:

        raise MarketDataError(
            "DATA_PREGAO ausente."
        )

    result = df.copy()

    result["DATA_PREGAO"] = (
        pd.to_datetime(
            result["DATA_PREGAO"],
            errors="coerce",
        )
        .dt.normalize()
    )

    if (
        result["DATA_PREGAO"]
        .isna()
        .any()
    ):

        raise MarketDataError(
            "FAIL-SAFE: DATA_PREGAO inválida."
        )

    future = (
        result["DATA_PREGAO"]
        >
        context.market_cutoff
    )

    if future.any():

        # Não usamos esses registros.
        result = result.loc[
            ~future
        ].copy()

    return result


# =============================================================================
# CONTAGEM DE SESSÕES
# =============================================================================

def trading_session_count(
    df: pd.DataFrame,
    context: PITContext,
) -> pd.DataFrame:

    result = market_history_pit(
        df,
        context,
    )

    result = filter_b3_shares(
        result
    )

    counts = (
        result
        .groupby(
            "TICKER",
            as_index=False,
        )
        .agg(
            TRADING_SESSIONS=(
                "DATA_PREGAO",
                "nunique",
            )
        )
    )

    return counts


# =============================================================================
# UNIVERSO POR LIQUIDEZ TEMPORAL
# =============================================================================

def eligible_by_sessions(
    df: pd.DataFrame,
    context: PITContext,
    min_sessions: Optional[int] = None,
) -> pd.DataFrame:

    if min_sessions is None:
        min_sessions = (
            MIN_TRADING_SESSIONS
        )

    if int(min_sessions) <= 0:

        raise MarketDataError(
            "min_sessions deve ser positivo."
        )

    counts = trading_session_count(
        df,
        context,
    )

    counts["SESSION_ELIGIBLE"] = (
        counts["TRADING_SESSIONS"]
        >= int(min_sessions)
    )

    return counts


# =============================================================================
# SNAPSHOT DE FORMAÇÃO
# =============================================================================

def formation_market_snapshot(
    df: pd.DataFrame,
    context: PITContext,
) -> pd.DataFrame:
    """
    Última cotação válida de cada ticker
    conhecida até market_cutoff.
    """

    result = market_history_pit(
        df,
        context,
    )

    result = filter_b3_shares(
        result
    )

    if result.empty:

        raise MarketDataError(
            "FAIL-SAFE: universo de ações vazio."
        )

    snapshot = market_snapshot(
        df=result,
        context=context,
        ticker_column="TICKER",
        date_column="DATA_PREGAO",
        dataset_name="COTAHIST",
    )

    snapshot = snapshot.rename(
        columns={
            "PREULT": "FORMATION_PRICE",
            "DATA_PREGAO": (
                "FORMATION_PRICE_DATE"
            ),
        }
    )

    return snapshot


# =============================================================================
# UNIVERSO DE MERCADO
# =============================================================================

def build_market_universe(
    df: pd.DataFrame,
    context: PITContext,
    *,
    min_sessions: Optional[int] = None,
) -> pd.DataFrame:
    """
    Constrói a camada de mercado do universo.

    Requisitos:
    - ação permitida;
    - mercado à vista;
    - preço válido;
    - número mínimo de sessões;
    - nenhum preço futuro.

    NÃO utiliza retorno futuro.
    """

    if min_sessions is None:
        min_sessions = (
            MIN_TRADING_SESSIONS
        )

    history = market_history_pit(
        df,
        context,
    )

    shares = filter_b3_shares(
        history
    )

    sessions = eligible_by_sessions(
        shares,
        context,
        min_sessions=min_sessions,
    )

    snapshot = formation_market_snapshot(
        shares,
        context,
    )

    universe = snapshot.merge(
        sessions,
        on="TICKER",
        how="left",
        validate="one_to_one",
    )

    universe[
        "TRADING_SESSIONS"
    ] = (
        universe[
            "TRADING_SESSIONS"
        ]
        .fillna(0)
        .astype(int)
    )

    universe[
        "SESSION_ELIGIBLE"
    ] = (
        universe[
            "SESSION_ELIGIBLE"
        ]
        .fillna(False)
        .astype(bool)
    )

    universe = universe.loc[
        universe[
            "SESSION_ELIGIBLE"
        ]
    ].copy()

    if universe.empty:

        raise MarketDataError(
            "FAIL-SAFE: nenhuma ação passou "
            "pelo filtro mínimo de sessões."
        )

    universe[
        "MARKET_DATA_OK"
    ] = True

    universe[
        "MARKET_CUTOFF"
    ] = context.market_cutoff

    universe[
        "FORMATION_DATE"
    ] = context.formation_date

    universe = universe.sort_values(
        "TICKER",
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    return universe


# =============================================================================
# AUDITORIA
# =============================================================================

def market_audit(
    universe: pd.DataFrame,
    context: PITContext,
) -> dict:

    required = {
        "TICKER",
        "FORMATION_PRICE",
        "FORMATION_PRICE_DATE",
        "TRADING_SESSIONS",
    }

    missing = (
        required
        - set(universe.columns)
    )

    if missing:

        raise MarketDataError(
            "FAIL-SAFE: universo sem colunas "
            f"obrigatórias: {sorted(missing)}"
        )

    dates = pd.to_datetime(
        universe[
            "FORMATION_PRICE_DATE"
        ],
        errors="coerce",
    )

    if dates.isna().any():

        raise MarketDataError(
            "FAIL-SAFE: data de preço inválida."
        )

    lookahead = (
        dates.dt.normalize()
        >
        context.market_cutoff
    )

    if lookahead.any():

        offenders = (
            universe.loc[
                lookahead,
                [
                    "TICKER",
                    "FORMATION_PRICE_DATE",
                ],
            ]
            .head(20)
            .to_dict(
                orient="records"
            )
        )

        raise MarketDataError(
            "FAIL-SAFE: LOOK-AHEAD DE MERCADO. "
            f"Exemplos={offenders}"
        )

    invalid_prices = (
        pd.to_numeric(
            universe[
                "FORMATION_PRICE"
            ],
            errors="coerce",
        )
        <= 0
    )

    if invalid_prices.any():

        raise MarketDataError(
            "FAIL-SAFE: preço de formação "
            "inválido no universo."
        )

    return {
        "status": "OK",
        "formation_date": str(
            context.formation_date.date()
        ),
        "market_cutoff": str(
            context.market_cutoff.date()
        ),
        "stocks": int(
            len(universe)
        ),
        "minimum_sessions": int(
            universe[
                "TRADING_SESSIONS"
            ].min()
        ),
        "maximum_sessions": int(
            universe[
                "TRADING_SESSIONS"
            ].max()
        ),
        "lookahead_rows": 0,
    }


# =============================================================================
# SALVAR UNIVERSO
# =============================================================================

def save_market_universe(
    universe: pd.DataFrame,
    context: PITContext,
) -> tuple[Path, Path]:

    date_tag = (
        context.formation_date
        .strftime("%Y%m%d")
    )

    csv_path = (
        MARKET_DIR
        / f"market_universe_{date_tag}.csv"
    )

    json_path = (
        MANIFEST_DIR
        / f"market_universe_{date_tag}_manifest.json"
    )

    audit = market_audit(
        universe,
        context,
    )

    universe.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "engine": "MARKET_DATA",
        "source": "B3_COTAHIST",
        "created_at_utc": (
            _utc_now_iso()
        ),
        **audit,
        "output": str(
            csv_path
        ),
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return (
        csv_path,
        json_path,
    )


# =============================================================================
# TESTE INTERNO
# =============================================================================

def _self_test():

    # Teste da regra de ticker.
    valid = [
        "PETR3",
        "PETR4",
        "UNIP5",
        "UNIP6",
    ]

    invalid = [
        "BOVA11",
        "HGLG11",
        "PETR3F",
        "ABEV3T",
        "",
    ]

    for ticker in valid:

        if not is_allowed_share_ticker(
            ticker
        ):

            raise MarketDataError(
                "SELF-TEST falhou para "
                f"ticker válido: {ticker}"
            )

    for ticker in invalid:

        if is_allowed_share_ticker(
            ticker
        ):

            raise MarketDataError(
                "SELF-TEST falhou para "
                f"ticker inválido: {ticker}"
            )

    return True


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("B3 STOCK SELECTOR — MARKET DATA ENGINE")
    print("=" * 72)

    _self_test()

    print("Self-test ticker: OK")
    print("Fonte principal: B3 COTAHIST")
    print("Mercado: à vista")
    print(
        "Classes permitidas:",
        ALLOWED_SHARE_SUFFIXES,
    )
    print(
        "Mínimo de sessões:",
        MIN_TRADING_SESSIONS,
    )
    print("Retorno futuro calculado: NÃO")
    print("Look-ahead permitido: NÃO")

    print("=" * 72)
