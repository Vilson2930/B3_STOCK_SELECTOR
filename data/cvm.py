# =============================================================================
# B3 STOCK SELECTOR
# Arquivo: data/cvm.py
#
# RESPONSABILIDADE:
# - Download de dados oficiais da CVM
# - Cache local
# - Leitura padronizada
# - DFP / ITR / FCA
#
# NÃO FAZ:
# - Ranking
# - Quality Score
# - Turnaround Score
# - Valuation
# - Retorno futuro
# - Seleção de ações
#
# PRINCÍPIOS:
# - Fonte oficial
# - Fail-safe
# - Reprodutibilidade
# - Nenhum dado futuro é utilizado silenciosamente
# =============================================================================

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


# =============================================================================
# IMPORTAÇÃO DA CONFIGURAÇÃO
# =============================================================================

try:
    from config import DATA_DIR
except ImportError as exc:
    raise RuntimeError(
        "FAIL-SAFE: não foi possível importar DATA_DIR de config.py."
    ) from exc


# =============================================================================
# DIRETÓRIOS
# =============================================================================

CVM_DIR = DATA_DIR / "cvm"
RAW_DIR = CVM_DIR / "raw"
MANIFEST_DIR = CVM_DIR / "manifests"

RAW_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURAÇÃO CVM
# =============================================================================

CVM_BASE_URL = (
    "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC"
)

SUPPORTED_DATASETS = {
    "DFP": {
        "folder": "DFP/DADOS",
        "filename": "dfp_cia_aberta_{year}.zip",
    },
    "ITR": {
        "folder": "ITR/DADOS",
        "filename": "itr_cia_aberta_{year}.zip",
    },
    "FCA": {
        "folder": "FCA/DADOS",
        "filename": "fca_cia_aberta_{year}.zip",
    },
}

REQUEST_TIMEOUT = 120
CHUNK_SIZE = 1024 * 1024

USER_AGENT = (
    "B3-STOCK-SELECTOR/0.1 "
    "(research; official CVM public data)"
)


# =============================================================================
# EXCEÇÕES
# =============================================================================

class CVMError(RuntimeError):
    """Erro controlado do módulo CVM."""


class CVMDownloadError(CVMError):
    """Erro no download."""


class CVMIntegrityError(CVMError):
    """Erro de integridade do arquivo."""


class CVMDataError(CVMError):
    """Erro na leitura/estrutura dos dados."""


# =============================================================================
# UTILIDADES
# =============================================================================

def _utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _sha256(path: Path) -> str:
    """
    Calcula SHA-256 do arquivo.
    """

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(CHUNK_SIZE),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _normalize_dataset(dataset: str) -> str:
    """
    Normaliza e valida o tipo de documento.
    """

    dataset = str(dataset).strip().upper()

    if dataset not in SUPPORTED_DATASETS:
        raise CVMDataError(
            f"Dataset não suportado: {dataset}. "
            f"Permitidos: {sorted(SUPPORTED_DATASETS)}"
        )

    return dataset


def _validate_year(year: int) -> int:
    """
    Validação básica do ano solicitado.
    """

    try:
        year = int(year)
    except Exception as exc:
        raise CVMDataError(
            f"Ano inválido: {year}"
        ) from exc

    current_year = datetime.now().year

    if year < 2000:
        raise CVMDataError(
            f"Ano fora do intervalo permitido: {year}"
        )

    if year > current_year:
        raise CVMDataError(
            "FAIL-SAFE: tentativa de solicitar "
            f"ano futuro ({year})."
        )

    return year


# =============================================================================
# URL
# =============================================================================

def build_url(
    dataset: str,
    year: int,
) -> str:
    """
    Constrói a URL oficial da CVM.
    """

    dataset = _normalize_dataset(dataset)
    year = _validate_year(year)

    config = SUPPORTED_DATASETS[dataset]

    filename = config["filename"].format(
        year=year
    )

    return (
        f"{CVM_BASE_URL}/"
        f"{config['folder']}/"
        f"{filename}"
    )


# =============================================================================
# CAMINHO LOCAL
# =============================================================================

def local_zip_path(
    dataset: str,
    year: int,
) -> Path:

    dataset = _normalize_dataset(dataset)
    year = _validate_year(year)

    filename = (
        SUPPORTED_DATASETS[dataset]["filename"]
        .format(year=year)
    )

    return RAW_DIR / filename


# =============================================================================
# VALIDAÇÃO ZIP
# =============================================================================

def validate_zip(path: Path) -> None:
    """
    Confirma que o arquivo existe e é um ZIP íntegro.
    """

    if not path.exists():
        raise CVMIntegrityError(
            f"Arquivo não encontrado: {path}"
        )

    if path.stat().st_size == 0:
        raise CVMIntegrityError(
            f"Arquivo vazio: {path}"
        )

    if not zipfile.is_zipfile(path):
        raise CVMIntegrityError(
            f"Arquivo não é ZIP válido: {path}"
        )

    try:
        with zipfile.ZipFile(path, "r") as archive:

            bad_file = archive.testzip()

            if bad_file is not None:
                raise CVMIntegrityError(
                    "ZIP corrompido. "
                    f"Primeiro arquivo defeituoso: {bad_file}"
                )

            if not archive.namelist():
                raise CVMIntegrityError(
                    f"ZIP sem conteúdo: {path}"
                )

    except CVMIntegrityError:
        raise

    except Exception as exc:
        raise CVMIntegrityError(
            f"Erro ao validar ZIP: {path}"
        ) from exc


# =============================================================================
# MANIFESTO
# =============================================================================

def _save_manifest(
    dataset: str,
    year: int,
    path: Path,
    source_url: str,
) -> Path:
    """
    Salva metadados necessários para auditoria e reprodutibilidade.
    """

    manifest = {
        "dataset": dataset,
        "year": year,
        "source": "CVM",
        "source_url": source_url,
        "local_file": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "downloaded_at_utc": _utc_now_iso(),
    }

    manifest_path = (
        MANIFEST_DIR
        / f"{dataset.lower()}_{year}_manifest.json"
    )

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

    return manifest_path


# =============================================================================
# DOWNLOAD
# =============================================================================

def download_dataset(
    dataset: str,
    year: int,
    force: bool = False,
) -> Path:
    """
    Baixa DFP, ITR ou FCA diretamente da CVM.

    Se o arquivo já existir e estiver íntegro,
    utiliza o cache local.
    """

    dataset = _normalize_dataset(dataset)
    year = _validate_year(year)

    destination = local_zip_path(
        dataset,
        year,
    )

    url = build_url(
        dataset,
        year,
    )

    # -------------------------------------------------------------------------
    # CACHE
    # -------------------------------------------------------------------------

    if destination.exists() and not force:

        try:
            validate_zip(destination)

            return destination

        except CVMIntegrityError:
            # Arquivo existente, porém inválido.
            destination.unlink(
                missing_ok=True
            )

    # -------------------------------------------------------------------------
    # DOWNLOAD TEMPORÁRIO
    # -------------------------------------------------------------------------

    temp_path = destination.with_suffix(
        ".zip.part"
    )

    temp_path.unlink(
        missing_ok=True
    )

    headers = {
        "User-Agent": USER_AGENT
    }

    try:

        with requests.get(
            url,
            stream=True,
            timeout=REQUEST_TIMEOUT,
            headers=headers,
        ) as response:

            if response.status_code != 200:
                raise CVMDownloadError(
                    "Falha no download CVM. "
                    f"HTTP {response.status_code}: {url}"
                )

            with temp_path.open("wb") as file:

                for chunk in response.iter_content(
                    chunk_size=CHUNK_SIZE
                ):

                    if chunk:
                        file.write(chunk)

    except CVMError:
        temp_path.unlink(
            missing_ok=True
        )
        raise

    except Exception as exc:

        temp_path.unlink(
            missing_ok=True
        )

        raise CVMDownloadError(
            f"Erro ao baixar {dataset} {year}: {exc}"
        ) from exc

    # -------------------------------------------------------------------------
    # INTEGRIDADE
    # -------------------------------------------------------------------------

    try:
        validate_zip(temp_path)

    except Exception:
        temp_path.unlink(
            missing_ok=True
        )
        raise

    temp_path.replace(
        destination
    )

    validate_zip(
        destination
    )

    _save_manifest(
        dataset=dataset,
        year=year,
        path=destination,
        source_url=url,
    )

    return destination


# =============================================================================
# LISTAGEM DO CONTEÚDO
# =============================================================================

def list_zip_files(
    dataset: str,
    year: int,
    auto_download: bool = True,
) -> list[str]:
    """
    Lista arquivos existentes dentro do ZIP.
    """

    path = local_zip_path(
        dataset,
        year,
    )

    if not path.exists():

        if not auto_download:
            raise CVMDataError(
                f"Arquivo não encontrado: {path}"
            )

        path = download_dataset(
            dataset,
            year,
        )

    validate_zip(path)

    with zipfile.ZipFile(
        path,
        "r",
    ) as archive:

        return sorted(
            archive.namelist()
        )


# =============================================================================
# BUSCA DE ARQUIVOS INTERNOS
# =============================================================================

def find_internal_files(
    dataset: str,
    year: int,
    contains: str | Iterable[str] | None = None,
) -> list[str]:
    """
    Procura arquivos dentro do ZIP.

    Exemplo:
        contains="DRE_con"
    """

    names = list_zip_files(
        dataset,
        year,
    )

    if contains is None:
        return names

    if isinstance(
        contains,
        str,
    ):
        terms = [contains]

    else:
        terms = list(contains)

    terms = [
        str(term).lower()
        for term in terms
    ]

    matches = []

    for name in names:

        name_lower = name.lower()

        if all(
            term in name_lower
            for term in terms
        ):
            matches.append(name)

    return matches


# =============================================================================
# LEITURA DE CSV INTERNO
# =============================================================================

def read_csv_from_zip(
    dataset: str,
    year: int,
    internal_file: str,
    *,
    encoding: str = "latin1",
    sep: str = ";",
    decimal: str = ",",
    low_memory: bool = False,
) -> pd.DataFrame:
    """
    Lê um CSV diretamente de dentro do ZIP da CVM.
    """

    path = local_zip_path(
        dataset,
        year,
    )

    if not path.exists():
        path = download_dataset(
            dataset,
            year,
        )

    validate_zip(path)

    with zipfile.ZipFile(
        path,
        "r",
    ) as archive:

        names = archive.namelist()

        if internal_file not in names:
            raise CVMDataError(
                f"Arquivo interno não encontrado: {internal_file}"
            )

        try:

            raw = archive.read(
                internal_file
            )

            dataframe = pd.read_csv(
                io.BytesIO(raw),
                sep=sep,
                encoding=encoding,
                decimal=decimal,
                low_memory=low_memory,
            )

        except Exception as exc:
            raise CVMDataError(
                "Falha ao interpretar CSV da CVM: "
                f"{internal_file}"
            ) from exc

    if dataframe.empty:
        raise CVMDataError(
            f"CSV vazio: {internal_file}"
        )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    return dataframe


# =============================================================================
# LEITURA POR PADRÃO
# =============================================================================

def read_matching_csv(
    dataset: str,
    year: int,
    contains: str | Iterable[str],
    *,
    require_single_match: bool = True,
) -> pd.DataFrame:
    """
    Localiza e lê um CSV por parte do nome.

    O modo padrão exige exatamente uma correspondência,
    evitando que o robô escolha silenciosamente o arquivo errado.
    """

    matches = find_internal_files(
        dataset=dataset,
        year=year,
        contains=contains,
    )

    if not matches:
        raise CVMDataError(
            "Nenhum arquivo correspondente encontrado. "
            f"Dataset={dataset}, ano={year}, busca={contains}"
        )

    if (
        require_single_match
        and len(matches) != 1
    ):
        raise CVMDataError(
            "FAIL-SAFE: mais de um arquivo corresponde à busca. "
            f"Encontrados: {matches}"
        )

    return read_csv_from_zip(
        dataset=dataset,
        year=year,
        internal_file=matches[0],
    )


# =============================================================================
# DOWNLOAD DE UMA JANELA DE ANOS
# =============================================================================

def download_years(
    dataset: str,
    years: Iterable[int],
    force: bool = False,
) -> dict[int, Path]:
    """
    Baixa uma sequência de anos.

    Fail-safe:
    se qualquer ano falhar, a função interrompe.
    """

    dataset = _normalize_dataset(
        dataset
    )

    results = {}

    for year in years:

        year = _validate_year(
            year
        )

        results[year] = download_dataset(
            dataset=dataset,
            year=year,
            force=force,
        )

    return results


# =============================================================================
# STATUS LOCAL
# =============================================================================

def local_status() -> pd.DataFrame:
    """
    Inventário dos arquivos CVM já armazenados localmente.
    """

    rows = []

    for dataset in SUPPORTED_DATASETS:

        pattern = (
            SUPPORTED_DATASETS[dataset]["filename"]
            .replace("{year}", "*")
        )

        for path in sorted(
            RAW_DIR.glob(pattern)
        ):

            try:
                validate_zip(path)

                status = "OK"
                sha256 = _sha256(path)

            except Exception:
                status = "INVALID"
                sha256 = None

            year_match = "".join(
                character
                for character in path.stem
                if character.isdigit()
            )

            rows.append(
                {
                    "dataset": dataset,
                    "year": (
                        int(year_match[-4:])
                        if len(year_match) >= 4
                        else None
                    ),
                    "file": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256,
                    "status": status,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# TESTE DO MÓDULO
# =============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("B3 STOCK SELECTOR — CVM DATA ENGINE")
    print("=" * 72)

    print(f"Diretório CVM: {CVM_DIR}")
    print(f"Diretório RAW: {RAW_DIR}")

    status = local_status()

    if status.empty:
        print(
            "\nNenhum arquivo CVM armazenado localmente."
        )

    else:
        print(
            "\nArquivos CVM disponíveis:"
        )

        print(
            status[
                [
                    "dataset",
                    "year",
                    "status",
                    "size_bytes",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print("Módulo CVM: OK")
    print("=" * 72)
