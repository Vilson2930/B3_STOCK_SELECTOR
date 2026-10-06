from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from data.cvm import CVMDataError, list_zip_files, read_csv_from_zip


class MarketCapError(RuntimeError):
    pass

class MarketCapDataError(MarketCapError):
    pass

class MarketCapPITError(MarketCapError):
    pass


@dataclass(frozen=True)
class MarketCapContext:
    formation_date: pd.Timestamp
    accounting_cutoff: pd.Timestamp

    def __post_init__(self):
        if self.accounting_cutoff > self.formation_date:
            raise MarketCapPITError("accounting_cutoff posterior à formation_date")


def _ts(x):
    v = pd.Timestamp(x)
    if pd.isna(v):
        raise MarketCapPITError(f"data inválida: {x}")
    return v.normalize()


def create_market_cap_context(formation_date, accounting_cutoff):
    return MarketCapContext(_ts(formation_date), _ts(accounting_cutoff))


def _ascii(x):
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip().upper()
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _key(x):
    return re.sub(r"[^A-Z0-9]+", "_", _ascii(x)).strip("_")


def _norm_cols(df):
    out = df.copy()
    out.columns = [_key(c) for c in out.columns]
    return out


def _first(df, names):
    cols = set(df.columns)
    for n in names:
        k = _key(n)
        if k in cols:
            return k
    return None


def _num(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    x = s.astype("string").str.strip().str.replace(r"\s+", "", regex=True)
    both = x.str.contains(",", na=False) & x.str.contains(".", regex=False, na=False)
    x.loc[both] = x.loc[both].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    comma = x.str.contains(",", na=False) & ~x.str.contains(".", regex=False, na=False)
    x.loc[comma] = x.loc[comma].str.replace(",", ".", regex=False)
    return pd.to_numeric(x, errors="coerce")


def _cnpj(x):
    d = re.sub(r"\D", "", str(x)) if x is not None and not pd.isna(x) else ""
    return d if len(d) == 14 else None


def _cvm(x):
    d = re.sub(r"\D", "", str(x)) if x is not None and not pd.isna(x) else ""
    return f"CVM:{int(d)}" if d else None


def _issuer_id(df):
    cnpj = _first(df, ("CNPJ_CIA", "CNPJ_COMPANHIA", "CNPJ_EMISSOR", "CNPJ"))
    cvm = _first(df, ("CD_CVM", "CODIGO_CVM", "COD_CVM"))
    if cnpj is None and cvm is None:
        raise MarketCapDataError("FRE sem CNPJ/código CVM reconhecido")
    out = pd.Series([None] * len(df), index=df.index, dtype="object")
    if cnpj is not None:
        out = df[cnpj].map(_cnpj)
    if cvm is not None:
        out = out.where(out.notna(), df[cvm].map(_cvm))
    return out


def _find_fre_file(year):
    names = list_zip_files("FRE", int(year))
    matches = [n for n in names if n.lower().endswith(".csv") and "fre_cia_capital_social_classe_acao" in n.lower()]
    matches = [n for n in matches if all(t not in n.lower() for t in ("aumento", "reducao", "desdobramento"))]
    if len(matches) != 1:
        raise MarketCapDataError(f"arquivo FRE capital por classe ambíguo/ausente em {year}: {matches}")
    return matches[0]


def _species(x):
    s = _ascii(x)
    if "ORDIN" in s or s == "ON":
        return "ON"
    if "PREFER" in s or s == "PN" or s.startswith("PN"):
        return "PN"
    return None


def _share_class(x):
    s = re.sub(r"[^A-Z0-9]", "", _ascii(x))
    mapping = {"A":"A", "CLASSEA":"A", "PNA":"A", "B":"B", "CLASSEB":"B", "PNB":"B",
               "C":"C", "CLASSEC":"C", "PNC":"C", "D":"D", "CLASSED":"D", "PND":"D"}
    return mapping.get(s)


def _security_class(species, share_class):
    sp = _species(species)
    cl = _share_class(share_class)
    if sp == "ON":
        return "ON"
    if sp == "PN":
        return f"PN{cl}" if cl else "PN"
    return None


DATE_COLS = ("DT_REFER", "DATA_REFERENCIA", "DT_REFERENCIA", "DATA_BASE", "DT_BASE")
RECEIPT_COLS = ("DT_RECEB", "DATA_RECEBIMENTO", "DT_RECEBIMENTO", "DATA_ENTREGA", "DT_ENTREGA")
VERSION_COLS = ("VERSAO", "VERSAO_DOCUMENTO", "VERSAO_FRE", "NUM_VERSAO", "NR_VERSAO")
SPECIES_COLS = ("ESPECIE", "ESPECIE_ACAO", "TP_ESPECIE", "TIPO_ESPECIE", "TIPO_ACAO")
CLASS_COLS = ("CLASSE", "CLASSE_ACAO", "TP_CLASSE", "TIPO_CLASSE")
QTY_COLS = ("QT_ACOES", "QT_ACAO", "QUANTIDADE_ACOES", "QUANTIDADE_ACAO", "QTD_ACOES", "QTD_ACAO", "QUANTIDADE")
TREASURY_COLS = ("QT_ACOES_TESOURARIA", "QT_ACAO_TESOURARIA", "QUANTIDADE_ACOES_TESOURARIA", "QTD_ACOES_TESOURARIA")
OUT_COLS = ("QT_ACOES_CIRCULACAO", "QT_ACAO_CIRCULACAO", "QUANTIDADE_ACOES_CIRCULACAO", "QTD_ACOES_CIRCULACAO")


def _read_fre_year(year, context):
    fn = _find_fre_file(year)
    df = _norm_cols(read_csv_from_zip("FRE", int(year), fn))
    df["ISSUER_ID"] = _issuer_id(df)
    ref = _first(df, DATE_COLS)
    rec = _first(df, RECEIPT_COLS)
    ver = _first(df, VERSION_COLS)
    sp = _first(df, SPECIES_COLS)
    cl = _first(df, CLASS_COLS)
    qty = _first(df, QTY_COLS)
    treasury = _first(df, TREASURY_COLS)
    outstanding = _first(df, OUT_COLS)
    if sp is None or (qty is None and outstanding is None):
        raise MarketCapDataError(f"schema FRE não reconhecido: {list(df.columns)}")
    if ref is None and rec is None:
        raise MarketCapPITError("FRE sem data temporal reconhecida")
    df["FRE_REFERENCE_DATE"] = pd.to_datetime(df[ref], errors="coerce", dayfirst=True) if ref else pd.NaT
    df["FRE_RECEIPT_DATE"] = pd.to_datetime(df[rec], errors="coerce", dayfirst=True) if rec else pd.NaT
    if rec:
        df["FRE_PIT_DATE"] = df["FRE_RECEIPT_DATE"]
        method = "RECEIPT_DATE"
    else:
        df["FRE_PIT_DATE"] = df["FRE_REFERENCE_DATE"]
        method = "REFERENCE_DATE_FALLBACK"
    df = df.loc[df["ISSUER_ID"].notna() & df["FRE_PIT_DATE"].notna() & (df["FRE_PIT_DATE"] <= context.accounting_cutoff)].copy()
    df["FRE_VERSION"] = _num(df[ver]) if ver else 0.0
    df["SECURITY_CLASS"] = [_security_class(r[sp], r[cl] if cl else None) for _, r in df.iterrows()]
    df["FRE_SHARES_ISSUED"] = _num(df[qty]) if qty else np.nan
    df["FRE_SHARES_TREASURY"] = _num(df[treasury]) if treasury else np.nan
    if outstanding:
        df["FRE_SHARES_OUTSTANDING"] = _num(df[outstanding])
    else:
        df["FRE_SHARES_OUTSTANDING"] = np.where(df["FRE_SHARES_ISSUED"].notna() & df["FRE_SHARES_TREASURY"].notna(), df["FRE_SHARES_ISSUED"] - df["FRE_SHARES_TREASURY"], np.nan)
    df.loc[df["FRE_SHARES_OUTSTANDING"] < 0, "FRE_SHARES_OUTSTANDING"] = np.nan
    df["FRE_PIT_METHOD"] = method
    df["FRE_SOURCE_YEAR"] = int(year)
    df["FRE_SOURCE_FILE"] = fn
    df = df.loc[df["SECURITY_CLASS"].notna()].copy()
    keep = ["ISSUER_ID","SECURITY_CLASS","FRE_SHARES_ISSUED","FRE_SHARES_TREASURY","FRE_SHARES_OUTSTANDING","FRE_REFERENCE_DATE","FRE_RECEIPT_DATE","FRE_PIT_DATE","FRE_PIT_METHOD","FRE_VERSION","FRE_SOURCE_YEAR","FRE_SOURCE_FILE"]
    if df.empty:
        return pd.DataFrame(columns=keep)
    df = df.sort_values(["ISSUER_ID","SECURITY_CLASS","FRE_PIT_DATE","FRE_VERSION"], kind="mergesort")
    return df.groupby(["ISSUER_ID","SECURITY_CLASS"], as_index=False, sort=True).tail(1)[keep].reset_index(drop=True)


def build_fre_share_classes(formation_date, accounting_cutoff, *, years: Optional[Iterable[int]] = None):
    context = create_market_cap_context(formation_date, accounting_cutoff)
    years = tuple(years) if years is not None else (context.formation_date.year - 1, context.formation_date.year)
    frames = []
    for year in sorted(set(map(int, years))):
        if year > context.formation_date.year:
            raise MarketCapPITError(f"ano futuro: {year}")
        try:
            f = _read_fre_year(year, context)
        except CVMDataError as exc:
            raise MarketCapDataError(f"falha FRE {year}: {exc}") from exc
        if not f.empty:
            frames.append(f)
    if not frames:
        raise MarketCapDataError("nenhuma observação FRE válida até o cutoff")
    out = pd.concat(frames, ignore_index=True).sort_values(["ISSUER_ID","SECURITY_CLASS","FRE_PIT_DATE","FRE_VERSION","FRE_SOURCE_YEAR"], kind="mergesort")
    out = out.groupby(["ISSUER_ID","SECURITY_CLASS"], as_index=False, sort=True).tail(1).reset_index(drop=True)
    if out.duplicated(["ISSUER_ID","SECURITY_CLASS"]).any():
        raise MarketCapDataError("duplicidade emissor/classe")
    return out


def _ticker_class(ticker, especie=None):
    spec = re.sub(r"[^A-Z0-9]", "", _ascii(especie))
    if "ORDIN" in spec or spec == "ON": return "ON"
    if "PNA" in spec: return "PNA"
    if "PNB" in spec: return "PNB"
    suffix = str(ticker).strip().upper()[-1:]
    if "PREFER" in spec or spec == "PN":
        return {"5":"PNA", "6":"PNB"}.get(suffix, "PN")
    return {"3":"ON", "4":"PN", "5":"PNA", "6":"PNB"}.get(suffix)


def prepare_security_prices(security_universe, context):
    req = {"ISSUER_ID","TICKER","FORMATION_PRICE","FORMATION_PRICE_DATE"}
    miss = req - set(security_universe.columns)
    if miss: raise MarketCapDataError(f"security_universe sem: {sorted(miss)}")
    df = security_universe.copy()
    df["FORMATION_PRICE"] = pd.to_numeric(df["FORMATION_PRICE"], errors="coerce")
    df["FORMATION_PRICE_DATE"] = pd.to_datetime(df["FORMATION_PRICE_DATE"], errors="coerce")
    df = df.loc[df["ISSUER_ID"].notna() & df["TICKER"].notna() & df["FORMATION_PRICE"].gt(0) & df["FORMATION_PRICE_DATE"].notna() & (df["FORMATION_PRICE_DATE"] <= context.formation_date)].copy()
    especie = "ESPECI" if "ESPECI" in df.columns else None
    df["SECURITY_CLASS"] = [_ticker_class(r["TICKER"], r[especie] if especie else None) for _, r in df.iterrows()]
    df = df.loc[df["SECURITY_CLASS"].notna()].copy()
    if df["TICKER"].duplicated().any(): raise MarketCapDataError("ticker duplicado no snapshot")
    return df


def _resolve(sec, fre):
    rows = []
    if fre.empty: return pd.DataFrame(), "FRE_CLASS_DATA_MISSING"
    for _, fr in fre.iterrows():
        cls = fr["SECURITY_CLASS"]
        cand = sec.loc[sec["SECURITY_CLASS"] == cls].copy()
        if cand.empty and cls == "PN":
            pn = sec.loc[sec["SECURITY_CLASS"].astype(str).str.startswith("PN")].copy()
            specific = fre.loc[fre["SECURITY_CLASS"].astype(str).str.match(r"^PN[A-Z]+$")]
            if len(pn) == 1 and specific.empty: cand = pn
        if len(cand) == 0: return pd.DataFrame(), f"NO_PRICE_FOR_CLASS:{cls}"
        if len(cand) > 1: return pd.DataFrame(), f"AMBIGUOUS_PRICE_FOR_CLASS:{cls}"
        shares = pd.to_numeric(pd.Series([fr["FRE_SHARES_OUTSTANDING"]]), errors="coerce").iloc[0]
        if pd.isna(shares) or shares < 0: return pd.DataFrame(), f"OUTSTANDING_SHARES_MISSING:{cls}"
        s = cand.iloc[0]
        rows.append({"ISSUER_ID":s["ISSUER_ID"],"TICKER":s["TICKER"],"SECURITY_CLASS":cls,"FORMATION_PRICE":float(s["FORMATION_PRICE"]),"FORMATION_PRICE_DATE":s["FORMATION_PRICE_DATE"],"SHARES_CLASS_OUTSTANDING":float(shares),"MARKET_VALUE_CLASS":float(s["FORMATION_PRICE"])*float(shares),"FRE_PIT_DATE":fr["FRE_PIT_DATE"],"FRE_PIT_METHOD":fr["FRE_PIT_METHOD"],"FRE_SOURCE_YEAR":fr["FRE_SOURCE_YEAR"],"FRE_SOURCE_FILE":fr["FRE_SOURCE_FILE"]})
    detail = pd.DataFrame(rows)
    if detail["TICKER"].duplicated().any(): return pd.DataFrame(), "TICKER_USED_BY_MULTIPLE_CLASSES"
    return detail, "OK"


def build_issuer_market_cap(security_universe, fre_share_classes, formation_date, accounting_cutoff):
    context = create_market_cap_context(formation_date, accounting_cutoff)
    sec = prepare_security_prices(security_universe, context)
    fre = fre_share_classes.copy()
    fre["FRE_PIT_DATE"] = pd.to_datetime(fre["FRE_PIT_DATE"], errors="coerce")
    if (fre["FRE_PIT_DATE"].dropna() > context.accounting_cutoff).any(): raise MarketCapPITError("FRE posterior ao cutoff")
    issuer_rows, details = [], []
    for issuer in sorted(sec["ISSUER_ID"].dropna().astype(str).unique()):
        sg = sec.loc[sec["ISSUER_ID"].astype(str) == issuer]
        fg = fre.loc[fre["ISSUER_ID"].astype(str) == issuer]
        detail, status = _resolve(sg, fg)
        cap = float(detail["MARKET_VALUE_CLASS"].sum()) if status == "OK" else np.nan
        valid = bool(status == "OK" and np.isfinite(cap) and cap > 0)
        if not valid and status == "OK": status, cap = "INVALID_MARKET_CAP", np.nan
        issuer_rows.append({"ISSUER_ID":issuer,"MARKET_CAP":cap,"MARKET_VALUE":cap,"MARKET_CAP_VALID":valid,"MARKET_CAP_STATUS":status,"MARKET_CAP_N_CLASSES":int(len(detail)) if valid else 0,"MARKET_CAP_FORMATION_DATE":context.formation_date,"MARKET_CAP_ACCOUNTING_CUTOFF":context.accounting_cutoff,"MARKET_CAP_SOURCE":"CVM_FRE_X_B3_COTAHIST","FUTURE_RETURN_USED_MARKET_CAP":False})
        if not detail.empty: details.append(detail)
    out = pd.DataFrame(issuer_rows).sort_values("ISSUER_ID").reset_index(drop=True)
    det = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return out, det


def merge_market_cap_into_fundamentals(fundamentals, issuer_market_cap):
    if fundamentals["ISSUER_ID"].duplicated().any() or issuer_market_cap["ISSUER_ID"].duplicated().any():
        raise MarketCapDataError("ISSUER_ID duplicado")
    cols = [c for c in issuer_market_cap.columns if c == "ISSUER_ID" or c.startswith("MARKET_") or c == "FUTURE_RETURN_USED_MARKET_CAP"]
    drop = [c for c in cols if c != "ISSUER_ID" and c in fundamentals.columns]
    return fundamentals.drop(columns=drop).merge(issuer_market_cap[cols], on="ISSUER_ID", how="left", validate="one_to_one")


def run_market_cap_bridge(security_universe, fundamentals, formation_date, accounting_cutoff):
    fre = build_fre_share_classes(formation_date, accounting_cutoff)
    issuer, detail = build_issuer_market_cap(security_universe, fre, formation_date, accounting_cutoff)
    enriched = merge_market_cap_into_fundamentals(fundamentals, issuer)
    return enriched, issuer, detail, fre


def audit_market_cap(issuer_market_cap):
    valid = issuer_market_cap["MARKET_CAP_VALID"].fillna(False).astype(bool)
    return {"n_issuers":int(len(issuer_market_cap)),"market_cap_valid":int(valid.sum()),"market_cap_invalid":int((~valid).sum()),"coverage":float(valid.mean()),"status_counts":issuer_market_cap["MARKET_CAP_STATUS"].fillna("MISSING").value_counts().to_dict(),"future_return_used":False}


def _self_test():
    sec = pd.DataFrame({"ISSUER_ID":["111","111"],"TICKER":["ABCD3","ABCD4"],"FORMATION_PRICE":[10.0,8.0],"FORMATION_PRICE_DATE":["2026-10-05","2026-10-05"],"ESPECI":["ON","PN"]})
    fre = pd.DataFrame({"ISSUER_ID":["111","111"],"SECURITY_CLASS":["ON","PN"],"FRE_SHARES_OUTSTANDING":[100.0,50.0],"FRE_PIT_DATE":["2026-09-30","2026-09-30"],"FRE_PIT_METHOD":["RECEIPT_DATE","RECEIPT_DATE"],"FRE_SOURCE_YEAR":[2026,2026],"FRE_SOURCE_FILE":["x.csv","x.csv"]})
    issuer, detail = build_issuer_market_cap(sec, fre, "2026-10-05", "2026-09-30")
    expected = 1400.0
    if float(issuer["MARKET_CAP"].iloc[0]) != expected or len(detail) != 2:
        raise MarketCapDataError("SELF-TEST falhou")
    print("MARKET CAP BRIDGE: OK")
    print("Future return: BLOQUEADO")


if __name__ == "__main__":
    _self_test()
