from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from latin_to_javanese import (
    TransliterationResult,
    choose_backend,
    normalize_latin_javanese,
    transliterate_lines,
)
from tokenizer_jawa import compute_text_metrics, filter_to_javanese_text, normalize_text

LEIPZIG_DOWNLOAD_PAGE = "https://wortschatz-leipzig.de/en/download/jav"
LEIPZIG_WIKIPEDIA_CORPUS_ID = "jav_wikipedia_2021"
LEIPZIG_COMMUNITY_CORPUS_ID = "jav_community_2017"
UD_JV_CSUI_RAW_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/UD_Javanese-CSUI/master/"
    "jv_csui-ud-test.conllu"
)
DEFAULT_WIKISOURCE_API = "https://jv.wikisource.org/w/api.php"
DEFAULT_WIKISOURCE_CATEGORIES = ["Kategori:Wewaton", "Kategori:Sastra Jawa"]
LATIN_SIGNAL_RE = re.compile(r"[aiueoéèêx]", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NOISE_LINE_RE = re.compile(
    r"^(?:indhèks|indeks|pdf|image|gambar|kategori|category|dijupuk saka|retrieved from|"
    r"besut|sunting|edit|undhuh|download|contents?)$",
    re.IGNORECASE,
)
ARCHIVE_CANDIDATE_EXTS = (".tar.gz", ".tgz", ".zip", ".gz", ".bz2", ".xz")


@dataclass(slots=True)
class SourceSentence:
    source: str
    document: str
    script: str  # latin | javanese
    text: str
    record_id: str


@dataclass(slots=True)
class SourcePaths:
    hanacaraka: Path
    latin_clean: Path | None


class BuildError(RuntimeError):
    pass


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "hanacaraka-nanogpt-corpus-builder/0.1 (+https://github.com/openai)",
            "Accept": "text/html,application/json,text/plain,*/*",
        }
    )
    return session


def fetch_text(session: requests.Session, url: str, *, timeout: int = 60) -> str:
    resp = session.get(url, timeout=timeout)
    if resp.status_code >= 400:
        raise BuildError(f"GET {url} failed with status {resp.status_code}")
    return resp.text


def fetch_json(session: requests.Session, url: str, *, params: dict[str, object], timeout: int = 60) -> dict:
    resp = session.get(url, params=params, timeout=timeout)
    if resp.status_code >= 400:
        raise BuildError(f"GET {resp.url} failed with status {resp.status_code}")
    try:
        return resp.json()
    except Exception as exc:  # pragma: no cover - defensive
        raise BuildError(f"Failed to decode JSON from {resp.url}: {exc}") from exc


def download_file(session: requests.Session, url: str, target_path: Path, *, timeout: int = 120) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=timeout) as resp:
        if resp.status_code >= 400:
            raise BuildError(f"Download failed for {url} with status {resp.status_code}")
        with target_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return target_path


def collapse_spaces(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def normalize_hanacaraka_line(text: str) -> str:
    text = filter_to_javanese_text(text, keep_whitespace=True, keep_ascii_punct=False, keep_non_javanese=False)
    text = collapse_spaces(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dedupe_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_latin_sentence(text: str, *, min_letters: int = 4) -> str:
    text = normalize_latin_javanese(text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*(?:doi|isbn|http|www\.)[^)]*\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^0-9a-zà-öø-ÿêéè\-\s']", " ", text)
    text = re.sub(r"-{2,}", "-", text)
    text = re.sub(r"\s+", " ", text).strip(" -'\t\n")
    if not text:
        return ""
    letters = sum(ch.isalpha() for ch in text)
    if letters < min_letters:
        return ""
    if not LATIN_SIGNAL_RE.search(text):
        return ""
    digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    if digit_ratio > 0.35:
        return ""
    return text


def clean_native_javanese_line(text: str, *, min_chars: int = 6) -> str:
    text = collapse_spaces(text)
    if not text or NOISE_LINE_RE.match(text):
        return ""
    text = normalize_hanacaraka_line(text)
    nonspace = [ch for ch in text if not ch.isspace()]
    if len(nonspace) < min_chars:
        return ""
    javanese_ratio = sum(0xA980 <= ord(ch) <= 0xA9DF for ch in nonspace) / max(len(nonspace), 1)
    if javanese_ratio < 0.75:
        return ""
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Hanacaraka-only corpus from Leipzig/UD/Wikisource sources.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory where the corpus and metadata are written.")
    parser.add_argument(
        "--backend",
        choices=["auto", "carakanjs", "builtin"],
        default="auto",
        help="Latin->Hanacaraka transliteration backend. auto prefers carakanjs when installed.",
    )
    parser.add_argument(
        "--plain_e_mode",
        choices=["pepet", "taling"],
        default="pepet",
        help="Fallback builtin transliterator policy for plain 'e'.",
    )
    parser.add_argument("--node_bin", type=str, default="node", help="Node executable to use for the carakanjs backend.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for transliterating Latin lines.")
    parser.add_argument("--timeout", type=int, default=90, help="HTTP timeout in seconds.")

    parser.add_argument("--skip_leipzig_wikipedia", action="store_true", help="Skip Leipzig Javanese Wikipedia 2021.")
    parser.add_argument("--skip_leipzig_community", action="store_true", help="Skip Leipzig Javanese Community 2017.")
    parser.add_argument("--skip_ud", action="store_true", help="Skip UD Javanese-CSUI.")
    parser.add_argument("--use_wikisource", action="store_true", help="Also fetch native-script pages from Javanese Wikisource.")
    parser.add_argument(
        "--wikisource_category",
        action="append",
        default=[],
        help="Javanese Wikisource category title, e.g. 'Kategori:Wewaton'. Can be repeated.",
    )
    parser.add_argument(
        "--wikisource_page",
        action="append",
        default=[],
        help="Explicit Javanese Wikisource page title. Can be repeated.",
    )
    parser.add_argument(
        "--wikisource_api",
        type=str,
        default=DEFAULT_WIKISOURCE_API,
        help="MediaWiki API endpoint for Javanese Wikisource.",
    )
    parser.add_argument("--wikisource_max_pages", type=int, default=40, help="Maximum pages to fetch from categories in total.")

    parser.add_argument(
        "--leipzig_wikipedia_archive",
        type=Path,
        help="Use a pre-downloaded Leipzig archive for jav_wikipedia_2021 instead of auto-discovery/download.",
    )
    parser.add_argument(
        "--leipzig_community_archive",
        type=Path,
        help="Use a pre-downloaded Leipzig archive for jav_community_2017 instead of auto-discovery/download.",
    )
    parser.add_argument("--leipzig_wikipedia_url", type=str, help="Manual archive URL for jav_wikipedia_2021.")
    parser.add_argument("--leipzig_community_url", type=str, help="Manual archive URL for jav_community_2017.")
    parser.add_argument(
        "--ud_local_file",
        type=Path,
        help="Use a local jv_csui-ud-test.conllu file instead of downloading from GitHub.",
    )

    parser.add_argument("--write_manifest", action="store_true", help="Write manifest.jsonl with per-line provenance.")
    parser.add_argument(
        "--prepare_after_build",
        action="store_true",
        help="Run prepare.py on the final combined Hanacaraka corpus after building.",
    )
    parser.add_argument("--prepared_dir", type=Path, help="Output directory for prepare.py when --prepare_after_build is used.")
    parser.add_argument("--prepared_tokenizer", choices=["syllable_bpe", "syllable", "char"], default="syllable_bpe")
    parser.add_argument("--prepared_val_frac", type=float, default=0.1)
    parser.add_argument("--prepared_bpe_target_vocab_size", type=int, default=4096)
    parser.add_argument("--prepared_bpe_min_pair_freq", type=int, default=2)
    parser.add_argument("--prepared_bpe_max_merges", type=int, default=None)
    return parser.parse_args()


def iter_archive_sentences(path: Path) -> Iterator[str]:
    suffix = "".join(path.suffixes[-2:]).lower()
    if suffix in {".tar.gz", ".tgz", ".tar.bz2", ".tar.xz"} or path.suffix.lower() == ".tar":
        with tarfile.open(path, "r:*") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            members.sort(key=lambda m: ("sentences" not in m.name.lower(), len(m.name)))
            for member in members:
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                yield from _iter_lines_from_text_stream(io.TextIOWrapper(extracted, encoding="utf-8", errors="ignore"))
                return
        raise BuildError(f"No readable text file found inside archive: {path}")

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = [name for name in zf.namelist() if not name.endswith("/")]
            names.sort(key=lambda name: ("sentences" not in name.lower(), len(name)))
            for name in names:
                with zf.open(name) as f:
                    yield from _iter_lines_from_text_stream(io.TextIOWrapper(f, encoding="utf-8", errors="ignore"))
                return
        raise BuildError(f"No readable text file found inside zip archive: {path}")

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        yield from _iter_lines_from_text_stream(f)


def _iter_lines_from_text_stream(stream: Iterable[str]) -> Iterator[str]:
    for raw_line in stream:
        line = raw_line.strip()
        if not line:
            continue
        if "\t" in line:
            _, sentence = line.split("\t", 1)
            sentence = sentence.strip()
            if sentence:
                yield sentence
            continue
        if re.match(r"^\d+\s+", line):
            sentence = re.split(r"\s+", line, maxsplit=1)[1].strip()
            if sentence:
                yield sentence
            continue
        yield line


def discover_leipzig_download_url(
    session: requests.Session,
    *,
    corpus_id: str,
    download_page_url: str,
    manual_url: str | None,
    cache_dir: Path,
    timeout: int,
) -> str:
    if manual_url:
        return manual_url

    html = fetch_text(session, download_page_url, timeout=timeout)
    page_cache = cache_dir / "leipzig_download_page.html"
    page_cache.write_text(html, encoding="utf-8")

    candidates = _extract_link_candidates(html, download_page_url)
    if not candidates:
        raise BuildError(
            "Failed to discover any archive links from Leipzig download page. "
            f"Inspect {page_cache} or pass --{corpus_id}_url manually."
        )

    year = corpus_id.rsplit("_", 1)[-1]
    size_tokens = ["alle", "all", "100k", "100000", "1m", "1000000"]

    scored: list[tuple[int, str]] = []
    for url, context in candidates:
        haystack = f"{url} {context}".lower()
        score = 0
        if corpus_id.lower() in haystack:
            score += 50
        if year.lower() in haystack:
            score += 8
        if any(token in haystack for token in size_tokens):
            score += 12
        if any(ext in url.lower() for ext in ARCHIVE_CANDIDATE_EXTS):
            score += 10
        if "download" in url.lower() or "downloads." in url.lower():
            score += 3
        if score:
            scored.append((score, url))

    if not scored:
        raise BuildError(
            f"Could not match an archive URL for {corpus_id}. Inspect {page_cache} or pass a manual URL."
        )

    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def _extract_link_candidates(html: str, base_url: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    out: list[tuple[str, str]] = []

    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        full = urljoin(base_url, href)
        if full in seen:
            continue
        seen.add(full)
        context = " ".join(
            filter(
                None,
                [
                    a.get_text(" ", strip=True),
                    a.get("title") or "",
                    a.get("aria-label") or "",
                    a.get("onclick") or "",
                    str(a),
                ],
            )
        )
        out.append((full, context))

    # Also search raw HTML for direct archive URLs that may not appear as normal links.
    regexes = [
        r"https?://[^\s\"']+?(?:\.tar\.gz|\.tgz|\.zip|\.gz|\.bz2|\.xz)",
        r"/[^\s\"']+?(?:\.tar\.gz|\.tgz|\.zip|\.gz|\.bz2|\.xz)",
    ]
    for pattern in regexes:
        for match in re.finditer(pattern, html, flags=re.IGNORECASE):
            full = urljoin(base_url, match.group(0))
            if full in seen:
                continue
            seen.add(full)
            out.append((full, match.group(0)))
    return out


def fetch_leipzig_source(
    session: requests.Session,
    *,
    source_name: str,
    corpus_id: str,
    local_archive: Path | None,
    manual_url: str | None,
    downloads_dir: Path,
    timeout: int,
) -> Iterator[SourceSentence]:
    if local_archive is not None:
        archive_path = local_archive
    else:
        url = discover_leipzig_download_url(
            session,
            corpus_id=corpus_id,
            download_page_url=LEIPZIG_DOWNLOAD_PAGE,
            manual_url=manual_url,
            cache_dir=downloads_dir,
            timeout=timeout,
        )
        filename = Path(url).name or f"{corpus_id}.archive"
        archive_path = downloads_dir / filename
        if not archive_path.exists():
            download_file(session, url, archive_path, timeout=max(timeout, 120))

    for idx, sentence in enumerate(iter_archive_sentences(archive_path), start=1):
        yield SourceSentence(
            source=source_name,
            document=archive_path.name,
            script="latin",
            text=sentence,
            record_id=f"{source_name}:{idx}",
        )


def iter_conllu_texts(path: Path) -> Iterator[str]:
    current_text: str | None = None
    token_forms: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if current_text:
                    yield current_text
                elif token_forms:
                    yield _reconstruct_conllu_text(token_forms)
                current_text = None
                token_forms = []
                continue
            if line.startswith("# text = "):
                current_text = line[len("# text = ") :].strip()
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if not cols or len(cols) < 2:
                continue
            tok_id = cols[0]
            if "-" in tok_id or "." in tok_id:
                continue
            token_forms.append(cols[1])
    if current_text:
        yield current_text
    elif token_forms:
        yield _reconstruct_conllu_text(token_forms)


def _reconstruct_conllu_text(token_forms: Sequence[str]) -> str:
    text = " ".join(token_forms)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def fetch_ud_source(
    session: requests.Session,
    *,
    source_name: str,
    local_file: Path | None,
    downloads_dir: Path,
    timeout: int,
) -> Iterator[SourceSentence]:
    if local_file is not None:
        path = local_file
    else:
        path = downloads_dir / "jv_csui-ud-test.conllu"
        if not path.exists():
            download_file(session, UD_JV_CSUI_RAW_URL, path, timeout=max(timeout, 120))

    for idx, sentence in enumerate(iter_conllu_texts(path), start=1):
        yield SourceSentence(
            source=source_name,
            document=path.name,
            script="latin",
            text=sentence,
            record_id=f"{source_name}:{idx}",
        )


def iter_wikisource_titles(
    session: requests.Session,
    *,
    api_url: str,
    categories: Sequence[str],
    explicit_pages: Sequence[str],
    max_pages: int,
    timeout: int,
) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()

    for page in explicit_pages:
        if page not in seen:
            titles.append(page)
            seen.add(page)

    remaining = max(0, max_pages - len(titles))
    for category in categories:
        if remaining <= 0:
            break
        cmcontinue: str | None = None
        while remaining > 0:
            params: dict[str, object] = {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "list": "categorymembers",
                "cmtype": "page",
                "cmlimit": min(50, remaining),
                "cmtitle": category,
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            data = fetch_json(session, api_url, params=params, timeout=timeout)
            members = data.get("query", {}).get("categorymembers", [])
            if not members:
                break
            for item in members:
                title = item.get("title")
                if not title or title in seen:
                    continue
                titles.append(title)
                seen.add(title)
                remaining -= 1
                if remaining <= 0:
                    break
            cmcontinue = data.get("continue", {}).get("cmcontinue")
            if not cmcontinue:
                break
    return titles


def fetch_wikisource_page_lines(
    session: requests.Session,
    *,
    api_url: str,
    title: str,
    timeout: int,
) -> Iterator[str]:
    params = {
        "action": "parse",
        "format": "json",
        "formatversion": 2,
        "page": title,
        "prop": "text",
    }
    data = fetch_json(session, api_url, params=params, timeout=timeout)
    html_text = data.get("parse", {}).get("text", "")
    if not html_text:
        return

    soup = BeautifulSoup(html_text, "html.parser")
    for bad in soup.select(
        "table,script,style,sup.reference,ol.references,div.reflist,span.mw-editsection,nav,.navbox,.catlinks,.metadata"
    ):
        bad.decompose()

    text = soup.get_text("\n")
    for raw_line in text.splitlines():
        raw_line = collapse_spaces(raw_line)
        if not raw_line:
            continue
        if NOISE_LINE_RE.match(raw_line):
            continue
        yield raw_line


def fetch_wikisource_source(
    session: requests.Session,
    *,
    api_url: str,
    categories: Sequence[str],
    explicit_pages: Sequence[str],
    max_pages: int,
    timeout: int,
) -> Iterator[SourceSentence]:
    titles = iter_wikisource_titles(
        session,
        api_url=api_url,
        categories=categories,
        explicit_pages=explicit_pages,
        max_pages=max_pages,
        timeout=timeout,
    )
    for title in titles:
        for idx, line in enumerate(
            fetch_wikisource_page_lines(session, api_url=api_url, title=title, timeout=timeout),
            start=1,
        ):
            yield SourceSentence(
                source="wikisource_native",
                document=title,
                script="javanese",
                text=line,
                record_id=f"wikisource_native:{title}:{idx}",
            )


def open_source_outputs(out_dir: Path, source_names: Sequence[str]) -> dict[str, SourcePaths]:
    per_source_dir = out_dir / "per_source"
    per_source_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, SourcePaths] = {}
    for source_name in source_names:
        hanacaraka_path = per_source_dir / f"{source_name}.hanacaraka.txt"
        latin_clean_path = None if source_name == "wikisource_native" else per_source_dir / f"{source_name}.latin_clean.txt"
        out[source_name] = SourcePaths(hanacaraka=hanacaraka_path, latin_clean=latin_clean_path)
    return out


def ensure_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    out_dir = args.out_dir
    downloads_dir = out_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    combined_hanacaraka_path = out_dir / "corpus_hanacaraka.txt"
    combined_latin_clean_path = out_dir / "corpus_latin_clean.txt"
    stats_path = out_dir / "stats.json"

    session = build_session()
    backend_info: TransliterationResult = choose_backend(
        args.backend,
        repo_root=repo_root,
        node_bin=args.node_bin,
    )

    enabled_sources: list[str] = []
    if not args.skip_leipzig_wikipedia:
        enabled_sources.append("leipzig_wikipedia")
    if not args.skip_leipzig_community:
        enabled_sources.append("leipzig_community")
    if not args.skip_ud:
        enabled_sources.append("ud_jv_csui")
    if args.use_wikisource:
        enabled_sources.append("wikisource_native")
    if not enabled_sources:
        raise BuildError("No sources are enabled. Remove skip flags or pass --use_wikisource.")

    per_source_paths = open_source_outputs(out_dir, enabled_sources)
    ensure_file(combined_hanacaraka_path)
    ensure_file(combined_latin_clean_path)
    if args.write_manifest:
        ensure_file(manifest_path)

    source_iters: list[Iterator[SourceSentence]] = []
    if "leipzig_wikipedia" in enabled_sources:
        source_iters.append(
            fetch_leipzig_source(
                session,
                source_name="leipzig_wikipedia",
                corpus_id=LEIPZIG_WIKIPEDIA_CORPUS_ID,
                local_archive=args.leipzig_wikipedia_archive,
                manual_url=args.leipzig_wikipedia_url,
                downloads_dir=downloads_dir,
                timeout=args.timeout,
            )
        )
    if "leipzig_community" in enabled_sources:
        source_iters.append(
            fetch_leipzig_source(
                session,
                source_name="leipzig_community",
                corpus_id=LEIPZIG_COMMUNITY_CORPUS_ID,
                local_archive=args.leipzig_community_archive,
                manual_url=args.leipzig_community_url,
                downloads_dir=downloads_dir,
                timeout=args.timeout,
            )
        )
    if "ud_jv_csui" in enabled_sources:
        source_iters.append(
            fetch_ud_source(
                session,
                source_name="ud_jv_csui",
                local_file=args.ud_local_file,
                downloads_dir=downloads_dir,
                timeout=args.timeout,
            )
        )
    if "wikisource_native" in enabled_sources:
        categories = args.wikisource_category or DEFAULT_WIKISOURCE_CATEGORIES
        source_iters.append(
            fetch_wikisource_source(
                session,
                api_url=args.wikisource_api,
                categories=categories,
                explicit_pages=args.wikisource_page,
                max_pages=args.wikisource_max_pages,
                timeout=args.timeout,
            )
        )

    stats: dict[str, object] = {
        "backend": asdict(backend_info),
        "sources": defaultdict(lambda: defaultdict(int)),
        "prepare_after_build": None,
    }
    seen: set[str] = set()

    combined_hana = combined_hanacaraka_path.open("w", encoding="utf-8")
    combined_latin = combined_latin_clean_path.open("w", encoding="utf-8")
    manifest_fp = manifest_path.open("w", encoding="utf-8") if args.write_manifest else None
    per_source_hana_handles = {name: paths.hanacaraka.open("w", encoding="utf-8") for name, paths in per_source_paths.items()}
    per_source_latin_handles = {
        name: (paths.latin_clean.open("w", encoding="utf-8") if paths.latin_clean is not None else None)
        for name, paths in per_source_paths.items()
    }

    latin_batch: list[tuple[SourceSentence, str]] = []

    def write_record(record: SourceSentence, hanacaraka_line: str, latin_clean_line: str | None) -> None:
        source_stats = stats["sources"][record.source]
        key = dedupe_key(hanacaraka_line)
        if not key:
            source_stats["skipped_empty_after_clean"] += 1
            return
        if key in seen:
            source_stats["duplicate_lines"] += 1
            return
        seen.add(key)

        combined_hana.write(hanacaraka_line + "\n")
        per_source_hana_handles[record.source].write(hanacaraka_line + "\n")
        if latin_clean_line:
            combined_latin.write(latin_clean_line + "\n")
            handle = per_source_latin_handles.get(record.source)
            if handle is not None:
                handle.write(latin_clean_line + "\n")
        source_stats["written_lines"] += 1

        if manifest_fp is not None:
            manifest_fp.write(
                json.dumps(
                    {
                        "record_id": record.record_id,
                        "source": record.source,
                        "document": record.document,
                        "script": record.script,
                        "raw_text": record.text,
                        "latin_clean": latin_clean_line,
                        "hanacaraka": hanacaraka_line,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def flush_latin_batch() -> None:
        nonlocal latin_batch
        if not latin_batch:
            return
        clean_lines = [item[1] for item in latin_batch]
        converted = transliterate_lines(
            clean_lines,
            backend=backend_info.used_backend,
            repo_root=repo_root,
            plain_e_mode=args.plain_e_mode,
            node_bin=args.node_bin,
            keep_digits=True,
        )
        for (record, clean_line), raw_hana in zip(latin_batch, converted):
            hana_line = normalize_hanacaraka_line(raw_hana)
            write_record(record, hana_line, clean_line)
        latin_batch = []

    try:
        for iterator in source_iters:
            for record in iterator:
                source_stats = stats["sources"][record.source]
                source_stats["input_lines"] += 1
                if record.script == "latin":
                    clean_line = clean_latin_sentence(record.text)
                    if not clean_line:
                        source_stats["skipped_before_transliteration"] += 1
                        continue
                    latin_batch.append((record, clean_line))
                    if len(latin_batch) >= args.batch_size:
                        flush_latin_batch()
                    continue

                clean_native = clean_native_javanese_line(record.text)
                if not clean_native:
                    source_stats["skipped_before_write"] += 1
                    continue
                write_record(record, clean_native, None)

        flush_latin_batch()
    finally:
        combined_hana.close()
        combined_latin.close()
        if manifest_fp is not None:
            manifest_fp.close()
        for handle in per_source_hana_handles.values():
            handle.close()
        for handle in per_source_latin_handles.values():
            if handle is not None:
                handle.close()

    combined_text = combined_hanacaraka_path.read_text(encoding="utf-8")
    combined_metrics = compute_text_metrics(combined_text).to_dict()
    stats["combined"] = {
        "unique_lines": len(seen),
        "metrics": combined_metrics,
        "corpus_hanacaraka": str(combined_hanacaraka_path),
        "corpus_latin_clean": str(combined_latin_clean_path),
    }
    stats["sources"] = {source: dict(counter) for source, counter in stats["sources"].items()}

    if args.prepare_after_build:
        prepared_dir = args.prepared_dir or (out_dir / "prepared")
        cmd = [
            sys.executable,
            str(repo_root / "prepare.py"),
            "--input",
            str(combined_hanacaraka_path),
            "--out_dir",
            str(prepared_dir),
            "--tokenizer",
            args.prepared_tokenizer,
            "--val_frac",
            str(args.prepared_val_frac),
            "--write_clean_text",
        ]
        if args.prepared_tokenizer == "syllable_bpe":
            cmd.extend(
                [
                    "--bpe_target_vocab_size",
                    str(args.prepared_bpe_target_vocab_size),
                    "--bpe_min_pair_freq",
                    str(args.prepared_bpe_min_pair_freq),
                ]
            )
            if args.prepared_bpe_max_merges is not None:
                cmd.extend(["--bpe_max_merges", str(args.prepared_bpe_max_merges)])
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        stats["prepare_after_build"] = {
            "command": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "prepared_dir": str(prepared_dir),
        }
        if proc.returncode != 0:
            raise BuildError(
                "prepare.py failed after corpus build. Check stats.json for captured stdout/stderr."
            )

    write_json(stats_path, stats)
    print("Built Hanacaraka corpus")
    print(f"  out_dir              : {out_dir}")
    print(f"  backend requested    : {backend_info.requested_backend}")
    print(f"  backend used         : {backend_info.used_backend}")
    print(f"  corpus_hanacaraka    : {combined_hanacaraka_path}")
    print(f"  corpus_latin_clean   : {combined_latin_clean_path}")
    print(f"  unique_lines         : {len(seen)}")
    print(f"  invalid_token_rate   : {combined_metrics['invalid_token_rate']:.6f}")
    print(f"  orphan_mark_rate     : {combined_metrics['orphan_mark_rate']:.6f}")
    print(f"  dotted_circle_rate   : {combined_metrics['dotted_circle_rate']:.6f}")
    if args.prepare_after_build:
        print(f"  prepared_dir         : {args.prepared_dir or (out_dir / 'prepared')}")


if __name__ == "__main__":
    main()
