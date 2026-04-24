"""Parsers for Sporza news: article list + article detail.

Body extraction strategy (in order):
1. ``__NEXT_DATA__`` JSON blob — the Next.js payload that contains the full
   article body even when the HTML is client-side rendered.
2. JSON-LD ``articleBody`` field.
3. HTML ``<article> <p>`` fallback for server-rendered pages.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

from bs4 import BeautifulSoup, Tag

from ..models import Image, NewsArticle
from ..utils import (
    clean_text,
    dedupe_preserve_order,
    normalise_url,
    parse_any_datetime,
    slugify,
    to_iso_string,
)

log = logging.getLogger(__name__)

# Section/landing paths that should NEVER be treated as article links.
_EXCLUDED_PATH_SEGMENTS = (
    "/video",
    "/audio",
    "/matchcenter",
    "/live",
    "/podcast",
    "/nieuwsbrief",
    "/over-sporza",
    "/contact",
    "/privacy",
    "/cookie",
    "/categorie",
    "/pas-verschenen",
    "/tag",
    "/auteur",
    "/programma",
    "/uitzending",
)

# Sporza article URLs contain a "~<digits>" ID segment.
_ARTICLE_ID_RE = re.compile(r"/~\d{4,}")

_SECTION_SLUGS = frozenset(
    {
        "voetbal", "wielrennen", "tennis", "basketbal", "volleybal",
        "hockey", "atletiek", "zwemmen", "formule-1", "motorsport",
        "rugby", "handbal", "golf", "darts", "snooker", "paardensport",
        "olympische-spelen", "jupiler-pro-league", "champions-league",
    }
)

# Keys that indicate a "body" field in a JSON object.
_BODY_KEYS = frozenset({
    "body", "bodytext", "bodyText", "content", "articleBody",
    "articleContent", "text", "richText", "richtext",
})

# Keys that indicate a "title" / "headline" field.
_TITLE_KEYS = frozenset({
    "title", "headline", "name", "titel", "heading",
})

# HTML-tag stripping for body text that arrives as HTML-in-string.
_HTML_TAG_RE = re.compile(r"<[^>]+>")


# ====================================================================
# Link discovery
# ====================================================================

def discover_news_links(html: str, base_url: str | None = None) -> list[str]:
    """Extract likely article URLs from an index/section page.

    Also looks inside ``__NEXT_DATA__`` for links that may not be present
    as regular ``<a>`` tags in the server-rendered HTML.
    """
    soup = BeautifulSoup(html, "lxml")
    candidates: list[str] = []

    # Strategy 1: Regular <a href> links.
    for a in soup.select("a[href]"):
        href = a.get("href")
        if not href or not isinstance(href, str):
            continue
        normalised = normalise_url(href, base=base_url or "https://sporza.be")
        if normalised and "sporza.be" in normalised and _looks_like_article(normalised):
            candidates.append(normalised)

    # Strategy 2: Links inside __NEXT_DATA__ blob.
    next_data = _extract_next_data_blob(soup)
    if next_data:
        for url_str in _find_article_urls_in_json(next_data):
            normalised = normalise_url(url_str, base=base_url or "https://sporza.be")
            if normalised and "sporza.be" in normalised and _looks_like_article(normalised):
                candidates.append(normalised)

    links = dedupe_preserve_order(candidates)
    if not links:
        log.warning("discover_news_links: found 0 article links in index page")
    return links


def _find_article_urls_in_json(obj: Any) -> Iterable[str]:
    """Recursively yield any string that looks like a Sporza article URL."""
    if isinstance(obj, str):
        if "~" in obj and ("sporza.be" in obj or obj.startswith("/")):
            if _ARTICLE_ID_RE.search(obj):
                yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _find_article_urls_in_json(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _find_article_urls_in_json(item)


def _looks_like_article(url: str) -> bool:
    lowered = url.lower()
    if "sporza.be" not in lowered:
        return False
    path = lowered.split("sporza.be", 1)[-1] or "/"
    if not path.startswith("/nl/"):
        return False
    stripped = path.rstrip("/")
    if stripped in ("", "/nl"):
        return False
    for seg in _EXCLUDED_PATH_SEGMENTS:
        if stripped == f"/nl{seg}" or stripped.startswith(f"/nl{seg}/"):
            return False
    parts = [p for p in stripped.split("/") if p]
    if len(parts) == 2 and parts[1] in _SECTION_SLUGS:
        return False
    return bool(_ARTICLE_ID_RE.search(stripped))


# ====================================================================
# Article detail parsing
# ====================================================================

def parse_news_article(html: str, url: str | None = None) -> NewsArticle | None:
    """Parse a Sporza article detail page into a ``NewsArticle``.

    Uses ``__NEXT_DATA__`` as the primary body source, then JSON-LD
    ``articleBody``, then HTML ``<p>`` as fallback.
    """
    soup = BeautifulSoup(html, "lxml")

    # --- Structured data sources ---
    ld_data = _extract_json_ld(soup)
    next_data = _extract_next_data_blob(soup)
    article_obj = _find_article_object(next_data) if next_data else {}

    if url is None:
        url = _extract_canonical(soup) or ""
    url = normalise_url(url) or ""

    title = (
        _str_or(article_obj, "title", "headline", "name")
        or _extract_title(soup, ld_data)
    )
    if not title and not ld_data and not article_obj:
        log.warning("parse_news_article: no title or structured data for %s", url)
        return None

    lead = (
        _str_or(article_obj, "lead", "intro", "subtitle", "description")
        or _extract_lead(soup)
    )
    authors = _extract_authors_combined(soup, ld_data, article_obj)
    published_at = (
        _extract_datetime_from_obj(article_obj, which="published")
        or _extract_datetime(soup, ld_data, which="published")
    )
    updated_at = (
        _extract_datetime_from_obj(article_obj, which="updated")
        or _extract_datetime(soup, ld_data, which="updated")
    )
    category = (
        _str_or(article_obj, "category", "section", "articleSection", "sport")
        or _extract_category(soup, ld_data)
    )
    tags = _extract_tags_combined(soup, article_obj)

    # --- Body extraction (most critical fix) ---
    body = _extract_body_from_next_data(article_obj)
    if not body:
        body = _extract_body_from_ld(ld_data)
    if not body:
        body = _extract_body_html(soup)

    images = _extract_images(soup, ld_data)
    related = _extract_related_links(soup, base=url)

    slug = _slug_from_url(url) or slugify(title or "untitled")

    if not body:
        log.warning("parse_news_article: empty body for %s", url)
    if not published_at:
        log.warning("parse_news_article: no publication date for %s", url)

    return NewsArticle(
        url=url,
        slug=slug,
        title=title,
        lead=lead,
        authors=authors,
        published_at=to_iso_string(published_at),
        updated_at=to_iso_string(updated_at),
        category=category,
        tags=tags,
        body_paragraphs=body,
        images=images,
        related_links=related,
    )


# ====================================================================
# __NEXT_DATA__ extraction (primary body source)
# ====================================================================

def _extract_next_data_blob(soup: BeautifulSoup) -> dict[str, Any] | None:
    """Parse the ``<script id="__NEXT_DATA__">`` tag into a dict."""
    script = soup.find("script", attrs={"id": "__NEXT_DATA__"})
    if not isinstance(script, Tag):
        return None
    try:
        return json.loads(script.string or "")
    except (TypeError, ValueError):
        log.debug("Could not parse __NEXT_DATA__ JSON")
        return None


def _find_article_object(blob: dict[str, Any]) -> dict[str, Any]:
    """Walk the __NEXT_DATA__ tree to find the article payload.

    Looks for objects containing both a title-like key and a body-like key.
    Falls back to the deepest object that has a body-like key.
    """
    best: dict[str, Any] = {}
    best_score = 0

    def _score(obj: dict[str, Any]) -> int:
        keys_lower = {k.lower() for k in obj.keys()}
        score = 0
        if keys_lower & {k.lower() for k in _BODY_KEYS}:
            score += 2
        if keys_lower & {k.lower() for k in _TITLE_KEYS}:
            score += 1
        if "author" in keys_lower or "authors" in keys_lower:
            score += 1
        if "datepublished" in keys_lower or "publishedat" in keys_lower or "publicationdate" in keys_lower:
            score += 1
        return score

    stack: list[Any] = [blob]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            s = _score(node)
            if s > best_score:
                best_score = s
                best = node
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)

    return best


def _extract_body_from_next_data(article_obj: dict[str, Any]) -> list[str]:
    """Extract body paragraphs from the article object found in __NEXT_DATA__."""
    if not article_obj:
        return []

    # Try every known body key.
    for key in article_obj:
        if key.lower() not in {k.lower() for k in _BODY_KEYS}:
            continue
        value = article_obj[key]
        paragraphs = _value_to_paragraphs(value)
        if paragraphs:
            return paragraphs

    # Fallback: walk all nested values for rich-text blocks.
    paragraphs = _walk_for_paragraphs(article_obj)
    if paragraphs:
        return paragraphs

    return []


def _value_to_paragraphs(value: Any) -> list[str]:
    """Convert a body value (string, HTML, list of blocks) to paragraphs."""
    if isinstance(value, str):
        return _string_to_paragraphs(value)
    if isinstance(value, list):
        return _block_list_to_paragraphs(value)
    if isinstance(value, dict):
        # Sometimes the body is wrapped: {"value": "...", "type": "rich-text"}
        for inner_key in ("value", "text", "content", "html", "raw"):
            if inner_key in value:
                result = _value_to_paragraphs(value[inner_key])
                if result:
                    return result
    return []


def _string_to_paragraphs(text: str) -> list[str]:
    """Convert a string (possibly HTML) to a list of clean paragraphs."""
    if not text or not text.strip():
        return []
    # If it looks like HTML, parse it.
    if "<p" in text or "<br" in text or "<div" in text:
        inner_soup = BeautifulSoup(text, "lxml")
        paragraphs: list[str] = []
        for p in inner_soup.find_all(["p", "div", "li"]):
            cleaned = clean_text(p.get_text(" ", strip=True))
            if cleaned and len(cleaned) > 10:
                paragraphs.append(cleaned)
        if paragraphs:
            return paragraphs
        # If <p> parsing found nothing, strip tags and split on newlines.
        stripped = _HTML_TAG_RE.sub("\n", text)
        return _split_plaintext(stripped)
    # Plain text.
    return _split_plaintext(text)


def _split_plaintext(text: str) -> list[str]:
    """Split plain text into paragraphs on double newlines or single newlines."""
    chunks = re.split(r"\n\s*\n|\n", text)
    paragraphs = [clean_text(c) for c in chunks if c]
    return [p for p in paragraphs if p and len(p) > 10]


def _block_list_to_paragraphs(blocks: list[Any]) -> list[str]:
    """Convert a list of rich-text block objects to paragraphs.

    Handles common CMS block formats:
    - [{"type": "paragraph", "children": [{"text": "..."}]}]
    - [{"type": "text", "value": "..."}]
    - [{"nodeType": "paragraph", "content": [{"value": "..."}]}]
    - Plain list of strings.
    """
    paragraphs: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            cleaned = clean_text(block)
            if cleaned and len(cleaned) > 10:
                paragraphs.append(cleaned)
            continue
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or block.get("nodeType") or "").lower()
        if block_type in ("image", "video", "embed", "advertisement", "ad"):
            continue
        # Extract text from the block.
        text_parts: list[str] = []
        _collect_text(block, text_parts)
        combined = clean_text(" ".join(text_parts))
        if combined and len(combined) > 10:
            paragraphs.append(combined)
    return paragraphs


def _collect_text(node: Any, out: list[str]) -> None:
    """Recursively collect text values from a nested object."""
    if isinstance(node, str):
        stripped = node.strip()
        if stripped:
            out.append(stripped)
        return
    if isinstance(node, dict):
        for key in ("text", "value", "content", "data"):
            if key in node:
                _collect_text(node[key], out)
        # Recurse into children / content arrays.
        for key in ("children", "content", "blocks", "items"):
            if key in node and isinstance(node[key], list):
                for child in node[key]:
                    _collect_text(child, out)
        return
    if isinstance(node, list):
        for item in node:
            _collect_text(item, out)


def _walk_for_paragraphs(obj: Any) -> list[str]:
    """Last-resort: walk the entire object for any text-rich arrays."""
    best: list[str] = []
    stack: list[Any] = [obj]
    while stack:
        node = stack.pop()
        if isinstance(node, list) and len(node) >= 2:
            candidate = _block_list_to_paragraphs(node)
            if len(candidate) > len(best):
                best = candidate
        if isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return best


# ====================================================================
# JSON-LD body extraction
# ====================================================================

def _extract_body_from_ld(ld: dict[str, Any]) -> list[str]:
    """Extract body from JSON-LD ``articleBody`` field."""
    body = ld.get("articleBody")
    if isinstance(body, str):
        paragraphs = _string_to_paragraphs(body)
        if paragraphs:
            return paragraphs
    return []


# ====================================================================
# HTML body extraction (fallback)
# ====================================================================

def _extract_body_html(soup: BeautifulSoup) -> list[str]:
    """Extract body paragraphs from server-rendered HTML."""
    paragraphs: list[str] = []
    roots: list[Tag] = []
    for selector in ("article", "main article", "[itemprop='articleBody']"):
        node = soup.select_one(selector)
        if isinstance(node, Tag):
            roots.append(node)
            break
    if not roots:
        roots = [soup]  # type: ignore[list-item]
    for root in roots:
        for p in root.find_all("p"):
            if _is_in_excluded_block(p):
                continue
            text = clean_text(p.get_text(" ", strip=True))
            if text and len(text) > 10:
                paragraphs.append(text)
    return paragraphs


# ====================================================================
# JSON-LD, metadata, and helper extractors (unchanged logic, tidied)
# ====================================================================

def _extract_json_ld(soup: BeautifulSoup) -> dict[str, Any]:
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = json.loads(script.string or "")
        except (TypeError, ValueError):
            continue
        for candidate in _iter_ld_objects(payload):
            if _is_article_ld(candidate):
                return candidate
    return {}


def _iter_ld_objects(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        yield payload
        graph = payload.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                if isinstance(item, dict):
                    yield item
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def _is_article_ld(obj: dict[str, Any]) -> bool:
    t = obj.get("@type")
    if isinstance(t, str):
        return t.lower().endswith("article")
    if isinstance(t, list):
        return any(isinstance(x, str) and x.lower().endswith("article") for x in t)
    return False


def _extract_canonical(soup: BeautifulSoup) -> str | None:
    link = soup.find("link", attrs={"rel": "canonical"})
    if isinstance(link, Tag):
        href = link.get("href")
        if isinstance(href, str):
            return href
    og = soup.find("meta", attrs={"property": "og:url"})
    if isinstance(og, Tag):
        content = og.get("content")
        if isinstance(content, str):
            return content
    return None


def _extract_title(soup: BeautifulSoup, ld: dict[str, Any]) -> str | None:
    headline = ld.get("headline")
    if isinstance(headline, str):
        text = clean_text(headline)
        if text:
            return text
    h1 = soup.find("h1")
    if isinstance(h1, Tag):
        return clean_text(h1.get_text(" ", strip=True))
    og = soup.find("meta", attrs={"property": "og:title"})
    if isinstance(og, Tag):
        content = og.get("content")
        if isinstance(content, str):
            return clean_text(content)
    return None


def _extract_lead(soup: BeautifulSoup) -> str | None:
    for selector in (
        "[data-testid='article-lead']",
        "article p.lead", "article .lead",
        "meta[name='description']",
        "meta[property='og:description']",
    ):
        node = soup.select_one(selector)
        if isinstance(node, Tag):
            if node.name == "meta":
                content = node.get("content")
                if isinstance(content, str):
                    text = clean_text(content)
                    if text:
                        return text
            else:
                text = clean_text(node.get_text(" ", strip=True))
                if text:
                    return text
    return None


def _extract_authors_combined(
    soup: BeautifulSoup, ld: dict[str, Any], article_obj: dict[str, Any]
) -> list[str]:
    authors: list[str] = []
    # From __NEXT_DATA__
    for key in ("author", "authors", "byline"):
        raw = article_obj.get(key)
        if isinstance(raw, str) and raw.strip():
            authors.append(raw.strip())
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    authors.append(item.strip())
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("displayName") or ""
                    if name.strip():
                        authors.append(name.strip())
        elif isinstance(raw, dict):
            name = raw.get("name") or raw.get("displayName") or ""
            if name.strip():
                authors.append(name.strip())
    # From JSON-LD
    author = ld.get("author")
    if isinstance(author, dict):
        name = author.get("name")
        if isinstance(name, str):
            authors.append(name)
    elif isinstance(author, list):
        for a in author:
            if isinstance(a, dict):
                name = a.get("name")
                if isinstance(name, str):
                    authors.append(name)
            elif isinstance(a, str):
                authors.append(a)
    # From HTML
    for node in soup.select("[rel='author'], .byline, .author"):
        text = clean_text(node.get_text(" ", strip=True))
        if text:
            authors.append(text)
    cleaned = [clean_text(a) or "" for a in authors]
    return dedupe_preserve_order([a for a in cleaned if a])


def _extract_datetime(soup: BeautifulSoup, ld: dict[str, Any], *, which: str) -> Any:
    ld_key = "datePublished" if which == "published" else "dateModified"
    ld_val = ld.get(ld_key)
    if isinstance(ld_val, str):
        parsed = parse_any_datetime(ld_val)
        if parsed:
            return parsed
    meta_name = (
        "article:published_time" if which == "published" else "article:modified_time"
    )
    meta = soup.find("meta", attrs={"property": meta_name})
    if isinstance(meta, Tag):
        content = meta.get("content")
        if isinstance(content, str):
            parsed = parse_any_datetime(content)
            if parsed:
                return parsed
    if which == "published":
        time_tag = soup.find("time")
        if isinstance(time_tag, Tag):
            attr = time_tag.get("datetime") or time_tag.get_text(" ", strip=True)
            if isinstance(attr, str):
                parsed = parse_any_datetime(attr)
                if parsed:
                    return parsed
    return None


def _extract_datetime_from_obj(obj: dict[str, Any], *, which: str) -> Any:
    """Extract datetime from __NEXT_DATA__ article object."""
    if which == "published":
        keys = ("datePublished", "publishedAt", "publicationDate", "date", "created")
    else:
        keys = ("dateModified", "updatedAt", "modifiedDate", "modified")
    for key in keys:
        val = obj.get(key)
        if isinstance(val, str):
            parsed = parse_any_datetime(val)
            if parsed:
                return parsed
    return None


def _extract_category(soup: BeautifulSoup, ld: dict[str, Any]) -> str | None:
    section = ld.get("articleSection")
    if isinstance(section, str):
        return clean_text(section)
    if isinstance(section, list) and section:
        first = section[0]
        if isinstance(first, str):
            return clean_text(first)
    meta = soup.find("meta", attrs={"property": "article:section"})
    if isinstance(meta, Tag):
        content = meta.get("content")
        if isinstance(content, str):
            return clean_text(content)
    return None


def _extract_tags_combined(soup: BeautifulSoup, article_obj: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    # From __NEXT_DATA__
    for key in ("tags", "keywords", "labels"):
        raw = article_obj.get(key)
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    cleaned = clean_text(item)
                    if cleaned:
                        tags.append(cleaned)
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("label") or item.get("title") or ""
                    cleaned = clean_text(name)
                    if cleaned:
                        tags.append(cleaned)
    # From HTML meta
    for meta in soup.find_all("meta", attrs={"property": "article:tag"}):
        content = meta.get("content") if isinstance(meta, Tag) else None
        if isinstance(content, str):
            cleaned = clean_text(content)
            if cleaned:
                tags.append(cleaned)
    return dedupe_preserve_order(tags)


def _extract_images(soup: BeautifulSoup, ld: dict[str, Any]) -> list[Image]:
    images: list[Image] = []
    seen: set[str] = set()

    def _add(url: str | None, alt: str | None) -> None:
        cleaned_url = normalise_url(url)
        if not cleaned_url or cleaned_url in seen:
            return
        seen.add(cleaned_url)
        images.append(Image(url=cleaned_url, alt=clean_text(alt)))

    ld_image = ld.get("image")
    if isinstance(ld_image, str):
        _add(ld_image, None)
    elif isinstance(ld_image, dict):
        _add(ld_image.get("url"), ld_image.get("caption"))
    elif isinstance(ld_image, list):
        for item in ld_image:
            if isinstance(item, str):
                _add(item, None)
            elif isinstance(item, dict):
                _add(item.get("url"), item.get("caption"))
    for img in soup.find_all("img"):
        if not isinstance(img, Tag):
            continue
        _add(img.get("src") or img.get("data-src"), img.get("alt"))
    return images


def _extract_related_links(soup: BeautifulSoup, base: str) -> list[str]:
    related: list[str] = []
    for selector in (".related a[href]", "aside a[href]", "[data-related] a[href]"):
        for a in soup.select(selector):
            href = a.get("href")
            if isinstance(href, str):
                normalised = normalise_url(href, base=base or "https://sporza.be")
                if normalised:
                    related.append(normalised)
    return dedupe_preserve_order(related)


def _is_in_excluded_block(node: Tag) -> bool:
    for parent in node.parents:
        if not isinstance(parent, Tag):
            continue
        classes = " ".join(parent.get("class") or []).lower()
        if any(x in classes for x in ("footer", "newsletter", "related", "share")):
            return True
        if parent.name in ("footer", "aside"):
            return True
    return False


def _slug_from_url(url: str) -> str | None:
    if not url:
        return None
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return slugify(tail) if tail else None


def _str_or(obj: dict[str, Any], *keys: str) -> str | None:
    """Return the first non-empty string value for any of the given keys."""
    for key in keys:
        val = obj.get(key)
        if isinstance(val, str):
            cleaned = clean_text(val)
            if cleaned:
                return cleaned
    return None
