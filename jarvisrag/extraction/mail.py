"""Apple Mail (.emlx) messages and their staged attachments.
"""

import html as html_lib
import os
import re

from llama_index.core.schema import Document

from jarvisrag.config import logger


class EmlxReader:
    """Parse an Apple Mail ``.emlx`` message into a single text Document.

    ``.emlx`` framing is an ASCII decimal byte-count line, the raw RFC-822
    message of exactly that many bytes, then an Apple metadata plist. Parse the
    message span with the stdlib ``email`` parser (same approach as
    ``MHTMLReader``), preferring the ``text/plain`` part and falling back to a
    tag-stripped ``text/html`` part. From/To/Cc/Date/Subject headers are kept as
    metadata so an answer can attribute a quote to a specific message."""

    _TAG_RE = re.compile(r"<[^>]+>")
    # <style>/<script> (and CSS/JS text between the tags) must be removed
    # wholesale: stripping only the tags would leave the rule/code text as body
    # content. DOTALL so a block spanning newlines is matched; the backreference
    # pairs each opening tag with its own closing tag.
    _STYLE_SCRIPT_RE = re.compile(
        r"<(style|script)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL
    )
    # HTML comments can also wrap content (e.g. conditional comments in mail).
    _COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
    # Zero-width / BOM characters that survive entity decoding and add no meaning.
    _ZERO_WIDTH_RE = re.compile(r"[\ufeff\u200b\u200c\u200d\u2060]")
    _WHITESPACE_RE = re.compile(r"\s+")

    @classmethod
    def _html_to_text(cls, html: str) -> str:
        """Reduce an HTML email body to readable text: drop <style>/<script>
        blocks (and their contents) and comments, strip remaining tags, decode
        entities, remove zero-width characters, and collapse whitespace."""
        without_blocks = cls._STYLE_SCRIPT_RE.sub(" ", html)
        without_comments = cls._COMMENT_RE.sub(" ", without_blocks)
        without_tags = cls._TAG_RE.sub(" ", without_comments)
        decoded = html_lib.unescape(without_tags)
        decoded = cls._ZERO_WIDTH_RE.sub("", decoded)
        return cls._WHITESPACE_RE.sub(" ", decoded).strip()

    @staticmethod
    def _message_bytes(raw: bytes) -> bytes:
        newline = raw.find(b"\n")
        if newline == -1:
            return raw
        try:
            count = int(raw[:newline].strip())
        except ValueError:
            return raw  # not framed as expected; parse the whole payload
        start = newline + 1
        return raw[start:start + count]

    @classmethod
    def _body_text(cls, message) -> str:
        plain = html = None
        for part in message.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain" and plain is None:
                try:
                    plain = part.get_content()
                except Exception:
                    plain = None
            elif ctype == "text/html" and html is None:
                try:
                    html = part.get_content()
                except Exception:
                    html = None
        if isinstance(plain, str) and plain.strip():
            return plain
        if isinstance(html, str) and html.strip():
            return cls._html_to_text(html)
        return ""

    def load_data(self, file_path, **kwargs):
        import email
        from email import policy

        file_path_str = str(file_path)
        file_name = os.path.basename(file_path_str)
        try:
            with open(file_path_str, "rb") as handle:
                raw = handle.read()
            message = email.message_from_bytes(self._message_bytes(raw), policy=policy.default)
        except Exception as e:
            logger.error(f"EmlxReader could not parse {file_name}: {e}")
            return []

        headers = {
            "email_from": str(message.get("From", "") or ""),
            "email_to": str(message.get("To", "") or ""),
            "email_cc": str(message.get("Cc", "") or ""),
            "email_date": str(message.get("Date", "") or ""),
            "email_subject": str(message.get("Subject", "") or ""),
        }
        body = self._body_text(message)
        header_block = "\n".join(
            f"{label}: {headers[key]}"
            for label, key in (
                ("From", "email_from"), ("To", "email_to"), ("Cc", "email_cc"),
                ("Date", "email_date"), ("Subject", "email_subject"),
            )
            if headers[key]
        )
        text = f"{header_block}\n\n{body}".strip()
        if not text:
            logger.info(f"EmlxReader found no readable content in {file_name}.")
            return []
        metadata = {"file_path": file_path_str, "file_name": file_name, "extracted_by": "emlx"}
        metadata.update({k: v for k, v in headers.items() if v})
        return [Document(text=text, metadata=metadata)]


def staged_email_attachments(staged_path: str) -> list[tuple[str, str]]:
    """Resolve a staged email's attachments to local files, on demand.

    The staged ``.emlx`` is the pointer: its MIME headers list each attachment
    (``Content-Disposition: attachment``; inline logos and tracking pixels are
    skipped) even when Apple Mail stored the bytes outside the message. External
    files come from the attachment mirror that mail refresh keeps beside the
    corpus; an attachment embedded in the message itself is written into that
    mirror the first time it is needed. Nothing is extracted or indexed here.
    Returns ``(attachment name, local path)`` pairs; unreadable input yields []."""
    import email
    from email import policy

    from jarvisrag.docsync.mail import mail_attachment_dir

    stem = os.path.basename(staged_path).split(".", 1)[0]
    mirror = os.path.join(mail_attachment_dir(os.path.dirname(os.path.abspath(staged_path))), stem)
    try:
        with open(staged_path, "rb") as handle:
            message = email.message_from_bytes(
                EmlxReader._message_bytes(handle.read()), policy=policy.default
            )
    except Exception:  # noqa: BLE001 - a bad pointer just yields no attachments
        return []

    mirrored: dict[str, str] = {}
    if os.path.isdir(mirror):
        for root, _dirs, files in os.walk(mirror):
            for fname in files:
                if not fname.startswith(".") and not fname.endswith(".mirror-tmp"):
                    mirrored.setdefault(fname.lower(), os.path.join(root, fname))

    found: list[tuple[str, str]] = []
    used: set[str] = set()
    for part in message.walk():
        if part.is_multipart() or part.get_content_disposition() != "attachment":
            continue
        name = os.path.basename(str(part.get_filename() or "").replace("\\", "/")).strip()
        if not name:
            continue
        try:
            payload = part.get_payload(decode=True) or b""
        except Exception:  # noqa: BLE001
            payload = b""
        path = None
        if payload:
            path = os.path.join(mirror, "embedded", name)
            try:
                if not (os.path.exists(path) and os.path.getsize(path) == len(payload)):
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as out:
                        out.write(payload)
            except OSError:
                path = None
        else:
            path = mirrored.get(name.lower())
        if path and path not in used:
            used.add(path)
            found.append((name, path))
    # Apple Mail may store a file under a slightly different name than the MIME
    # header; keep any mirrored file that no header matched rather than lose it.
    for path in mirrored.values():
        if path not in used:
            used.add(path)
            found.append((os.path.basename(path), path))
    return found
