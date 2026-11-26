"""Combining evidence from the main index, domain corpora, and MCP sources.
"""

from jarvisrag.text.query import _source_label_from_evidence_line


# Default global ceiling on the evidence lines sent to synthesis. The three
# retrieval tiers (MCP, domain, main index) each have their own per-source cap;
# without a global ceiling they SUM to a diluting pile (~70 lines) that buries
# the answer and slows synthesis. This bounds the total. Overridable.
_DEFAULT_SYNTHESIS_EVIDENCE_BUDGET = 40


# When a domain routed, the general (main) index is usually noise for that
# domain-specific question, so its contribution is reserved to a small share.
_DEFAULT_MAIN_INDEX_RESERVE = 6


def _evidence_source_labels(evidence_lines: list[str]) -> list[str]:
    """The source label of each evidence line: the connector label of an MCP
    line ("[MCP · <label> · ...]"), else the file label before the colon."""
    labels = []
    for line in evidence_lines or ():
        text = str(line)
        if text.startswith("[MCP ·"):
            labels.append(text[len("[MCP ·"):].split("·", 1)[0].strip())
        else:
            labels.append(_source_label_from_evidence_line(text))
    return labels


def _budget_combined_evidence(
    mcp_lines: list[str],
    domain_lines: list[str],
    main_lines: list[str],
    *,
    budget: int,
    main_reserve: int,
) -> list[str]:
    """Combine the three evidence tiers under a single global ceiling.

    Precedence MCP > domain > main is preserved. The AUTHORITATIVE tiers
    (MCP + domain) are NEVER truncated -- only the general main-index tail is
    trimmed to fit ``budget`` -- so a value the domain retriever ranked deep in
    its own (fully retained) list can never be cut here. When a domain
    contributed (``domain_lines`` non-empty), the main index is first reserved to
    ``main_reserve`` lines because it is usually noise for a domain-specific
    question. Pure and DOMAIN-AGNOSTIC: no query- or corpus-specific logic, just
    precedence-ordered list assembly with a ceiling.
    """
    authoritative = list(mcp_lines) + list(domain_lines)
    if domain_lines:
        main_lines = main_lines[: max(0, main_reserve)]
    remaining = max(0, budget - len(authoritative))
    return authoritative + main_lines[:remaining]
