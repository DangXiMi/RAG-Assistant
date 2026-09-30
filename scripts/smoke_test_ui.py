"""Headlessly execute the Streamlit app to prove it renders without errors.

`streamlit.testing.v1.AppTest` runs the real script, so this catches runtime
errors that a syntax check cannot: bad widget arguments, unhandled exceptions
in the sidebar, or a broken backend call.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("BACKEND_URL", "http://127.0.0.1:8010")

# The app's title contains an emoji, which a Windows cp1252 console cannot
# encode. Force UTF-8 so reporting never crashes on the app's own content.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stdout
    pass

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py")

at = AppTest.from_file(APP, default_timeout=120)
at.run()

print("=" * 66)
print("STREAMLIT APP SMOKE TEST")
print("=" * 66)

errors = []
try:
    if at.exception:
        for exc in at.exception:
            errors.append(f"{exc.type}: {exc.message}")
except Exception as exc:  # pragma: no cover - harness difference
    errors.append(f"could not read exceptions: {exc}")

if errors:
    print("[XX] the app raised during render:")
    for err in errors:
        print("   ", err)
else:
    print("[ok] app rendered with no exceptions")

# Confirm the expected widgets exist.
def present(items):
    return [getattr(i, "label", None) or getattr(i, "value", None) for i in items]


print()
print("titles        :", present(at.title))
print("selectboxes   :", present(at.selectbox))
print("chat_input    :", len(at.chat_input))

if at.selectbox:
    options = at.selectbox[0].options
    print("mode options  :", options)
    if options != ["Hybrid", "HyDE", "Multi-Query", "Reranked"]:
        errors.append(f"unexpected retrieval modes: {options}")

# Collect text from every element type the app writes into.
text_parts = []
for collection in ("markdown", "caption", "info", "success", "warning", "error", "subheader"):
    for element in getattr(at, collection, []):
        value = getattr(element, "value", None)
        if isinstance(value, str):
            text_parts.append(value)
page_text = " ".join(text_parts)

print()
for needle in ("Backend", "Retrieval Strategy", "refund", "Connected"):
    found = needle.lower() in page_text.lower()
    print(f"page mentions {needle!r}: {found}")
    if needle in ("Backend", "Connected") and not found:
        errors.append(f"missing expected UI text: {needle}")

# Drive the actual chat flow: type the assignment's example question and let
# the app call the API. This is the real demo path.
print()
print("--- driving a chat question through the UI ---")
if not at.chat_input:
    errors.append("no chat input to drive")
else:
    at.chat_input[0].set_value("What is the refund policy?").run(timeout=600)

    if at.exception:
        for exc in at.exception:
            errors.append(f"chat run raised {exc.type}: {exc.message}")
    else:
        replies = [m.value for m in at.markdown]
        assistant = [value for value in replies if isinstance(value, str) and len(value) > 20]
        print(f"[ok] chat turn completed, {len(assistant)} markdown block(s) rendered")
        body = " ".join(assistant)
        if len(body) > 20:
            print("[ok] an answer was rendered")
        else:
            errors.append("answer text was too short to be useful")

        # The brief's expected output is an answer plus a source reference.
        # Requiring a particular phrase would make this flaky, since the model
        # may legitimately answer from a different retrieved chunk.
        if any(
            name in body
            for name in (
                "refund_policy.txt",
                "refund_matrix.docx",
                "service_levels.xlsx",
                "returns_faq.csv",
                "terms_and_conditions.pdf",
            )
        ):
            print("[ok] a source filename appears in the rendered output")
        else:
            errors.append("no source filename found in the rendered output")

        if "don't know" in body.lower() or "could not find" in body.lower():
            errors.append("the app refused to answer a question the corpus covers")

print()
print("RESULT:", "FAILED" if errors else "PASSED")
if errors:
    for err in errors:
        print("  -", err)
sys.exit(1 if errors else 0)
