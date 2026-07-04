"""Tiny terminal-UI helpers: Homebrew-style headers, colors, status emojis.

Colors auto-disable when stdout is not a TTY, when ``NO_COLOR`` is set, or when
``TERM=dumb`` — so piped and CI output stays clean. ``FORCE_COLOR`` overrides.
Emojis are printed regardless (they render fine when redirected to a file).
"""

import os
import sys


def _supports_color(stream=sys.stdout):
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    return os.environ.get("TERM") != "dumb"


_COLOR = _supports_color()


def _wrap(code):
    esc = f"\033[{code}m" if _COLOR else ""
    reset = "\033[0m" if _COLOR else ""
    return lambda s: f"{esc}{s}{reset}"


bold   = _wrap("1")
dim    = _wrap("2")
blue   = _wrap("34")
green  = _wrap("32")
red    = _wrap("31")
yellow = _wrap("33")
cyan   = _wrap("36")
grey   = _wrap("90")


def header(title, subtitle=None):
    """Homebrew-style ``==>`` section header (blue arrow, bold title)."""
    line = f"\n{blue(bold('==>'))} {bold(title)}"
    if subtitle:
        line += f"  {grey(subtitle)}"
    print(line, flush=True)


def step(script, args=None):
    """A command about to run (dim, indented)."""
    extra = f" {dim(' '.join(map(str, args)))}" if args else ""
    print(f"  {cyan('▸')} {script}{extra}", flush=True)


def ok(msg):
    print(f"  ✅ {msg}", flush=True)


def fail(msg):
    print(f"  ❌ {red(msg)}", flush=True)


def skipped(msg):
    print(f"  ⏭️  {yellow(msg)}", flush=True)


def info(msg):
    print(f"  {grey('·')} {grey(msg)}", flush=True)


def warn(msg):
    print(f"{yellow('⚠️  Warning:')} {msg}", flush=True)


def error(msg):
    print(f"{red('❌ Error:')} {msg}", flush=True)


def beer(msg):
    """Homebrew-style success finish."""
    print(f"\n🍺 {bold(msg)}", flush=True)
