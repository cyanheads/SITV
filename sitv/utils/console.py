"""
Console output utilities for SITV.

This module provides utilities for formatted console output including
banners, sections, and formatted values.
"""

# ============================================================================
# CONSOLE DISPLAY CONSTANTS
# ============================================================================
# These constants ensure consistent formatting across the entire application

# Banner and section formatting
BANNER_WIDTH = 70  # Standard width for banners and sections
BANNER_CHAR = "="  # Character for major section separators
SECTION_CHAR = "-"  # Character for subsection separators
SUBSECTION_CHAR = "─"  # Character for minor subsections (Unicode)

# Progress bar formatting
PROGRESS_BAR_WIDTH = 50  # Width of progress bar display


def print_banner(text: str, char: str | None = None, width: int | None = None) -> None:
    """Print a banner with centered text.

    Args:
        text: Text to display in banner
        char: Character to use for banner (default: BANNER_CHAR "=")
        width: Width of banner (default: BANNER_WIDTH 70)

    Examples:
        >>> print_banner("EXPERIMENT START")
        ======================================================================
        EXPERIMENT START
        ======================================================================
    """
    if char is None:
        char = BANNER_CHAR
    if width is None:
        width = BANNER_WIDTH

    print(f"\n{char * width}")
    print(text)
    print(f"{char * width}")


def print_section(title: str, char: str | None = None, width: int | None = None) -> None:
    """Print a section header.

    Args:
        title: Section title
        char: Character to use for underline (default: SECTION_CHAR "-")
        width: Width of underline (default: BANNER_WIDTH 70)

    Examples:
        >>> print_section("Configuration")
        Configuration
        ----------------------------------------------------------------------
    """
    if char is None:
        char = SECTION_CHAR
    if width is None:
        width = BANNER_WIDTH

    print(f"\n{title}")
    print(f"{char * width}")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string

    Examples:
        >>> format_duration(125.5)
        '2.1m'
        >>> format_duration(45.2)
        '45.2s'
    """
    if seconds >= 60:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        return f"{seconds:.1f}s"


def format_percentage(value: float, total: float) -> str:
    """Format a value as a percentage.

    Args:
        value: Current value
        total: Total value

    Returns:
        Formatted percentage string

    Examples:
        >>> format_percentage(25, 100)
        '25.0%'
    """
    if total == 0:
        return "0.0%"
    return f"{(value / total) * 100:.1f}%"


def print_separator(char: str | None = None, width: int | None = None) -> None:
    """Print a simple separator line.

    Args:
        char: Character to use for separator (default: BANNER_CHAR "=")
        width: Width of separator (default: BANNER_WIDTH 70)

    Examples:
        >>> print_separator()
        ======================================================================
    """
    if char is None:
        char = BANNER_CHAR
    if width is None:
        width = BANNER_WIDTH

    print(char * width)


def print_progress(
    current: int, total: int, prefix: str = "", suffix: str = "", width: int | None = None
) -> None:
    """Print a progress bar.

    Args:
        current: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        width: Width of progress bar (default: PROGRESS_BAR_WIDTH 50)

    Examples:
        >>> print_progress(50, 100, prefix="Processing")
        Processing [=========================     ] 50.0%
    """
    if width is None:
        width = PROGRESS_BAR_WIDTH

    if total == 0:
        return

    percent = (current / total) * 100
    filled_width = int(width * current // total)
    bar = "=" * filled_width + " " * (width - filled_width)

    print(f"\r{prefix} [{bar}] {percent:.1f}% {suffix}", end="", flush=True)

    if current == total:
        print()  # New line when complete
