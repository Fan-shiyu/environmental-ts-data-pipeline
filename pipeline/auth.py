"""GEE authentication: initialise the Earth Engine API using a service account or OAuth."""

import sys


def init_gee(project: str) -> None:
    """Initialize Earth Engine. Assumes user has run `earthengine authenticate`."""
    try:
        import ee
        ee.Initialize(project=project)
        print("  GEE initialised OK.\n")
    except Exception as exc:
        print(f"\nERROR: GEE authentication failed.\n  {exc}")
        print(
            "\nFix: run  earthengine authenticate  in your terminal, then retry.\n"
            f"     Make sure the project '{project}' exists and you have access."
        )
        sys.exit(1)
