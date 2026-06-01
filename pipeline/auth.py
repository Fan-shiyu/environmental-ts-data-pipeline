"""GEE authentication: initialise the Earth Engine API using a service account or OAuth."""

import json
import sys


def init_gee(project: str, service_account_key: str | None = None) -> None:
    """Initialize Earth Engine.

    If service_account_key (a JSON string) is provided, authenticate with the
    service account — used by GitHub Actions. Otherwise fall back to interactive
    credentials (assumes the user has run `earthengine authenticate`) — local dev.

    Parameters
    ----------
    project : str
        GEE project ID, e.g. 'sensingclues-ndvi'.
    service_account_key : str | None
        JSON string of the service account key (from an environment variable).
        If None, uses interactive auth.
    """
    try:
        import ee
        if service_account_key:
            key_data = json.loads(service_account_key)
            credentials = ee.ServiceAccountCredentials(
                email=key_data["client_email"],
                key_data=service_account_key,
            )
            ee.Initialize(credentials, project=project)
        else:
            ee.Initialize(project=project)
        print("  GEE initialised OK.\n")
    except Exception as exc:
        print(f"\nERROR: GEE authentication failed.\n  {exc}")
        print(
            "\nFix: run  earthengine authenticate  in your terminal, then retry.\n"
            f"     Make sure the project '{project}' exists and you have access."
        )
        sys.exit(1)
