"""Authenticated machine-readable interface to BMD Compute.

This package lives outside ``backend/`` on purpose: ``backend/`` is the
runtime package uploaded to POWER, while this package only runs in the web
service. It holds authentication, request validation and response projection.
It contains no calculation methodology; scientific resolution is supplied by
the web application's existing shared functions (see ``main.plan_calculation_request``).
"""

API_VERSION = "v1"

__all__ = ["API_VERSION"]
