"""Synthetic personas and transaction records for AltScore AI.

Contains profiles and wholly synthetic transaction histories for three personas:
- Ramesh Kumar: Gig worker (rideshare and delivery) in an urban area.
- Sunita Devi: Rural artisan and dairy cooperative entrepreneur.
- Aarav Sharma: Small shopkeeper (neighborhood kirana store) in an urban area.

Demographic attributes (gender, geography) are stored purely as profile and audit
metadata for later fairness evaluation; they must never be passed to scoring models.
"""

from typing import Any, Dict


def get_personas() -> Dict[str, Dict[str, Any]]:
    """Return synthetic applicant profiles and their transaction histories.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping persona IDs to profile data,
        demographic metadata, liquid balance, and synthetic transaction lists.
    """
    return {
        "ramesh_kumar": {
            "name": "Ramesh Kumar",
            "role": "Gig Worker (Rideshare & Delivery Partner)",
            "summary": (
                "Earns variable income through rideshare and delivery platform payouts. "
                "Maintains consistent bill payments and moderate savings buffer."
            ),
            # Demographics are audit metadata only; never input to credit models
            "demographics": {
                "gender": "Male",
                "geography": "Urban",
            },
            "liquid_balance": 18500.0,
            "transactions": [
                {
                    "date": "2026-07-05",
                    "description": "Platform weekly earnings settlement",
                    "category": "Income",
                    "type": "Credit",
                    "amount": 24000.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-07-12",
                    "description": "Mobile & internet recharge",
                    "category": "Utility Bill",
                    "type": "Debit",
                    "amount": 1200.0,
                    "status": "Paid on time",
                },
                {
                    "date": "2026-07-20",
                    "description": "Vehicle fuel & routine maintenance",
                    "category": "Essential Outflow",
                    "type": "Debit",
                    "amount": 11000.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-08-05",
                    "description": "Platform weekly earnings settlement",
                    "category": "Income",
                    "type": "Credit",
                    "amount": 21500.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-08-15",
                    "description": "Residential electricity bill",
                    "category": "Utility Bill",
                    "type": "Debit",
                    "amount": 1800.0,
                    "status": "Paid on time",
                },
                {
                    "date": "2026-08-22",
                    "description": "Vehicle fuel & household groceries",
                    "category": "Essential Outflow",
                    "type": "Debit",
                    "amount": 12500.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-09-06",
                    "description": "Platform weekly earnings settlement",
                    "category": "Income",
                    "type": "Credit",
                    "amount": 26000.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-09-14",
                    "description": "Mobile postpaid bill",
                    "category": "Utility Bill",
                    "type": "Debit",
                    "amount": 1300.0,
                    "status": "Paid on time",
                },
                {
                    "date": "2026-09-24",
                    "description": "Vehicle fuel & servicing",
                    "category": "Essential Outflow",
                    "type": "Debit",
                    "amount": 10500.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-10-05",
                    "description": "Platform weekly earnings settlement",
                    "category": "Income",
                    "type": "Credit",
                    "amount": 23000.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-10-16",
                    "description": "Residential electricity bill",
                    "category": "Utility Bill",
                    "type": "Debit",
                    "amount": 1950.0,
                    "status": "Paid on time",
                },
                {
                    "date": "2026-10-25",
                    "description": "Vehicle fuel & groceries",
                    "category": "Essential Outflow",
                    "type": "Debit",
                    "amount": 11200.0,
                    "status": "Completed",
                },
            ],
        },
        "sunita_devi": {
            "name": "Sunita Devi",
            "role": "Rural Artisan & Dairy Entrepreneur",
            "summary": (
                "Runs dairy sales through a village cooperative and artisanal handicrafts. "
                "Maintains steady cash inflows, disciplined utility payments, and healthy liquidity."
            ),
            # Demographics are audit metadata only; never input to credit models
            "demographics": {
                "gender": "Female",
                "geography": "Rural",
            },
            "liquid_balance": 32000.0,
            "transactions": [
                {
                    "date": "2026-07-08",
                    "description": "Dairy cooperative milk settlement",
                    "category": "Income",
                    "type": "Credit",
                    "amount": 18000.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-07-15",
                    "description": "Solar home lighting bill",
                    "category": "Utility Bill",
                    "type": "Debit",
                    "amount": 850.0,
                    "status": "Paid on time",
                },
                {
                    "date": "2026-07-28",
                    "description": "Cattle feed & household supplies",
                    "category": "Essential Outflow",
                    "type": "Debit",
                    "amount": 7500.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-08-09",
                    "description": "Dairy cooperative milk settlement",
                    "category": "Income",
                    "type": "Credit",
                    "amount": 19500.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-08-18",
                    "description": "Water cooperative user charge",
                    "category": "Utility Bill",
                    "type": "Debit",
                    "amount": 700.0,
                    "status": "Paid on time",
                },
                {
                    "date": "2026-08-29",
                    "description": "Cattle feed & monthly groceries",
                    "category": "Essential Outflow",
                    "type": "Debit",
                    "amount": 8000.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-09-10",
                    "description": "Artisan craft exhibition batch payout",
                    "category": "Income",
                    "type": "Credit",
                    "amount": 22000.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-09-16",
                    "description": "Solar home lighting bill",
                    "category": "Utility Bill",
                    "type": "Debit",
                    "amount": 850.0,
                    "status": "Paid on time",
                },
                {
                    "date": "2026-09-27",
                    "description": "Artisan raw materials & household supplies",
                    "category": "Essential Outflow",
                    "type": "Debit",
                    "amount": 7800.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-10-09",
                    "description": "Dairy cooperative milk settlement",
                    "category": "Income",
                    "type": "Credit",
                    "amount": 18500.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-10-18",
                    "description": "Water cooperative user charge",
                    "category": "Utility Bill",
                    "type": "Debit",
                    "amount": 750.0,
                    "status": "Paid on time",
                },
                {
                    "date": "2026-10-28",
                    "description": "Cattle feed & provisions",
                    "category": "Essential Outflow",
                    "type": "Debit",
                    "amount": 8200.0,
                    "status": "Completed",
                },
            ],
        },
        "aarav_sharma": {
            "name": "Aarav Sharma",
            "role": "Small Shopkeeper (Kirana Store Owner)",
            "summary": (
                "Operates a neighborhood retail grocery store with digital QR payments. "
                "Handles high cash turnover and inventory reinvestment, with tighter liquid reserves."
            ),
            # Demographics are audit metadata only; never input to credit models
            "demographics": {
                "gender": "Male",
                "geography": "Urban",
            },
            "liquid_balance": 12000.0,
            "transactions": [
                {
                    "date": "2026-07-04",
                    "description": "UPI QR customer payment settlements",
                    "category": "Income",
                    "type": "Credit",
                    "amount": 35000.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-07-10",
                    "description": "Commercial shop electricity bill",
                    "category": "Utility Bill",
                    "type": "Debit",
                    "amount": 3200.0,
                    "status": "Paid on time",
                },
                {
                    "date": "2026-07-25",
                    "description": "Wholesale grocery restocking",
                    "category": "Essential Outflow",
                    "type": "Debit",
                    "amount": 28000.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-08-05",
                    "description": "UPI QR customer payment settlements",
                    "category": "Income",
                    "type": "Credit",
                    "amount": 29000.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-08-14",
                    "description": "Commercial shop electricity bill",
                    "category": "Utility Bill",
                    "type": "Debit",
                    "amount": 3400.0,
                    "status": "Paid late",
                },
                {
                    "date": "2026-08-27",
                    "description": "FMCG inventory restocking",
                    "category": "Essential Outflow",
                    "type": "Debit",
                    "amount": 27500.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-09-04",
                    "description": "UPI QR customer payment settlements",
                    "category": "Income",
                    "type": "Credit",
                    "amount": 31000.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-09-12",
                    "description": "Commercial shop electricity bill",
                    "category": "Utility Bill",
                    "type": "Debit",
                    "amount": 3100.0,
                    "status": "Paid on time",
                },
                {
                    "date": "2026-09-26",
                    "description": "Wholesale grocery restocking",
                    "category": "Essential Outflow",
                    "type": "Debit",
                    "amount": 29000.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-10-06",
                    "description": "UPI QR customer payment settlements",
                    "category": "Income",
                    "type": "Credit",
                    "amount": 26500.0,
                    "status": "Completed",
                },
                {
                    "date": "2026-10-15",
                    "description": "Commercial shop electricity bill",
                    "category": "Utility Bill",
                    "type": "Debit",
                    "amount": 3500.0,
                    "status": "Unpaid",
                },
                {
                    "date": "2026-10-26",
                    "description": "Wholesale inventory restocking",
                    "category": "Essential Outflow",
                    "type": "Debit",
                    "amount": 26000.0,
                    "status": "Completed",
                },
            ],
        },
    }

