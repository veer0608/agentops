"""Seed a demo customer + subscription. Run: `python -m app.seed`."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Customer, Subscription

DEMO_EMAIL = "ada@example.com"


def seed_demo(session: Session) -> str:
    """Insert the demo customer + subscription if absent. Returns the customer id."""
    customer = session.scalars(
        select(Customer).where(Customer.email == DEMO_EMAIL)
    ).first()
    if customer is None:
        customer = Customer(name="Ada Lovelace", email=DEMO_EMAIL, tier="pro")
        session.add(customer)
        session.flush()
        session.add(
            Subscription(
                customer_id=customer.id,
                plan="Pro",
                status="active",
                renews_at="2026-08-01",
                mrr_cents=4900,
            )
        )
        session.commit()
    return customer.id


def main() -> None:
    from app.db import SessionLocal

    with SessionLocal() as session:
        customer_id = seed_demo(session)
        print(f"seeded demo customer_id={customer_id} (email={DEMO_EMAIL})")


if __name__ == "__main__":
    main()
