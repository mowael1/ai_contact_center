from sqlalchemy import select

from app.db.session import SessionLocal
from app.models import Company, Customer


def main():
    db = SessionLocal()

    try:
        statement = select(Customer)

        customers = list(
            db.scalars(statement).all()
        )

        print("Customers:")

        for customer in customers:
            print(
                customer.id,
                customer.full_name,
                customer.phone,
                customer.company.name,
                customer.is_active
            )

    finally:
        db.close()


if __name__ == "__main__":
    main()