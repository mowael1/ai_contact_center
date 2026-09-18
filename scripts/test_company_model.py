from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.company import Company


def main():
    with SessionLocal() as db:

        statement = select(Company)

        companies = db.scalars(statement).all()

        print(f"Companies count: {len(companies)}")

        for company in companies:
            print(
                company.id,
                company.name,
                company.email,
                company.phone,
                company.is_active
            )


if __name__ == "__main__":
    main()