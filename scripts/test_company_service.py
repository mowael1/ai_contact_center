from app.db.session import SessionLocal
from app.services.company_service import (
    get_all_companies,
    get_company_by_id,
    get_company_by_name,
)


def main():
    with SessionLocal() as db:

        print("ALL COMPANIES")
        print("-" * 50)

        companies = get_all_companies(db)

        for company in companies:
            print(
                company.id,
                company.name
            )

        print("\nGET BY ID")
        print("-" * 50)

        company = get_company_by_id(
            db,
            5
        )

        if company:
            print(
                company.id,
                company.name
            )
        else:
            print("Company not found")

        print("\nGET BY NAME")
        print("-" * 50)

        company = get_company_by_name(
            db,
            "ABC"
        )

        if company:
            print(
                company.id,
                company.name
            )
        else:
            print("Company not found")


if __name__ == "__main__":
    main()