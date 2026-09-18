from app.db.session import SessionLocal
from app.repositories.company_repository import (
    get_all,
    get_by_id,
    get_by_name,
)


def main():

    with SessionLocal() as db:

        print("ALL COMPANIES")
        print("-" * 50)

        companies = get_all(db)

        for company in companies:
            print(
                company.id,
                company.name
            )

        print("\nGET BY ID")
        print("-" * 50)

        company = get_by_id(
            db,
            5
        )

        if company:
            print(
                company.id,
                company.name
            )

        print("\nGET BY NAME")
        print("-" * 50)

        company = get_by_name(
            db,
            "ABC"
        )

        if company:
            print(
                company.id,
                company.name
            )


if __name__ == "__main__":
    main()