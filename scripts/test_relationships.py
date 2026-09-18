from sqlalchemy import select

from app.db.session import SessionLocal

from app.models.company import Company
from app.models.role import Role
from app.models.user import User


def main():

    with SessionLocal() as db:

        users = db.scalars(
            select(User)
        ).all()

        print(f"Users count: {len(users)}")
        print("-" * 50)

        for user in users:

            print(f"User: {user.full_name}")
            print(f"Email: {user.email}")

            if user.company:
                print(f"Company: {user.company.name}")
            else:
                print("Company: None")

            print(f"Role: {user.role.name}")

            print("-" * 50)
            
        print("\nCOMPANIES AND THEIR USERS")
        print("=" * 50)

        companies = db.scalars(
            select(Company)
        ).all()

        for company in companies:

            print(f"\nCompany: {company.name}")

            for user in company.users:
                print(
                    f"- {user.full_name} ({user.role.name})"
                )


if __name__ == "__main__":
    main()