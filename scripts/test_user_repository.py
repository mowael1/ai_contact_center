from app.db.session import SessionLocal
from app.repositories.user_repository import (
    get_all,
    get_by_email,
    get_by_id,
)


def main():
    with SessionLocal() as db:

        print("ALL USERS")
        print("-" * 50)

        users = get_all(db)

        for user in users:
            print(
                user.id,
                user.full_name,
                user.email,
                user.company_id,
                user.role_id
            )

        print("\nGET BY ID")
        print("-" * 50)

        user = get_by_id(
            db,
            1
        )

        if user:
            print(
                user.id,
                user.full_name,
                user.email
            )
        else:
            print("User not found")

        print("\nGET BY EMAIL")
        print("-" * 50)

        user = get_by_email(
            db,
            "mw86596@gmail.com"
        )

        if user:
            print(
                user.id,
                user.full_name,
                user.email
            )
        else:
            print("User not found")


if __name__ == "__main__":
    main()