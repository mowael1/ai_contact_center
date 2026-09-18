from app.core.security import verify_password
from app.db.session import SessionLocal
from app.repositories.user_repository import get_by_email


def main():
    with SessionLocal() as db:

        user = get_by_email(
            db,
            "superadmin@test.com"
        )

        if user is None:
            print("User not found")
            return

        result = verify_password(
            "123456",
            user.password_hash
        )

        print("Password valid:", result)


if __name__ == "__main__":
    main()