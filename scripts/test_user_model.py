from sqlalchemy import select

from app.db.session import SessionLocal
from app.models import Role, User


def main():
    with SessionLocal() as db:

        roles = db.scalars(
            select(Role)
        ).all()

        print("Roles:")

        for role in roles:
            print(
                role.id,
                role.name
            )

        print("-" * 50)

        users = db.scalars(
            select(User)
        ).all()

        print(f"Users count: {len(users)}")

        for user in users:
            print(
                user.id,
                user.full_name,
                user.email,
                user.company_id,
                user.role_id,
                user.is_active
            )


if __name__ == "__main__":
    main()