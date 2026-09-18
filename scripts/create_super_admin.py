from sqlalchemy import select

from app.core.security import hash_password
from app.db.session import SessionLocal
from app.models import Role
from app.repositories import user_repository

def main():
    with SessionLocal() as db:

        super_admin_role = db.scalars(
            select(Role)
            .where(Role.name == "super_admin")
        ).first()
        
        print(super_admin_role)

        if super_admin_role is None:
            print("super_admin role not found")
            return

        email = "superadmin@test.com"

        existing_user = user_repository.get_by_email(
            db,
            email
        )

        if existing_user:
            print("Super Admin already exists")
            return

        user = user_repository.create(
            db,
            full_name="Super Admin",
            email=email,
            password_hash=hash_password(
                "123456"
            ),
            role_id=super_admin_role.id,
            company_id=None,
            is_active=True
        )

        print("Super Admin created")
        print("ID:", user.id)
        print("Email:", user.email)


if __name__ == "__main__":
    main()