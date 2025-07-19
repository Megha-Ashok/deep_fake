import sqlite3
from werkzeug.security import generate_password_hash

DATABASE = 'database.db'

def create_users_table():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Create users table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin BOOLEAN DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Table 'users' ensured.")


def insert_admin_user():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    name = 'Admin User'
    email = 'megharashokashok@gmail.com'
    password = generate_password_hash('admin123')

    try:
        cursor.execute("INSERT INTO users (name, email, password, is_admin) VALUES (?, ?, ?, ?)",
                       (name, email, password, 1))
        conn.commit()
        print("✅ Admin user inserted successfully.")
    except sqlite3.IntegrityError:
        print("ℹ️ Admin already exists.")
    conn.close()


def get_admin_users():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT id, name, email FROM users WHERE is_admin = 1")
    admins = cursor.fetchall()

    if not admins:
        print("⚠️ No admin users found.")
    else:
        print("\n📋 Admin Users:")
        for admin in admins:
            print(f"ID: {admin['id']}, Name: {admin['name']}, Email: {admin['email']}")
    conn.close()


if __name__ == '__main__':
    create_users_table()
    insert_admin_user()
    get_admin_users()
