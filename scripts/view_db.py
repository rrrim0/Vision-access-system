import sqlite3

conn = sqlite3.connect("data/access_control.db")
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Таблицы:")
for row in cursor.fetchall():
    print(" -", row[0])

print("\nПользователи:")
cursor.execute("SELECT id, username, full_name, role, is_active, created_at FROM users;")
for row in cursor.fetchall():
    print(row)

print("\nШаблоны лиц:")
cursor.execute("SELECT id, user_id, image_path, created_at FROM face_templates;")
for row in cursor.fetchall():
    print(row)

print("\nНастройки:")
cursor.execute("SELECT key, value FROM system_settings;")
for row in cursor.fetchall():
    print(row)

conn.close()