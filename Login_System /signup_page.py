import streamlit as st
import re
import mysql.connector

def validate_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email)

def create_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="user_auth"
    )

def create_users_table():
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL
    )
    """)
    connection.commit()
    cursor.close()
    connection.close()

def save_user(username, email, password):
    try:
        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (username, email, password))
        connection.commit()
        cursor.close()
        connection.close()
    except Exception as e:
        st.error(f"Error saving user: {e}")

def check_login(email, password):
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    return user

def main():
    st.title("User Authentication System")

    menu = ["Signup", "Login" , "Profile"]
    choice = st.sidebar.selectbox("Menu", menu)

    create_users_table()

    if choice == "Signup":
        st.subheader("Signup Page")

        username = st.text_input("Username", max_chars=20)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Signup"):
            if not username:
                st.error("Username is required")
            elif not email or not validate_email(email):
                st.error("Please enter a valid email address")
            elif not password or len(password) < 6:
                st.error("Password must be at least 6 characters long")
            elif password != confirm_password:
                st.error("Passwords do not match")
            else:
                save_user(username, email, password)
                st.success(f"Welcome {username}! You have successfully signed up.")
                st.info("You can now log in using your credentials.")

    elif choice == "Login":
        st.subheader("Login Page")

        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = check_login(email, password)
            if user:
                st.success(f"Welcome back, {user[1]}!")
            else:
                st.error("Invalid email or password")
    elif choice == "Profile":
        st.subheader("Profile Page")
        # show welcome page with greetings
        st.write(f"Welcome User!")


if __name__ == "__main__":
    main()
