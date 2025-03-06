# User Authentication System

This project is a simple **User Authentication System** built with **Streamlit** and **MySQL**. It provides a basic interface for users to **Sign Up** and **Log In**. User data, including usernames, emails, and passwords, is stored securely in a MySQL database.

## Features

✅ User Signup (with validation checks)

✅ User Login (with credential verification)

✅ MySQL database integration

✅ Automatic user table creation

---

## Requirements

- Python 3.x
- Streamlit
- MySQL Server

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Install Dependencies

```bash
pip install streamlit mysql-connector-python
```

### 3. Configure Database Connection

Open the `signup_page.py` file and update the `create_connection()` function with your MySQL credentials:

```python
    host="localhost",
    user="your_username",
    password="your_password",
    database="your_database"
```

### 4. Create MySQL Database

Ensure that you have a MySQL database set up. You can create one using the following commands in your MySQL shell:

```sql
CREATE DATABASE your_database;
USE your_database;
```

The system will automatically create a `users` table if it doesn't exist.

### 5. Run the Application

```bash
streamlit run signup_page.py
```

The application will open in your default web browser.

---

## How It Works

1. **Signup Page**: Users enter their username, email, and password to create an account. Input validation ensures correctness.

2. **Login Page**: Users can log in with their registered email and password. If credentials are correct, they are welcomed back.

## Future Improvements

- Password hashing for better security
- Session management
- Enhanced error handling

Feel free to extend this project based on your needs!

