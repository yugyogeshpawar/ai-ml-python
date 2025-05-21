# 📘 FastAPI User Authentication API

This is a basic FastAPI backend API that provides user **signup** and **login** functionality using a **MySQL** database and **JWT** tokens for authentication.

---

## 🔧 Tech Stack

* **FastAPI** (Web framework)
* **MySQL** (Database)
* **bcrypt / passlib** (Password hashing – although hashing is not fully implemented)
* **JWT (via python-jose)** (Token-based authentication)
* **CORS Middleware** (Cross-origin request support)

---

## 🔐 JWT Configuration

* **SECRET\_KEY**: `my-secret-token`
* **ALGORITHM**: `HS256`
* **Token Expiry**: 30 minutes

---

## 🗂 Endpoints

### `GET /`

**Description**: Health check endpoint.
**Response**:

```json
{
  "message": "Hello, Okay!"
}
```

---

### `POST /signup`

**Description**: Register a new user.
**Form Data Parameters**:

* `fullName` (str): Full name of the user.
* `email` (str): User email (must be unique).
* `password` (str): Password.
* `confirmPassword` (str): Must match `password`.

**Response**:

```json
{
  "message": "User registered successfully."
}
```

**Errors**:

* `400 Bad Request`: Passwords do not match or email already exists.
* `500 Internal Server Error`: Database connection or insertion error.

---

### `POST /login`

**Description**: Authenticates a user and returns a JWT token.
**Form Data Parameters**:

* `email` (str): User email.
* `password` (str): User password.

**Response**:

```json
{
  "access_token": "jwt_token_here",
  "token_type": "bearer"
}
```

**Errors**:

* `401 Unauthorized`: Invalid email or password.
* `500 Internal Server Error`: Database connection issue.

---

## 🔐 Security

* **OAuth2PasswordBearer** is initialized for future integration of protected routes.
* **Password storage**: Passwords are currently **not hashed**. Replace the `verify_password` and password saving logic with hashed password handling using `passlib`.

Example:

```python
hashed_password = pwd_context.hash(password)
pwd_context.verify(plain_password, hashed_password)
```

---

## ⚠️ Notes

* ✅ CORS is enabled for all origins (`*`) – consider restricting this in production.
* ❌ Passwords are stored as **plain text** – this should be fixed for production environments.
* 🔒 Add HTTPS and secure JWT storage for a production deployment.
* 🧪 Ensure to use migrations or ORM for scalable DB management.

