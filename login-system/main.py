
from fastapi import FastAPI, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr
import mysql.connector
from mysql.connector import Error

# Create a FastAPI application
app = FastAPI()

# Allow cross-origin requests (update origin for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# JWT Configuration
SECRET_KEY = "my-secret-token"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


# Pydantic model for request body validation
class User(BaseModel):
    name: str
    email: EmailStr
    password: str

# Password hashing config
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 token URL
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Database connection configuration
db_config = {
    "host": "mysql-486c28e-yogieee6094-c2ae.j.aivencloud.com",
    "user": "avnadmin",
    "password": "AVNS_6KdboIlyiUKBqA0d0O7",
    "port": 23230,
    "database": "loginsystem"
}


connection = mysql.connector.connect(**db_config)
cursor = connection.cursor()

# Define a route at the root web address ("/")
@app.get("/")
def read_root():
    return {"message": "Hello, Okay!"}


# Signup route
@app.post("/signup")
async def signup(
    fullName: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirmPassword: str = Form(...)
):
    if password != confirmPassword:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    try:

        insert_query = """
        INSERT INTO user (name, email, password)
        VALUES (%s, %s, %s)
        """
        values = (fullName, email, password)

        cursor.execute(insert_query, values)
        connection.commit()

        return {"message": "User registered successfully."}

    except Error as e:
        print(e.errno)
        if e.errno == 1062:
            raise HTTPException(status_code=400, detail="Email already exists.")
        raise HTTPException(status_code=500, detail="Database error.")

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()







# Utility function to verify password
def verify_password(plain_password, hashed_password):
    return plain_password == hashed_password

# Utility function to create JWT token
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Login route
@app.post("/login")
async def login(email: str = Form(...), password: str = Form(...)):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM user WHERE email = %s", (email,))
        user = cursor.fetchone()

        if not user or not verify_password(password, user["password"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Create JWT token
        access_token = create_access_token(data={"sub": user["email"]})
        return {"access_token": access_token, "token_type": "bearer"}

    except Error as e:
        print("Database error:", e)
        raise HTTPException(status_code=500, detail="Database error")

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


