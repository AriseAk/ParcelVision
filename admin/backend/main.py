"""
=============================================================================
ParcelVision Admin Panel — Flask Backend (main.py)
=============================================================================
Single-file Flask server containing:
  - Configuration & MongoDB connection
  - User model helpers (register, login, bcrypt hashing)
  - RatePlan model helpers (CRUD with nested weight tiers)
  - JWT authentication middleware
  - Auth routes:  POST /api/auth/register, POST /api/auth/login
  - Rates routes: GET/POST /api/rates, PUT/DELETE /api/rates/<id>

Run:  python main.py
=============================================================================
"""

import os
import datetime
import functools
import uuid

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId, errors as bson_errors
from dotenv import load_dotenv
import bcrypt
import jwt

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
load_dotenv()

MONGO_URI  = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME    = os.getenv("DB_NAME", "parcelvision_admin")
JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-change-me-in-production")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5001))

# ---------------------------------------------------------------------------
# 2. APP INITIALIZATION
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, supports_credentials=True)

# MongoDB connection
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]

# Collections
users_col        = db["users"]
rate_plans_col   = db["rate_plans"]
shipments_col    = db["shipments"]
transactions_col = db["transactions"]

# Ensure unique email index
users_col.create_index("email", unique=True)


# ---------------------------------------------------------------------------
# 3. HELPER — Convert ObjectId to string in documents
# ---------------------------------------------------------------------------
def serialize_doc(doc):
    """Convert MongoDB document to JSON-serializable dict."""
    if doc is None:
        return None
    doc["_id"] = str(doc["_id"])
    return doc


def serialize_list(cursor):
    """Convert a PyMongo cursor to a list of serialized dicts."""
    return [serialize_doc(doc) for doc in cursor]


# ---------------------------------------------------------------------------
# 4. USER MODEL HELPERS
# ---------------------------------------------------------------------------
# Schema:
# {
#   _id: ObjectId,
#   email: str (unique),
#   password_hash: str,
#   role: "admin" | "user",
#   auth_provider: "local" | "google",
#   created_at: datetime
# }

def hash_password(plain_text):
    """Hash a plaintext password with bcrypt."""
    return bcrypt.hashpw(plain_text.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def check_password(plain_text, hashed):
    """Verify a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(plain_text.encode("utf-8"), hashed.encode("utf-8"))


def create_user(email, password, role="user", auth_provider="local"):
    """Insert a new user and return the created document."""
    user_doc = {
        "email": email,
        "password_hash": hash_password(password),
        "role": role,
        "auth_provider": auth_provider,
        "created_at": datetime.datetime.utcnow(),
    }
    result = users_col.insert_one(user_doc)
    user_doc["_id"] = str(result.inserted_id)
    return user_doc


def find_user_by_email(email):
    """Find a user document by email address."""
    return users_col.find_one({"email": email})


# ---------------------------------------------------------------------------
# 5. RATE PLAN MODEL HELPERS
# ---------------------------------------------------------------------------
# Schema:
# {
#   _id: ObjectId,
#   origin: str,
#   destination: str,
#   carrier: str (DHL, FedEx, UPS, Aramex),
#   category: str (Medicine, Documents, General, Electronics, Food),
#   currency: str (USD, EUR, GBP, INR, AED),
#   unit: str (KG, LB),
#   estimated_days: int,
#   food_allowed: bool,
#   tiers: [{ start_weight: float, end_weight: float, price: float }],
#   created_at: datetime,
#   updated_at: datetime
# }

VALID_CARRIERS   = ["DHL", "FedEx", "UPS", "Aramex"]
VALID_CATEGORIES = ["Medicine", "Documents", "General", "Electronics", "Food"]
VALID_CURRENCIES = ["USD", "EUR", "GBP", "INR", "AED"]
VALID_UNITS      = ["KG", "LB"]


def validate_rate_plan(data):
    """Validate rate plan fields. Returns (clean_data, error_message)."""
    required = ["origin", "destination", "carrier", "category",
                 "currency", "unit", "estimated_days", "tiers"]

    for field in required:
        if field not in data or data[field] is None:
            return None, f"Missing required field: {field}"

    if data["carrier"] not in VALID_CARRIERS:
        return None, f"Invalid carrier. Must be one of: {VALID_CARRIERS}"
    if data["category"] not in VALID_CATEGORIES:
        return None, f"Invalid category. Must be one of: {VALID_CATEGORIES}"
    if data["currency"] not in VALID_CURRENCIES:
        return None, f"Invalid currency. Must be one of: {VALID_CURRENCIES}"
    if data["unit"] not in VALID_UNITS:
        return None, f"Invalid unit. Must be one of: {VALID_UNITS}"

    # Validate tiers
    tiers = data.get("tiers", [])
    if not isinstance(tiers, list) or len(tiers) == 0:
        return None, "At least one weight tier is required"

    clean_tiers = []
    for i, tier in enumerate(tiers):
        try:
            clean_tier = {
                "start_weight": float(tier["start_weight"]),
                "end_weight":   float(tier["end_weight"]),
                "price":        float(tier["price"]),
            }
        except (KeyError, ValueError, TypeError):
            return None, f"Tier {i + 1}: must have numeric start_weight, end_weight, and price"

        if clean_tier["start_weight"] < 0 or clean_tier["end_weight"] <= 0:
            return None, f"Tier {i + 1}: weights must be positive"
        if clean_tier["end_weight"] <= clean_tier["start_weight"]:
            return None, f"Tier {i + 1}: end_weight must be greater than start_weight"
        if clean_tier["price"] < 0:
            return None, f"Tier {i + 1}: price cannot be negative"

        clean_tiers.append(clean_tier)

    clean_data = {
        "origin":         str(data["origin"]).strip(),
        "destination":    str(data["destination"]).strip(),
        "carrier":        data["carrier"],
        "category":       data["category"],
        "currency":       data["currency"],
        "unit":           data["unit"],
        "estimated_days": int(data["estimated_days"]),
        "food_allowed":   bool(data.get("food_allowed", False)),
        "tiers":          clean_tiers,
    }

    return clean_data, None


# ---------------------------------------------------------------------------
# 6. JWT MIDDLEWARE
# ---------------------------------------------------------------------------
def generate_token(user_doc):
    """Generate a JWT token for a user."""
    payload = {
        "user_id": str(user_doc["_id"]),
        "email":   user_doc["email"],
        "role":    user_doc["role"],
        "exp":     datetime.datetime.utcnow() + datetime.timedelta(days=7),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def token_required(f):
    """Decorator that protects routes with JWT verification."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401

        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            g.current_user = payload
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# 7. AUTH ROUTES — /api/auth
# ---------------------------------------------------------------------------
@app.route("/api/auth/register", methods=["POST"])
def register():
    """Register a new user account."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    email    = (data.get("email") or "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    # Check if user already exists
    if find_user_by_email(email):
        return jsonify({"error": "A user with this email already exists"}), 409

    user = create_user(email, password)
    token = generate_token({"_id": user["_id"], "email": user["email"], "role": user["role"]})

    return jsonify({
        "message": "User registered successfully",
        "token":   token,
        "user": {
            "id":    user["_id"],
            "email": user["email"],
            "role":  user["role"],
        },
    }), 201


@app.route("/api/auth/login", methods=["POST"])
def login():
    """Authenticate a user and return a JWT."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    email    = (data.get("email") or "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = find_user_by_email(email)
    if not user or not check_password(password, user["password_hash"]):
        return jsonify({"error": "Invalid email or password"}), 401

    token = generate_token(user)

    return jsonify({
        "message": "Login successful",
        "token":   token,
        "user": {
            "id":    str(user["_id"]),
            "email": user["email"],
            "role":  user["role"],
        },
    }), 200


@app.route("/api/auth/google", methods=["POST"])
def google_auth():
    """Authenticate or register a user logging in via Google OAuth."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    email = (data.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "Email is required"}), 400

    user = find_user_by_email(email)
    if not user:
        # Create a new user with Google as the auth provider
        placeholder_password = str(uuid.uuid4())
        user = create_user(email, placeholder_password, role="user", auth_provider="google")
    else:
        # If user exists but didn't have an auth provider set (or it was local),
        # we can ensure auth_provider is set.
        if user.get("auth_provider") != "google":
            users_col.update_one(
                {"_id": ObjectId(user["_id"]) if isinstance(user["_id"], str) else user["_id"]},
                {"$set": {"auth_provider": "google"}}
            )
            user["auth_provider"] = "google"

    token = generate_token(user)

    return jsonify({
        "message": "Google authentication successful",
        "token":   token,
        "user": {
            "id":    str(user["_id"]),
            "email": user["email"],
            "role":  user["role"],
        },
    }), 200


# ---------------------------------------------------------------------------
# 8. RATES ROUTES — /api/rates
# ---------------------------------------------------------------------------
@app.route("/api/rates", methods=["GET"])
@token_required
def get_rates():
    """List all rate plans. Supports optional query filters."""
    query = {}

    # Optional filters from query params
    for field in ["carrier", "category", "origin", "destination"]:
        value = request.args.get(field)
        if value:
            query[field] = value

    rates = serialize_list(rate_plans_col.find(query).sort("created_at", -1))
    return jsonify({"rates": rates, "count": len(rates)}), 200


@app.route("/api/rates", methods=["POST"])
@token_required
def create_rate():
    """Create a new rate plan."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    clean_data, error = validate_rate_plan(data)
    if error:
        return jsonify({"error": error}), 400

    now = datetime.datetime.utcnow()
    clean_data["created_at"] = now
    clean_data["updated_at"] = now
    clean_data["created_by"] = g.current_user["user_id"]

    result = rate_plans_col.insert_one(clean_data)
    clean_data["_id"] = str(result.inserted_id)

    return jsonify({"message": "Rate plan created", "rate": clean_data}), 201


@app.route("/api/rates/<rate_id>", methods=["GET"])
@token_required
def get_rate(rate_id):
    """Get a single rate plan by ID."""
    try:
        oid = ObjectId(rate_id)
    except bson_errors.InvalidId:
        return jsonify({"error": "Invalid rate plan ID"}), 400

    rate = rate_plans_col.find_one({"_id": oid})
    if not rate:
        return jsonify({"error": "Rate plan not found"}), 404

    return jsonify({"rate": serialize_doc(rate)}), 200


@app.route("/api/rates/<rate_id>", methods=["PUT"])
@token_required
def update_rate(rate_id):
    """Update an existing rate plan."""
    try:
        oid = ObjectId(rate_id)
    except bson_errors.InvalidId:
        return jsonify({"error": "Invalid rate plan ID"}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    clean_data, error = validate_rate_plan(data)
    if error:
        return jsonify({"error": error}), 400

    clean_data["updated_at"] = datetime.datetime.utcnow()

    result = rate_plans_col.update_one({"_id": oid}, {"$set": clean_data})
    if result.matched_count == 0:
        return jsonify({"error": "Rate plan not found"}), 404

    updated = serialize_doc(rate_plans_col.find_one({"_id": oid}))
    return jsonify({"message": "Rate plan updated", "rate": updated}), 200


@app.route("/api/rates/<rate_id>", methods=["DELETE"])
@token_required
def delete_rate(rate_id):
    """Delete a rate plan."""
    try:
        oid = ObjectId(rate_id)
    except bson_errors.InvalidId:
        return jsonify({"error": "Invalid rate plan ID"}), 400

    result = rate_plans_col.delete_one({"_id": oid})
    if result.deleted_count == 0:
        return jsonify({"error": "Rate plan not found"}), 404

    return jsonify({"message": "Rate plan deleted"}), 200


# ---------------------------------------------------------------------------
# 9. SHIPMENTS ROUTES — /api/shipments
# ---------------------------------------------------------------------------
VALID_STATUSES = ["pending", "picked_up", "in_transit", "out_for_delivery", "delivered", "cancelled", "returned"]


def validate_shipment(data):
    """Validate shipment fields. Returns (clean_data, error_message)."""
    required = ["tracking_number", "sender_name", "sender_address",
                 "receiver_name", "receiver_address", "origin", "destination",
                 "carrier", "weight"]

    for field in required:
        if field not in data or data[field] is None or str(data[field]).strip() == "":
            return None, f"Missing required field: {field}"

    if data.get("carrier") and data["carrier"] not in VALID_CARRIERS:
        return None, f"Invalid carrier. Must be one of: {VALID_CARRIERS}"

    status = data.get("status", "pending")
    if status not in VALID_STATUSES:
        return None, f"Invalid status. Must be one of: {VALID_STATUSES}"

    try:
        weight = float(data["weight"])
        if weight <= 0:
            return None, "Weight must be positive"
    except (ValueError, TypeError):
        return None, "Weight must be a valid number"

    clean_data = {
        "tracking_number":  str(data["tracking_number"]).strip(),
        "sender_name":      str(data["sender_name"]).strip(),
        "sender_address":   str(data["sender_address"]).strip(),
        "receiver_name":    str(data["receiver_name"]).strip(),
        "receiver_address": str(data["receiver_address"]).strip(),
        "origin":           str(data["origin"]).strip(),
        "destination":      str(data["destination"]).strip(),
        "carrier":          data["carrier"],
        "weight":           weight,
        "unit":             data.get("unit", "KG"),
        "status":           status,
        "notes":            str(data.get("notes", "")).strip(),
    }

    return clean_data, None


@app.route("/api/shipments", methods=["GET"])
@token_required
def get_shipments():
    """List all shipments with optional filters."""
    query = {}
    for field in ["carrier", "status", "origin", "destination"]:
        value = request.args.get(field)
        if value:
            query[field] = value

    search = request.args.get("search")
    if search:
        query["$or"] = [
            {"tracking_number": {"$regex": search, "$options": "i"}},
            {"sender_name":     {"$regex": search, "$options": "i"}},
            {"receiver_name":   {"$regex": search, "$options": "i"}},
        ]

    shipments = serialize_list(shipments_col.find(query).sort("created_at", -1))
    return jsonify({"shipments": shipments, "count": len(shipments)}), 200


@app.route("/api/shipments", methods=["POST"])
@token_required
def create_shipment():
    """Create a new shipment."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    clean_data, error = validate_shipment(data)
    if error:
        return jsonify({"error": error}), 400

    now = datetime.datetime.utcnow()
    clean_data["created_at"] = now
    clean_data["updated_at"] = now
    clean_data["created_by"] = g.current_user["user_id"]

    result = shipments_col.insert_one(clean_data)
    clean_data["_id"] = str(result.inserted_id)

    return jsonify({"message": "Shipment created", "shipment": clean_data}), 201


@app.route("/api/shipments/<shipment_id>", methods=["PUT"])
@token_required
def update_shipment(shipment_id):
    """Update an existing shipment."""
    try:
        oid = ObjectId(shipment_id)
    except bson_errors.InvalidId:
        return jsonify({"error": "Invalid shipment ID"}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    clean_data, error = validate_shipment(data)
    if error:
        return jsonify({"error": error}), 400

    clean_data["updated_at"] = datetime.datetime.utcnow()
    result = shipments_col.update_one({"_id": oid}, {"$set": clean_data})
    if result.matched_count == 0:
        return jsonify({"error": "Shipment not found"}), 404

    updated = serialize_doc(shipments_col.find_one({"_id": oid}))
    return jsonify({"message": "Shipment updated", "shipment": updated}), 200


@app.route("/api/shipments/<shipment_id>", methods=["DELETE"])
@token_required
def delete_shipment(shipment_id):
    """Delete a shipment."""
    try:
        oid = ObjectId(shipment_id)
    except bson_errors.InvalidId:
        return jsonify({"error": "Invalid shipment ID"}), 400

    result = shipments_col.delete_one({"_id": oid})
    if result.deleted_count == 0:
        return jsonify({"error": "Shipment not found"}), 404

    return jsonify({"message": "Shipment deleted"}), 200


# ---------------------------------------------------------------------------
# 10. USERS ROUTES — /api/users
# ---------------------------------------------------------------------------
@app.route("/api/users", methods=["GET"])
@token_required
def get_users():
    """List all registered users (admin only). Password hashes are excluded."""
    search = request.args.get("search")
    query = {}
    if search:
        query["email"] = {"$regex": search, "$options": "i"}

    role_filter = request.args.get("role")
    if role_filter:
        query["role"] = role_filter

    users = list(users_col.find(query, {"password_hash": 0}).sort("created_at", -1))
    serialized = serialize_list(users)
    return jsonify({"users": serialized, "count": len(serialized)}), 200


@app.route("/api/users/<user_id>", methods=["PUT"])
@token_required
def update_user(user_id):
    """Update a user's role."""
    try:
        oid = ObjectId(user_id)
    except bson_errors.InvalidId:
        return jsonify({"error": "Invalid user ID"}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    update_fields = {}
    if "role" in data and data["role"] in ["admin", "user"]:
        update_fields["role"] = data["role"]

    if not update_fields:
        return jsonify({"error": "No valid fields to update"}), 400

    result = users_col.update_one({"_id": oid}, {"$set": update_fields})
    if result.matched_count == 0:
        return jsonify({"error": "User not found"}), 404

    updated = serialize_doc(users_col.find_one({"_id": oid}, {"password_hash": 0}))
    return jsonify({"message": "User updated", "user": updated}), 200


@app.route("/api/users/<user_id>", methods=["DELETE"])
@token_required
def delete_user(user_id):
    """Delete a user."""
    try:
        oid = ObjectId(user_id)
    except bson_errors.InvalidId:
        return jsonify({"error": "Invalid user ID"}), 400

    # Prevent self-deletion
    if str(oid) == g.current_user.get("user_id"):
        return jsonify({"error": "Cannot delete your own account"}), 400

    result = users_col.delete_one({"_id": oid})
    if result.deleted_count == 0:
        return jsonify({"error": "User not found"}), 404

    return jsonify({"message": "User deleted"}), 200


# ---------------------------------------------------------------------------
# 11. TRANSACTIONS ROUTES — /api/transactions
# ---------------------------------------------------------------------------
VALID_PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "bank_transfer", "cash", "upi"]
VALID_TXN_STATUSES    = ["pending", "completed", "failed", "refunded"]


def validate_transaction(data):
    """Validate transaction fields. Returns (clean_data, error_message)."""
    required = ["shipment_id", "amount", "currency", "payment_method"]

    for field in required:
        if field not in data or data[field] is None or str(data[field]).strip() == "":
            return None, f"Missing required field: {field}"

    try:
        amount = float(data["amount"])
        if amount <= 0:
            return None, "Amount must be positive"
    except (ValueError, TypeError):
        return None, "Amount must be a valid number"

    if data["currency"] not in VALID_CURRENCIES:
        return None, f"Invalid currency. Must be one of: {VALID_CURRENCIES}"

    if data["payment_method"] not in VALID_PAYMENT_METHODS:
        return None, f"Invalid payment method. Must be one of: {VALID_PAYMENT_METHODS}"

    status = data.get("status", "pending")
    if status not in VALID_TXN_STATUSES:
        return None, f"Invalid status. Must be one of: {VALID_TXN_STATUSES}"

    clean_data = {
        "shipment_id":    str(data["shipment_id"]).strip(),
        "amount":         amount,
        "currency":       data["currency"],
        "payment_method": data["payment_method"],
        "status":         status,
        "description":    str(data.get("description", "")).strip(),
    }

    return clean_data, None


@app.route("/api/transactions", methods=["GET"])
@token_required
def get_transactions():
    """List all transactions with optional filters."""
    query = {}

    for field in ["status", "payment_method", "currency"]:
        value = request.args.get(field)
        if value:
            query[field] = value

    search = request.args.get("search")
    if search:
        query["$or"] = [
            {"shipment_id":  {"$regex": search, "$options": "i"}},
            {"description":  {"$regex": search, "$options": "i"}},
        ]

    transactions = serialize_list(transactions_col.find(query).sort("created_at", -1))
    return jsonify({"transactions": transactions, "count": len(transactions)}), 200


@app.route("/api/transactions", methods=["POST"])
@token_required
def create_transaction():
    """Create a new transaction."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    clean_data, error = validate_transaction(data)
    if error:
        return jsonify({"error": error}), 400

    now = datetime.datetime.utcnow()
    clean_data["created_at"] = now
    clean_data["updated_at"] = now
    clean_data["created_by"] = g.current_user["user_id"]

    result = transactions_col.insert_one(clean_data)
    clean_data["_id"] = str(result.inserted_id)

    return jsonify({"message": "Transaction created", "transaction": clean_data}), 201


@app.route("/api/transactions/<txn_id>", methods=["PUT"])
@token_required
def update_transaction(txn_id):
    """Update a transaction's status."""
    try:
        oid = ObjectId(txn_id)
    except bson_errors.InvalidId:
        return jsonify({"error": "Invalid transaction ID"}), 400

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    update_fields = {}
    if "status" in data:
        if data["status"] not in VALID_TXN_STATUSES:
            return jsonify({"error": f"Invalid status. Must be one of: {VALID_TXN_STATUSES}"}), 400
        update_fields["status"] = data["status"]

    if not update_fields:
        return jsonify({"error": "No valid fields to update"}), 400

    update_fields["updated_at"] = datetime.datetime.utcnow()
    result = transactions_col.update_one({"_id": oid}, {"$set": update_fields})
    if result.matched_count == 0:
        return jsonify({"error": "Transaction not found"}), 404

    updated = serialize_doc(transactions_col.find_one({"_id": oid}))
    return jsonify({"message": "Transaction updated", "transaction": updated}), 200


@app.route("/api/transactions/<txn_id>", methods=["DELETE"])
@token_required
def delete_transaction(txn_id):
    """Delete a transaction."""
    try:
        oid = ObjectId(txn_id)
    except bson_errors.InvalidId:
        return jsonify({"error": "Invalid transaction ID"}), 400

    result = transactions_col.delete_one({"_id": oid})
    if result.deleted_count == 0:
        return jsonify({"error": "Transaction not found"}), 404

    return jsonify({"message": "Transaction deleted"}), 200


# ---------------------------------------------------------------------------
# 12. HEALTH CHECK
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "ok", "service": "ParcelVision Admin API"}), 200


# ---------------------------------------------------------------------------
# 13. RUN SERVER
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"🚀 ParcelVision Admin API starting on port {FLASK_PORT}")
    print(f"📦 MongoDB: {MONGO_URI}/{DB_NAME}")
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=True, use_reloader=False)
