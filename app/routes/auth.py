import logging
from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from marshmallow import ValidationError

# Import custom error handling and exceptions
from ..utils.error_handlers import handle_errors
from ..utils.exceptions import APIError, ErrorCodes

# Import models and schemas
from ..models import db, User
from ..schemas import UserRegisterSchema, UserLoginSchema 


auth_bp = Blueprint("auth", __name__)
logger = logging.getLogger(__name__)


@auth_bp.route("/register", methods=["POST"])
@handle_errors
def register():
    # Attempt to load and validate input data using Marshmallow
    schema = UserRegisterSchema()
    try:
        data = schema.load(request.get_json())
        logger.info("Register attempt - data validated", extra={"username": data.get("username"), "email": data.get("email")})
    except ValidationError as e:
        # MarshmallowValidationError is caught by @handle_errors and converted to APIError (INVALID_FORMAT, 400)
        logger.warning("Register failed: validation error", extra={"messages": e.messages})
        raise e

    # Check for existing username
    if User.query.filter_by(username=data["username"]).first():
        logger.warning("Register failed: username already exists", extra={"username": data["username"]})
        raise APIError(ErrorCodes.DUPLICATE_ENTRY, "Username already exists", status_code=409, details={"field": "username"})

    # Check for existing email
    if User.query.filter_by(email=data["email"]).first():
        logger.warning("Register failed: email already exists", extra={"email": data["email"]})
        raise APIError(ErrorCodes.DUPLICATE_ENTRY, "Email already exists", status_code=409, details={"field": "email"})

    user = User(username=data["username"], email=data["email"])
    user.set_password(data["password"])

    try:
        db.session.add(user)
        db.session.commit()
        logger.info("User registered successfully", extra={"user_id": user.id, "username": user.username})
    except Exception as e:
        db.session.rollback()
        logger.exception("Database error during registration", exc_info=e)
        raise APIError(ErrorCodes.INTERNAL_ERROR, "Failed to register user due to a database error.", status_code=500)

    return jsonify({"message": "User registered successfully"}), 201


@auth_bp.route("/login", methods=["POST"])
@handle_errors
def login():
    # Attempt to load and validate input data using Marshmallow
    schema = UserLoginSchema()
    try:
        data = schema.load(request.get_json())
        logger.info("Login attempt - data validated", extra={"username": data.get("username")})
    except ValidationError as e:
        logger.warning("Login failed: validation error", extra={"messages": e.messages})
        raise e

    user = User.query.filter_by(username=data["username"]).first()

    # Consolidate invalid username/password to avoid revealing if username exists
    if not user or not user.check_password(data["password"]):
        logger.warning("Login failed: invalid credentials", extra={"username": data.get("username")})
        # Use 401 Unauthorized for invalid credentials
        raise APIError(ErrorCodes.INVALID_FORMAT, "Invalid username or password", status_code=401)

    access_token = create_access_token(identity=str(user.id))
    logger.info("Login successful", extra={"user_id": user.id, "username": user.username})

    return (
        jsonify(
            {
                "access_token": access_token,
                "user": {"id": user.id, "username": user.username, "email": user.email},
            }
        ),
        200,
    )


@auth_bp.route("/me", methods=["GET"])
@jwt_required()
@handle_errors 
def get_current_user():
    user_id = get_jwt_identity()
    logger.info("Get current user request", extra={"user_id": user_id})

    user = User.query.get(user_id)

    if not user:
        logger.warning("Current user not found", extra={"user_id": user_id})
        # If JWT is valid but user no longer exists (e.g., deleted), return 404
        raise APIError(ErrorCodes.RECORD_NOT_FOUND, "User not found", status_code=404)

    logger.info("Current user retrieved", extra={"user_id": user.id, "username": user.username})
    return jsonify({"id": user.id, "username": user.username, "email": user.email}), 200