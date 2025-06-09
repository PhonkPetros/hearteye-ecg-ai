import logging
from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from ..models import db, User

auth_bp = Blueprint("auth", __name__)
logger = logging.getLogger(__name__)


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    logger.info("Register attempt", extra={"data_keys": list(data.keys()) if data else None})

    if (
        not data
        or "username" not in data
        or "password" not in data
        or "email" not in data
    ):
        logger.warning("Register failed: missing required fields", extra={"received_data": data})
        return jsonify({"error": "Missing required fields"}), 400

    if User.query.filter_by(username=data["username"]).first():
        logger.warning("Register failed: username exists", extra={"username": data["username"]})
        return jsonify({"error": "Username already exists"}), 400

    if User.query.filter_by(email=data["email"]).first():
        logger.warning("Register failed: email exists", extra={"email": data["email"]})
        return jsonify({"error": "Email already exists"}), 400

    user = User(username=data["username"], email=data["email"])
    user.set_password(data["password"])

    try:
        db.session.add(user)
        db.session.commit()
        logger.info("User registered successfully", extra={"user_id": user.id, "username": user.username})
    except Exception as e:
        logger.exception("Database error during registration", exc_info=e)
        db.session.rollback()
        return jsonify({"error": "Internal server error"}), 500

    return jsonify({"message": "User registered successfully"}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    logger.info("Login attempt", extra={"data_keys": list(data.keys()) if data else None})

    if not data or "username" not in data or "password" not in data:
        logger.warning("Login failed: missing username or password", extra={"received_data": data})
        return jsonify({"error": "Missing username or password"}), 400

    user = User.query.filter_by(username=data["username"]).first()

    if not user or not user.check_password(data["password"]):
        logger.warning("Login failed: invalid credentials", extra={"username": data.get("username")})
        return jsonify({"error": "Invalid username or password"}), 401

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
def get_current_user():
    user_id = get_jwt_identity()
    logger.info("Get current user request", extra={"user_id": user_id})

    user = User.query.get(user_id)

    if not user:
        logger.warning("Current user not found", extra={"user_id": user_id})
        return jsonify({"error": "User not found"}), 404

    logger.info("Current user retrieved", extra={"user_id": user.id, "username": user.username})
    return jsonify({"id": user.id, "username": user.username, "email": user.email}), 200
