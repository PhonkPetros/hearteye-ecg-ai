from marshmallow import Schema, fields, validate, validates, ValidationError
import re

class ECGUploadSchema(Schema):
    patient_name = fields.Str(required=True, allow_none=False)
    age = fields.Int(validate=validate.Range(min=0, max=120), allow_none=False)
    gender = fields.Str(
        validate=validate.OneOf(["M", "F", "O"]),
        allow_none=False
    )
    notes = fields.Str(required=False, allow_none=True)

class UserRegisterSchema(Schema):
    username = fields.String(
        required=True,
        validate=[
            validate.Length(min=3, max=80, error="Username must be between 3 and 80 characters."),
            validate.Regexp(
                r"^[a-zA-Z0-9_.-]+$",
                error="Username can only contain letters, numbers, underscores, hyphens, and periods."
            )
        ]
    )
    email = fields.Email(required=True, validate=validate.Length(max=120, error="Email must be less than 120 characters."))
    password = fields.String(required=True, validate=validate.Length(min=4, error="Password must be at least 4 characters long."))



class UserLoginSchema(Schema):
    username = fields.String(required=True, validate=validate.Length(min=1))
    password = fields.String(required=True, validate=validate.Length(min=4))