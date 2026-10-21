"""Shared-password sign-in with two roles.

POLICY_DESK_PASSWORD lets employees ask questions; POLICY_DESK_ADMIN_PASSWORD
also lets admins add and remove documents. With neither set, sign-in is off
and everyone acts as admin, which suits local development.
"""

import hmac
import os
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


def passwords() -> dict[str, str]:
    """Role -> password from the environment, admin checked first."""

    roles = {
        "admin": os.environ.get("POLICY_DESK_ADMIN_PASSWORD"),
        "user": os.environ.get("POLICY_DESK_PASSWORD"),
    }
    return {role: password for role, password in roles.items() if password}


def auth_required() -> bool:
    return bool(passwords())


def current_role(request: Request) -> str | None:
    if not auth_required():
        return "admin"

    role = request.session.get("role")
    return role if role in passwords() else None


def require_user(request: Request) -> str:
    role = current_role(request)

    if role is None:
        raise HTTPException(401, "Sign in to continue")

    return role


def require_admin(role: str = Depends(require_user)) -> str:
    if role != "admin":
        raise HTTPException(403, "Only admins can add or remove documents")

    return role


class LoginRequest(BaseModel):
    password: str


@router.post("/login")
def login(body: LoginRequest, request: Request):
    for role, password in passwords().items():
        if hmac.compare_digest(body.password.encode(), password.encode()):
            request.session.clear()
            request.session["role"] = role
            return {"role": role}

    if not auth_required():
        return {"role": "admin"}

    # ponytail: slows each guess but not parallel guessing; add per-IP
    # lockout before exposing this to the internet.
    time.sleep(1)
    raise HTTPException(401, "That password isn't right")


@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"role": None}


@router.get("/me")
def me(request: Request):
    return {"role": current_role(request), "auth_required": auth_required()}
