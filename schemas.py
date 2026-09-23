from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, ConfigDict


class SignupRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)
    name: str = Field(min_length=1, max_length=80)


class LoginRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    email: str
    name: str

    class Config:
        from_attributes = True


class NewConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role_id: str = Field(min_length=1, max_length=64)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    conversation_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=12000)


class MessageOut(BaseModel):
    sender: str
    content: str
    timestamp: datetime

    class Config:
        from_attributes = True


class ConversationOut(BaseModel):
    id: str
    role_id: str
    created_at: datetime
    messages: list[MessageOut] = []

    class Config:
        from_attributes = True


class ConversationSummaryOut(BaseModel):
    id: str
    role_id: str
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentOut(BaseModel):
    id: str
    filename: str
    uploaded_at: datetime
    chunk_count: int

    class Config:
        from_attributes = True


class MemoryOut(BaseModel):
    id: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class ReminderOut(BaseModel):
    id: str
    content: str
    done: bool
    created_at: datetime

    class Config:
        from_attributes = True
