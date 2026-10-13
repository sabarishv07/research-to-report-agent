"""Email delivery over SMTP.

Deliberately NOT registered in the model's tool registry: sending email is a side effect
with real-world impact, so the pipeline calls it directly with the address the user gave.
The model can never choose a recipient.
"""

from __future__ import annotations

import re
import smtplib
from email.message import EmailMessage

from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import Settings
from ..exceptions import ToolError
from .base import Tool, ToolResult

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class SendEmailInput(BaseModel):
    to: str
    subject: str = Field(min_length=1, max_length=200)
    body: str
    attachment_name: str | None = None


class EmailTool(Tool):
    name = "send_email"
    description = "Send the finished report by email."
    input_model = SendEmailInput

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def configured(self) -> bool:
        s = self.settings
        return bool(s.smtp_host and (s.smtp_from or s.smtp_user))

    def build_message(self, args: SendEmailInput) -> EmailMessage:
        s = self.settings
        msg = EmailMessage()
        msg["From"] = s.smtp_from or s.smtp_user
        msg["To"] = args.to
        msg["Subject"] = args.subject.replace("\n", " ")  # header injection guard
        msg.set_content(args.body)
        if args.attachment_name:
            msg.add_attachment(
                args.body.encode("utf-8"),
                maintype="text",
                subtype="markdown",
                filename=args.attachment_name,
            )
        return msg

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=8),
        retry=retry_if_exception_type((smtplib.SMTPException, OSError)),
        reraise=True,
    )
    def _send(self, msg: EmailMessage) -> None:
        s = self.settings
        with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=20) as smtp:
            if s.smtp_starttls:
                smtp.starttls()
            if s.smtp_user:
                smtp.login(s.smtp_user, s.smtp_password)
            smtp.send_message(msg)

    def run(self, args: SendEmailInput) -> ToolResult:
        if not self.configured:
            raise ToolError("SMTP is not configured (set SMTP_HOST and SMTP_FROM/SMTP_USER)")
        if not EMAIL_RE.match(args.to):
            raise ToolError(f"'{args.to}' is not a valid email address")
        try:
            self._send(self.build_message(args))
        except (smtplib.SMTPException, OSError) as exc:
            raise ToolError(f"could not send email: {exc}") from exc
        return ToolResult(content=f"Email sent to {args.to}", data={"to": args.to})
