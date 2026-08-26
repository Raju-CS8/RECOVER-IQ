from __future__ import annotations

from enum import Enum


class RecoveryAction(str, Enum):
    """
    Actions that RecoverIQ may select during a payment-recovery episode.

    The action space is deliberately closed. The decision engine may select
    only an action represented here; it cannot invent arbitrary provider
    operations or mutate payment state directly.
    """

    RETRY_PAYMENT = "retry_payment"
    REQUEST_PAYMENT_METHOD_UPDATE = "request_payment_method_update"
    SEND_RECOVERY_MESSAGE = "send_recovery_message"
    WAIT = "wait"
    STOP_RECOVERY = "stop_recovery"