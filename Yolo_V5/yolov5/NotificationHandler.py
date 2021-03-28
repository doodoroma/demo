import os
import pymsteams
from twilio.rest import Client


class NotificationHandler:
    def __init__(
        self,
        teams_url,
        to_numbers = None,
        activation: dict = None,
    ) -> None:
        self.teams_url = teams_url
        self.myTeamsMessage = pymsteams.connectorcard(self.teams_url)

        account_sid = os.getenv("SMS_SID")
        auth_token = os.getenv("SMS_TOKEN")
        self.client = Client(account_sid, auth_token)
        self.to_numbers = to_numbers or []

        self.activation = activation or {}

    def _notify_teams(self, msg: str) -> None:
        self.myTeamsMessage.text(msg)
        self.myTeamsMessage.send()


    def _notify_sms(self, msg: str) -> None:
        for number in self.to_numbers:
            (
                self.client.messages
                            .create(
                                body=msg,
                                from_='+16515041302',
                                to=number
                            )        
            )

    def notify(self, msg) -> None:
        if self.activation.get('Teams', False):
            self._notify_teams(msg)
        if self.activation.get('SMS', False):
            self._notify_sms(msg)


    