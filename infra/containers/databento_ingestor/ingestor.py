import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

import databento as db
from azure.eventhub import EventData
from azure.eventhub.aio import EventHubProducerClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


def get_secret_from_keyvault(vault_url: str, secret_name: str) -> str:
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=vault_url, credential=credential)
    return client.get_secret(secret_name).value


class DatabentoToEventHub:
    def __init__(
        self,
        databento_key: str,
        eventhub_conn_str: str,
        eventhub_name: str,
        symbols: list[str],
        schema: str = "mbo",
    ):
        self.databento_key = databento_key
        self.eventhub_conn_str = eventhub_conn_str
        self.eventhub_name = eventhub_name
        self.symbols = symbols
        self.schema = schema
        self.producer: Optional[EventHubProducerClient] = None
        self.client: Optional[db.Live] = None

    def transform_to_bronze_envelope(self, msg) -> dict:
        return {
            "event_time": msg.ts_event,
            "ingest_time": int(datetime.now(timezone.utc).timestamp() * 1e9),
            "venue": "GLBX.MDP3",
            "symbol": msg.instrument_id,
            "instrument_type": "FUT",
            "underlier": "ES",
            "contract_id": str(msg.instrument_id),
            "action": chr(msg.action) if hasattr(msg, "action") else None,
            "order_id": msg.order_id if hasattr(msg, "order_id") else None,
            "side": chr(msg.side) if hasattr(msg, "side") else None,
            "price": msg.price / 1e9 if hasattr(msg, "price") else None,
            "size": msg.size if hasattr(msg, "size") else None,
            "sequence": msg.sequence if hasattr(msg, "sequence") else None,
            "payload": {
                "ts_recv": msg.ts_recv if hasattr(msg, "ts_recv") else None,
                "flags": msg.flags if hasattr(msg, "flags") else None,
            },
        }

    async def send_batch(self, events: list[dict]):
        if not self.producer:
            self.producer = EventHubProducerClient.from_connection_string(
                self.eventhub_conn_str, eventhub_name=self.eventhub_name
            )

        event_data_batch = await self.producer.create_batch()
        for event in events:
            try:
                event_data_batch.add(EventData(json.dumps(event)))
            except ValueError:
                await self.producer.send_batch(event_data_batch)
                event_data_batch = await self.producer.create_batch()
                event_data_batch.add(EventData(json.dumps(event)))

        if len(event_data_batch) > 0:
            await self.producer.send_batch(event_data_batch)

    async def run(self):
        self.client = db.Live(key=self.databento_key)

        self.client.subscribe(
            dataset="GLBX.MDP3",
            schema=self.schema,
            stype_in="parent",
            symbols=self.symbols,
        )

        batch = []
        batch_start = time.time()
        batch_size = 100
        batch_timeout = 0.1

        async for msg in self.client:
            if msg.rtype in (0, 1, 2, 3):
                envelope = self.transform_to_bronze_envelope(msg)
                batch.append(envelope)

                if len(batch) >= batch_size or (time.time() - batch_start) >= batch_timeout:
                    await self.send_batch(batch)
                    batch = []
                    batch_start = time.time()

    async def close(self):
        if self.producer:
            await self.producer.close()


async def main():
    vault_url = os.environ.get("KEY_VAULT_URL", "https://kvspymasterdevoxxrlojskv.vault.azure.net/")
    databento_key = os.environ.get("DATABENTO_API_KEY")
    if not databento_key:
        databento_key = get_secret_from_keyvault(vault_url, "databento-api-key")

    eventhub_conn_str = os.environ.get("EVENTHUB_CONNECTION_STRING")
    if not eventhub_conn_str:
        eventhub_conn_str = get_secret_from_keyvault(vault_url, "eventhub-connection-string")

    eventhub_name = os.environ.get("EVENTHUB_NAME", "mbo_raw")
    symbols = os.environ.get("SYMBOLS", "ES.FUT").split(",")

    ingestor = DatabentoToEventHub(
        databento_key=databento_key,
        eventhub_conn_str=eventhub_conn_str,
        eventhub_name=eventhub_name,
        symbols=symbols,
    )

    try:
        await ingestor.run()
    finally:
        await ingestor.close()


if __name__ == "__main__":
    asyncio.run(main())
