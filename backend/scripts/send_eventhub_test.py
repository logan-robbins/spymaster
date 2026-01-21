import json
import os
import time

from azure.eventhub import EventHubProducerClient, EventData


def main() -> None:
    conn_str = os.environ.get("EVENTHUB_CONNECTION_STRING")
    if not conn_str:
        raise RuntimeError("EVENTHUB_CONNECTION_STRING is required")

    eventhub_name = os.environ.get("EVENTHUB_NAME", "mbo_raw")
    now_ns = time.time_ns()
    test_event = {
        "event_time": now_ns,
        "ingest_time": now_ns,
        "venue": "GLBX.MDP3",
        "symbol": "ESH6",
        "instrument_type": "FUT",
        "underlier": "ES",
        "contract_id": "ESH6",
        "action": "T",
        "order_id": 1,
        "side": "B",
        "price": 5000.0,
        "size": 1,
        "sequence": 1,
        "payload": "{}",
    }

    producer = EventHubProducerClient.from_connection_string(
        conn_str, eventhub_name=eventhub_name
    )
    with producer:
        batch = producer.create_batch()
        batch.add(EventData(json.dumps(test_event)))
        producer.send_batch(batch)


if __name__ == "__main__":
    main()
