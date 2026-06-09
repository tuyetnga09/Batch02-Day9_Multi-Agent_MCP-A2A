"""Đo latency end-to-end của hệ Multi-Agent (Stage 5).

Gửi 1 câu hỏi tới Customer Agent và đo tổng thời gian (wall-clock) từ lúc
gửi request đến lúc nhận đủ câu trả lời. Dùng cho bài tập cộng điểm latency.

Usage:
    uv run python measure_latency.py            # 1 lần đo
    uv run python measure_latency.py 3          # đo 3 lần, in trung bình
"""

import asyncio
import os
import sys
import time
from uuid import uuid4

import httpx
from dotenv import load_dotenv

load_dotenv()

CUSTOMER_AGENT_URL = os.getenv("CUSTOMER_AGENT_URL", "http://localhost:10100")

QUESTION = (
    "If a company breaks a contract and avoids taxes, "
    "what are the legal and regulatory consequences?"
)


async def one_run(http_client) -> tuple[float, int]:
    """Gửi 1 request, trả về (latency_giây, số_ký_tự_response)."""
    from a2a.client import A2AClient
    from a2a.types import (
        AgentCard,
        Message,
        MessageSendParams,
        Part,
        Role,
        SendMessageRequest,
        TextPart,
    )

    card_url = f"{CUSTOMER_AGENT_URL}/.well-known/agent.json"
    card_resp = await http_client.get(card_url)
    card_resp.raise_for_status()
    agent_card = AgentCard.model_validate(card_resp.json())
    client = A2AClient(httpx_client=http_client, agent_card=agent_card)

    message = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text=QUESTION))],
        message_id=str(uuid4()),
    )
    request = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(message=message))

    start = time.perf_counter()
    response = await client.send_message(request)
    elapsed = time.perf_counter() - start

    # Đếm độ dài text response (chỉ để xác nhận có kết quả thật)
    text = ""
    root = getattr(response, "root", response)
    result = getattr(root, "result", None)
    if result is not None:
        for art in (getattr(result, "artifacts", None) or []):
            for part in getattr(art, "parts", []) or []:
                p = getattr(part, "root", part)
                text += getattr(p, "text", "") or ""
        if not text:
            for part in (getattr(result, "parts", None) or []):
                p = getattr(part, "root", part)
                text += getattr(p, "text", "") or ""
    return elapsed, len(text)


async def main() -> None:
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"Model: {os.getenv('OPENROUTER_MODEL')}")
    print(f"Đo latency Stage 5 — {runs} lần chạy")
    print("-" * 60)

    latencies: list[float] = []
    async with httpx.AsyncClient(timeout=600.0) as http_client:
        for i in range(runs):
            try:
                elapsed, n = await one_run(http_client)
                latencies.append(elapsed)
                print(f"  Lần {i + 1}: {elapsed:6.1f}s   (response {n} ký tự)")
            except Exception as exc:  # noqa: BLE001
                print(f"  Lần {i + 1}: LỖI — {exc}")

    if latencies:
        avg = sum(latencies) / len(latencies)
        print("-" * 60)
        print(f"Latency trung bình: {avg:.1f}s  (min {min(latencies):.1f}s / max {max(latencies):.1f}s)")


if __name__ == "__main__":
    asyncio.run(main())
