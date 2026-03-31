import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger("background-task-service")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def parse_iso_or_none(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


@dataclass
class TaskRunResult:
    ok: bool
    details: str


class BrowserAutomationAdapter:
    async def claim_daily_rewards(self, task: Dict[str, Any]) -> TaskRunResult:
        return TaskRunResult(ok=False, details="Browser adapter not configured for claim_daily_rewards")

    async def close_advertisement(self, task: Dict[str, Any]) -> TaskRunResult:
        return TaskRunResult(ok=False, details="Browser adapter not configured for close_advertisement")


class ServiceAutomationAdapter:
    async def memory_update(self, task: Dict[str, Any]) -> TaskRunResult:
        log_path = Path("KMS") / "background_memory_update.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"{iso_now()} memory updated\n", encoding="utf-8")
        return TaskRunResult(ok=True, details=f"Wrote {log_path}")

    async def reload_service(self, task: Dict[str, Any]) -> TaskRunResult:
        log_path = Path("KMS") / "background_service_reload.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"{iso_now()} service reload requested\n", encoding="utf-8")
        return TaskRunResult(ok=True, details=f"Wrote {log_path}")


class BackgroundTaskService:
    def __init__(
        self,
        task_file: str = "background_tasks.json",
        bounty_board_file: str = "shared/bounty_board.json",
        browser_adapter: Optional[BrowserAutomationAdapter] = None,
        service_adapter: Optional[ServiceAutomationAdapter] = None,
    ) -> None:
        self.task_file = Path(task_file)
        self.bounty_board_file = Path(bounty_board_file)
        self.browser_adapter = browser_adapter or BrowserAutomationAdapter()
        self.service_adapter = service_adapter or ServiceAutomationAdapter()

    def _load(self) -> Dict[str, Any]:
        if not self.task_file.exists():
            raise FileNotFoundError(f"Task file not found: {self.task_file}")
        return json.loads(self.task_file.read_text(encoding="utf-8"))

    def _save(self, data: Dict[str, Any]) -> None:
        self.task_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _available(self, task: Dict[str, Any], now: datetime) -> bool:
        runtime = task.get("runtime", {})
        state = task.get("state", {})
        if not runtime.get("enabled", True):
            return False
        if state.get("status") == "running":
            return False

        next_at = parse_iso_or_none(state.get("next_available_at"))
        if next_at and now < next_at:
            return False
        return True

    def _refresh_availability(self, data: Dict[str, Any]) -> None:
        now = utc_now()
        for task in data.get("tasks", []):
            task.setdefault("state", {})
            task["state"]["available"] = self._available(task, now)

    def _to_bounty_board(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self._refresh_availability(data)
        items: List[Dict[str, Any]] = []
        for t in data.get("tasks", []):
            items.append(
                {
                    "id": t.get("id"),
                    "name": t.get("name"),
                    "action": t.get("action"),
                    "domain": t.get("domain"),
                    "available": t.get("state", {}).get("available", False),
                    "cooldown_sec": t.get("bounty", {}).get("cooldown_sec", 0),
                    "next_available_at": t.get("state", {}).get("next_available_at"),
                    "reward": t.get("bounty", {}).get("reward", 0),
                    "priority": t.get("bounty", {}).get("priority", 0),
                    "state_key": t.get("q_learning", {}).get("state_key"),
                    "action_key": t.get("q_learning", {}).get("action_key"),
                    "reward_weight": t.get("q_learning", {}).get("reward_weight", 1.0),
                    "last_result": t.get("state", {}).get("last_result"),
                }
            )
        return {
            "updated_at": iso_now(),
            "loop_type": "closed",
            "tasks": items,
        }

    def _write_bounty_board(self, data: Dict[str, Any]) -> None:
        board = self._to_bounty_board(data)
        self.bounty_board_file.parent.mkdir(parents=True, exist_ok=True)
        self.bounty_board_file.write_text(json.dumps(board, indent=2), encoding="utf-8")

    async def _execute_action(self, task: Dict[str, Any]) -> TaskRunResult:
        action = (task.get("action") or "").strip()
        if action == "memory_update":
            return await self.service_adapter.memory_update(task)
        if action == "reload_service":
            return await self.service_adapter.reload_service(task)
        if action == "claim_daily_rewards":
            return await self.browser_adapter.claim_daily_rewards(task)
        if action == "close_advertisement":
            return await self.browser_adapter.close_advertisement(task)
        return TaskRunResult(ok=False, details=f"Unknown action: {action}")

    def _pick_task_id(self, data: Dict[str, Any], preferred_task_id: Optional[str]) -> Optional[str]:
        tasks = data.get("tasks", [])
        if preferred_task_id:
            for t in tasks:
                if t.get("id") == preferred_task_id and t.get("state", {}).get("available", False):
                    return preferred_task_id
            return None

        candidates = [t for t in tasks if t.get("state", {}).get("available", False)]
        if not candidates:
            return None

        # Greedy baseline for closed loop: Q-agent can override by passing task id.
        candidates.sort(
            key=lambda t: (
                float(t.get("bounty", {}).get("reward", 0)) * float(t.get("q_learning", {}).get("reward_weight", 1.0)),
                int(t.get("bounty", {}).get("priority", 0)),
            ),
            reverse=True,
        )
        return candidates[0].get("id")

    def _mark_post_run(self, task: Dict[str, Any], result: TaskRunResult) -> None:
        state = task.setdefault("state", {})
        bounty = task.get("bounty", {})
        cooldown = int(bounty.get("cooldown_sec", 0))
        now = utc_now()
        state["status"] = "success" if result.ok else "failed"
        state["last_completed_at"] = now.isoformat()
        state["last_result"] = result.details
        state["consecutive_failures"] = 0 if result.ok else int(state.get("consecutive_failures", 0)) + 1
        state["next_available_at"] = (now + timedelta(seconds=cooldown)).isoformat() if cooldown > 0 else None
        state["available"] = False if cooldown > 0 else True

    async def run_once(self, preferred_task_id: Optional[str] = None) -> Dict[str, Any]:
        data = self._load()
        self._refresh_availability(data)

        task_id = self._pick_task_id(data, preferred_task_id)
        if not task_id:
            self._write_bounty_board(data)
            self._save(data)
            return {"ran": False, "reason": "no_available_task"}

        task = next(t for t in data.get("tasks", []) if t.get("id") == task_id)
        state = task.setdefault("state", {})
        state["status"] = "running"
        state["last_started_at"] = iso_now()
        state["available"] = False

        result = await self._execute_action(task)
        self._mark_post_run(task, result)
        self._refresh_availability(data)
        self._write_bounty_board(data)
        self._save(data)

        return {"ran": True, "task_id": task_id, "ok": result.ok, "details": result.details}

    async def run_forever(self, interval_sec: float = 5.0) -> None:
        while True:
            result = await self.run_once()
            if result.get("ran"):
                logger.info("Task run: %s", result)
            await asyncio.sleep(interval_sec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Background task scheduler for browser/service maintenance.")
    parser.add_argument("--task-file", default="background_tasks.json")
    parser.add_argument("--bounty-board-file", default="shared/bounty_board.json")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-sec", type=float, default=5.0)
    parser.add_argument("--task-id", default=None, help="Force a specific task id for one run.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


async def _main_async(args: argparse.Namespace) -> None:
    service = BackgroundTaskService(
        task_file=args.task_file,
        bounty_board_file=args.bounty_board_file,
    )
    if args.once:
        result = await service.run_once(preferred_task_id=args.task_id)
        logger.info("Run once: %s", result)
        return
    await service.run_forever(interval_sec=args.interval_sec)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
