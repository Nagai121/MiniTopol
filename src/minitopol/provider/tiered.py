"""Tier-aware production Provider and bounded learning-batch lookahead."""
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from uuid import uuid4

from .optimized import OptimizedProvider
from ..storage_tiers import TieredStorage


class TieredProvider(OptimizedProvider):
    """Keep Provider v1 data buffers unchanged; add an optional tier scheduler."""

    def __init__(self, workspace, budget):
        super().__init__(workspace, budget)
        self.tiers = TieredStorage(self.workspace.path)
        self._plans = {}
        self._plan_lock = RLock()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tier-prefetch")

    def configure_storage(self, request):
        allowed = {"enabled", "external_root", "cache_percent", "write_policy",
                   "prefetch_batches"}
        if not isinstance(request, dict) or set(request) - allowed:
            raise ValueError("Invalid storage configuration")
        return self.tiers.configure(**request)

    def storage_status(self):
        return self.tiers.status()

    def offload_storage(self, request):
        if not isinstance(request, dict) or set(request) != {"relative"}:
            raise ValueError("Expected a logical relative path")
        return self.tiers.offload(request["relative"])

    def ensure_storage(self, request):
        if not isinstance(request, dict) or set(request) != {"relative"}:
            raise ValueError("Expected a logical relative path")
        return self.tiers.ensure(request["relative"])

    def ingest_storage(self, request):
        if not isinstance(request, dict) or set(request) != {"trajectory"}:
            raise ValueError("Expected a trajectory ID")
        return self.tiers.ingest_sources(request["trajectory"])

    def prune_storage(self, request):
        if not isinstance(request, dict) or set(request) - {"all_cache"}:
            raise ValueError("Invalid prune parameters")
        value = request.get("all_cache", False)
        if type(value) is not bool:
            raise ValueError("all_cache must be boolean")
        return self.tiers.prune(all_cache=value)

    def _schedule(self, plan):
        config = self.tiers.settings()
        width = config["prefetch_batches"]
        while plan["submitted"] < min(len(plan["batches"]), plan["consumed"] + width):
            index = plan["submitted"]
            request = plan["batches"][index]
            plan["states"][index] = "prefetching"
            future = self._pool.submit(self.tiers.prefetch, request, self.workspace)
            plan["futures"][index] = future
            plan["submitted"] += 1

            def done(task, *, position=index, target=plan):
                with self._plan_lock:
                    try:
                        task.result()
                        target["states"][position] = "ready"
                    except BaseException as exc:
                        target["states"][position] = "failed"
                        target["errors"][position] = f"{type(exc).__name__}: {exc}"

            future.add_done_callback(done)

    def plan_batches(self, request):
        if not self.tiers.settings()["enabled"]:
            raise ValueError("Enable multi-tier batching before planning")
        if (not isinstance(request, dict) or set(request) != {"batches"}
                or not isinstance(request["batches"], list)
                or not 1 <= len(request["batches"]) <= 1024):
            raise ValueError("Supply 1 to 1024 batches")
        batches = request["batches"]
        if not all(isinstance(batch, dict) and isinstance(batch.get("trajectory"), str)
                   for batch in batches):
            raise ValueError("Each batch needs a trajectory")
        plan_id = uuid4().hex
        plan = {"id": plan_id, "batches": batches, "consumed": 0, "submitted": 0,
                "states": ["queued"] * len(batches), "futures": {},
                "errors": {}}
        with self._plan_lock:
            self._plans[plan_id] = plan
            self._schedule(plan)
        return self.plan_status(plan_id)

    def plan_status(self, plan_id):
        with self._plan_lock:
            plan = self._plans.get(plan_id)
            if plan is None:
                raise KeyError("Unknown batch plan")
            return {"plan_id": plan_id, "total": len(plan["batches"]),
                    "consumed": plan["consumed"], "submitted": plan["submitted"],
                    "states": list(plan["states"]), "errors": dict(plan["errors"])}

    def cancel_plan(self, plan_id):
        with self._plan_lock:
            plan = self._plans.pop(plan_id, None)
            if plan is None:
                raise KeyError("Unknown batch plan")
            for index, task in plan["futures"].items():
                if task.cancel():
                    plan["states"][index] = "cancelled"
            return {"cancelled": True, "plan_id": plan_id}

    def get(self, request):
        if not isinstance(request, dict):
            return super().get(request)
        body = dict(request)
        plan_id = body.pop("plan_id", None)
        plan_index = body.pop("plan_index", None)
        if plan_id is not None or plan_index is not None:
            with self._plan_lock:
                plan = self._plans.get(plan_id)
                if plan is None or plan_index != plan["consumed"]:
                    raise ValueError("Unknown plan or out-of-order learning batch")
                expected = plan["batches"][plan_index]
                if body != expected:
                    raise ValueError("Batch request differs from its planned contents")
                if plan["states"][plan_index] == "failed":
                    raise RuntimeError(plan["errors"][plan_index])
        with self.tiers.hold_request(body, self.workspace):
            answer = super().get(body)
        if plan_id is not None:
            with self._plan_lock:
                plan = self._plans.get(plan_id)
                if plan is not None and plan["consumed"] == plan_index:
                    plan["consumed"] += 1
                    plan["states"][plan_index] = "consumed"
                    self._schedule(plan)
        return answer

    def close(self):
        self._pool.shutdown(wait=True, cancel_futures=True)
        self.cache.close()
