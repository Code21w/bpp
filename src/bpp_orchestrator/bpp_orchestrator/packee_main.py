#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Packee 메인 오케스트레이터 노드."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from typing import Deque, Dict, Optional, Tuple

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future

from shopee_interfaces.msg import PackeePackingComplete
from shopee_interfaces.srv import (
    PackeeArmPackingComplete,
    PackeeMainStartMTC,
    PackeePackingStart,
    PackeeVisionBppStart,
)


@dataclass
class OrderContext:
    """단순 주문 상태 컨텍스트."""

    robot_id: int
    order_id: int
    products: Optional[list] = None
    sequences: Optional[list] = None


class PackeeMainNode(Node):
    """Packee 메인 노드: 포장→BPP→MTC 흐름 조율."""

    def __init__(self) -> None:
        super().__init__("packee_main_node")

        self._cb_group = ReentrantCallbackGroup()
        self._orders: Dict[Tuple[int, int], OrderContext] = {}
        self._pending_mtc: Deque[Tuple[int, int]] = deque()
        self._pending_mtc_keys: set[Tuple[int, int]] = set()
        self._mtc_future: Optional[Future] = None
        self._mtc_current: Optional[Tuple[int, int]] = None
        # 디버깅을 위한 BPP 완료 로그 토글 (나중에 파라미터로 끔)
        self._log_bpp_complete = (
            self.declare_parameter("log_bpp_complete_debug", True)
            .get_parameter_value()
            .bool_value
        )

        # 서비스 서버 등록
        self.packing_start_srv = self.create_service(
            PackeePackingStart,
            "/packee/packing/start",
            self.on_packing_start,
            callback_group=self._cb_group,
        )
        self.bpp_complete_srv = self.create_service(
            PackeeMainStartMTC,
            "/packee/vision/bpp_complete",
            self.on_bpp_complete,
            callback_group=self._cb_group,
        )
        self.mtc_finish_srv = self.create_service(
            PackeeArmPackingComplete,
            "/packee/mtc/finish",
            self.on_mtc_finish,
            callback_group=self._cb_group,
        )

        # 서비스 클라이언트 등록
        self.cli_bpp_start = self.create_client(
            PackeeVisionBppStart,
            "/packee/vision/bpp_start",
            callback_group=self._cb_group,
        )
        self.cli_mtc_start = self.create_client(
            PackeeMainStartMTC,
            "/packee/mtc/startmtc",
            callback_group=self._cb_group,
        )

        # 퍼블리셔
        self.pub_packing_complete = self.create_publisher(
            PackeePackingComplete,
            "/packee/packing_complete",
            10,
        )
        self._mtc_timer = self.create_timer(
            0.5, self._pump_mtc_queue, callback_group=self._cb_group
        )

        self.get_logger().info("[PackeeMain] ready.")

    # 1) /packee/packing/start -> /packee/vision/bpp_start
    def on_packing_start(
        self, req: PackeePackingStart.Request, resp: PackeePackingStart.Response
    ) -> PackeePackingStart.Response:
        rid, oid = int(req.robot_id), int(req.order_id)
        key = (rid, oid)
        self._orders[key] = OrderContext(robot_id=rid, order_id=oid, products=list(req.products))

        if not self.cli_bpp_start.wait_for_service(timeout_sec=2.0):
            resp.box_id = -1
            resp.success = False
            resp.message = "[PackeeMain] /packee/vision/bpp_start not available"
            self.get_logger().warn(resp.message)
            return resp

        bpp_req = PackeeVisionBppStart.Request()
        bpp_req.robot_id = rid
        bpp_req.order_id = oid
        bpp_req.products = req.products

        future: Future = self.cli_bpp_start.call_async(bpp_req)
        bpp_res = self._await_future(future, timeout_sec=5.0)
        if bpp_res is None:
            resp.box_id = -1
            resp.success = False
            resp.message = "[PackeeMain] BPP start timed out or no response"
            self.get_logger().error(resp.message)
            return resp
        resp.box_id = -1
        resp.success = bool(bpp_res.success)
        resp.message = (
            f"[PackeeMain] BPP started: {bpp_res.message}"
            if bpp_res.success
            else f"[PackeeMain] BPP failed to start: {bpp_res.message}"
        )
        log = self.get_logger().info if resp.success else self.get_logger().warn
        log(resp.message)
        return resp

    # 2) /packee/vision/bpp_complete -> /packee/mtc/startmtc
    def on_bpp_complete(
        self, req: PackeeMainStartMTC.Request, resp: PackeeMainStartMTC.Response
    ) -> PackeeMainStartMTC.Response:
        rid, oid = int(req.robot_id), int(req.order_id)
        key = (rid, oid)
        ctx = self._orders.get(key) or OrderContext(robot_id=rid, order_id=oid)
        ctx.sequences = list(req.sequences)
        self._orders[key] = ctx

        if self._log_bpp_complete:
            self._log_bpp_payload(rid, oid, ctx.sequences)

        self._queue_mtc_start(key)
        self._pump_mtc_queue()

        resp.success = True
        if self.cli_mtc_start.service_is_ready():
            resp.message = "[PackeeMain] MTC start scheduled"
            self.get_logger().info(
                f"[PackeeMain] MTC start scheduled (robot={rid} order={oid})"
            )
        else:
            resp.message = "[PackeeMain] MTC start queued (service not available)"
            self.get_logger().warn(
                f"[PackeeMain] MTC start queued (service not available) (robot={rid} order={oid})"
            )
        return resp

    def _queue_mtc_start(self, key: Tuple[int, int], force: bool = False) -> None:
        if not force and key in self._pending_mtc_keys:
            return
        self._pending_mtc.append(key)
        self._pending_mtc_keys.add(key)

    def _pump_mtc_queue(self) -> None:
        if self._mtc_future:
            if self._mtc_future.done():
                self._finalize_mtc_future()
            else:
                return

        if not self._pending_mtc:
            return
        if not self.cli_mtc_start.service_is_ready():
            return

        key = self._pending_mtc.popleft()
        self._pending_mtc_keys.discard(key)
        ctx = self._orders.get(key)
        if not ctx or not ctx.sequences:
            self.get_logger().warn(
                f"[PackeeMain] pending MTC start dropped (robot={key[0]} order={key[1]})"
            )
            return

        mtc_req = PackeeMainStartMTC.Request()
        mtc_req.robot_id = key[0]
        mtc_req.order_id = key[1]
        mtc_req.sequences = ctx.sequences

        self._mtc_current = key
        self._mtc_future = self.cli_mtc_start.call_async(mtc_req)

    def _finalize_mtc_future(self) -> None:
        if not self._mtc_future or not self._mtc_future.done():
            return
        key = self._mtc_current
        try:
            result = self._mtc_future.result()
            success = bool(getattr(result, "success", False))
            message = getattr(result, "message", "")
            if success:
                rid = key[0] if key else -1
                oid = key[1] if key else -1
                self.get_logger().info(
                    f"[PackeeMain] MTC start success robot={rid} order={oid} msg={message}"
                )
            else:
                rid = key[0] if key else -1
                oid = key[1] if key else -1
                self.get_logger().warn(
                    f"[PackeeMain] MTC start failed robot={rid} order={oid} msg={message}"
                )
                if key:
                    self._queue_mtc_start(key, force=True)
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().error(f"[PackeeMain] MTC start exception: {exc}")
            if key:
                self._queue_mtc_start(key, force=True)
        finally:
            self._mtc_future = None
            self._mtc_current = None

    def _log_bpp_payload(self, robot_id: int, order_id: int, sequences: list) -> None:
        count = len(sequences) if sequences else 0
        self.get_logger().info(
            f"[PackeeMain] BPP complete payload robot={robot_id} order={order_id} sequences={count}"
        )
        if not sequences:
            return
        preview = sequences[: min(5, count)]
        for seq in preview:
            self.get_logger().info(
                f"  seq={seq.seq:02d} id={seq.id} "
                f"pos=({seq.x:.3f}, {seq.y:.3f}, {seq.z:.3f}) "
                f"rot=({seq.rx:.2f}, {seq.ry:.2f}, {seq.rz:.2f})"
            )
        if count > len(preview):
            self.get_logger().info(f"  ... ({count - len(preview)} more sequences)")

    def _await_future(self, future: Future, timeout_sec: float) -> Optional[object]:
        event = threading.Event()

        def _wake(_):
            event.set()

        future.add_done_callback(_wake)
        if not future.done():
            if not event.wait(timeout_sec):
                return None
        if future.cancelled():
            return None
        if future.exception():
            self.get_logger().error(f"Service call failed: {future.exception()}")
            return None
        return future.result()

    # 3) /packee/mtc/finish -> /packee/packing_complete publish
    def on_mtc_finish(
        self, req: PackeeArmPackingComplete.Request, resp: PackeeArmPackingComplete.Response
    ) -> PackeeArmPackingComplete.Response:
        rid, oid = int(req.robot_id), int(req.order_id)
        key = (rid, oid)
        ctx = self._orders.get(key) or OrderContext(robot_id=rid, order_id=oid)
        ctx.sequences = list(req.sequences)
        self._orders[key] = ctx

        msg = PackeePackingComplete()
        msg.robot_id = rid
        msg.order_id = oid
        msg.success = bool(req.success)
        msg.packed_items = len(req.sequences) if req.sequences is not None else 0
        msg.message = f"[PackeeMain] MTC finish: {req.message}"
        self.pub_packing_complete.publish(msg)

        resp.success = True
        resp.message = "[PackeeMain] finish received & packing_complete published"
        self.get_logger().info(
            f"{resp.message} (robot={rid}, order={oid}, items={msg.packed_items}, ok={msg.success})"
        )
        return resp


def main() -> None:
    rclpy.init()
    node = PackeeMainNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
