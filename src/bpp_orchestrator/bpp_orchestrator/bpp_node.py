from __future__ import annotations

import copy
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Deque, List, Optional

import rclpy
from rclpy.node import Node

from shopee_interfaces.msg import ProductInfo, Sequence
from shopee_interfaces.srv import PackeeMainStartMTC, PackeeVisionBppStart

from .pack_planner import PlannedPose, ProductSpec, plan_sequences, visualize_layout

UNIT_SCALE = 0.001  # mm dimensions scaled down by an extra 0.1 factor for Packee


@dataclass
class BppJob:
    robot_id: int
    order_id: int
    products: List[ProductInfo]


class BppOrchestrator(Node):
    """BPP 요청을 받아 계산 후 결과를 별도 서비스로 전달한다."""

    def __init__(self) -> None:
        super().__init__("bpp_orchestrator")

        self._job_queue: Deque[BppJob] = deque()
        self._planner_future: Optional[Future] = None
        self._completion_future: Optional[Future] = None
        self._pending_job: Optional[BppJob] = None
        self._pending_sequences: Optional[List[Sequence]] = None
        self._latest_plan: Optional[List[PlannedPose]] = None

        self._pool = ThreadPoolExecutor(max_workers=2)
        self._enable_local_complete = (
            self.declare_parameter("enable_local_complete", True)
            .get_parameter_value()
            .bool_value
        )
        self._enable_layout_plot = (
            self.declare_parameter("enable_layout_plot", False)
            .get_parameter_value()
            .bool_value
        )

        self._start_service = self.create_service(
            PackeeVisionBppStart, "/packee/vision/bpp_start", self._handle_start_request
        )
        self._complete_client = self.create_client(
            PackeeMainStartMTC, "/packee/vision/bpp_complete"
        )
        self._timer = self.create_timer(0.2, self._poll_jobs)

        self.get_logger().info("BPP 오케스트레이터 노드를 시작했습니다.")

    def _handle_start_request(
        self,
        request: PackeeVisionBppStart.Request,
        response: PackeeVisionBppStart.Response,
    ) -> PackeeVisionBppStart.Response:
        job = BppJob(
            robot_id=request.robot_id,
            order_id=request.order_id,
            products=copy.deepcopy(request.products),
        )
        self._job_queue.append(job)
        self.get_logger().info(
            f"BPP 작업 수신 robot={job.robot_id} order={job.order_id} (대기열={len(self._job_queue)})"
        )
        response.success = True
        response.message = "BPP 계산 요청을 접수했습니다."
        return response

    def _poll_jobs(self) -> None:
        self._check_planner_future()
        self._check_completion_future()
        self._try_dispatch_completion()
        self._start_next_job()

    def _start_next_job(self) -> None:
        if self._planner_future or self._completion_future:
            return
        if not self._job_queue:
            return

        job = self._job_queue.popleft()
        specs = [_spec_from_msg(product) for product in job.products]
        self._planner_future = self._pool.submit(plan_sequences, specs)
        self._pending_job = job
        self.get_logger().info(
            f"BPP 계산 시작 robot={job.robot_id} order={job.order_id} (제품 수={len(specs)})"
        )

    def _check_planner_future(self) -> None:
        if not self._planner_future:
            return
        if not self._planner_future.done():
            return

        try:
            plan_result = self._planner_future.result()
            self._latest_plan = plan_result.poses
            self._pending_sequences = [_pose_to_msg(pose) for pose in plan_result.poses]
            finished_order = self._pending_job.order_id if self._pending_job else -1
            finished_seq = len(self._pending_sequences or [])
            self.get_logger().info(
                f"BPP 계산 완료 order={finished_order} (시퀀스={finished_seq})"
            )
            if self._enable_layout_plot:
                try:
                    plotted = visualize_layout(
                        plan_result, title=f"Order {finished_order} BPP Layout"
                    )
                    if not plotted:
                        self.get_logger().warn("배치 시각화를 표시하지 못했습니다.")
                except Exception as exc:  # pylint: disable=broad-except
                    self.get_logger().warn(f"배치 시각화 중 오류가 발생했습니다: {exc}")
        except Exception as exc:
            self.get_logger().error(f"BPP 계산에 실패했습니다: {exc}")
            self._pending_job = None
            self._pending_sequences = None
            self._latest_plan = None

        self._planner_future = None

    def _try_dispatch_completion(self) -> None:
        if self._completion_future or not self._pending_sequences or not self._pending_job:
            return

        if not self._complete_client.service_is_ready():
            ready = self._complete_client.wait_for_service(timeout_sec=0.1)
            if not ready:
                if self._enable_local_complete:
                    self._handle_local_completion()
                else:
                    self.get_logger().warn("bpp_complete 서비스 대기 중...")
                return

        request = PackeeMainStartMTC.Request()
        request.robot_id = self._pending_job.robot_id
        request.order_id = self._pending_job.order_id
        request.sequences = self._pending_sequences
        self._completion_future = self._complete_client.call_async(request)
        self.get_logger().info(
            f"BPP 결과 전송 중 robot={request.robot_id} order={request.order_id} "
            f"(시퀀스={len(request.sequences)})"
        )

    def _check_completion_future(self) -> None:
        if not self._completion_future:
            return
        if not self._completion_future.done():
            return

        try:
            result = self._completion_future.result()
            msg = getattr(result, "message", "")
            self.get_logger().info(
                f"bpp_complete 응답: success={result.success} message={msg}"
            )
        except Exception as exc:
            self.get_logger().error(f"bpp_complete 서비스 호출 실패: {exc}")

        self._completion_future = None
        self._pending_job = None
        self._pending_sequences = None
        self._latest_plan = None

    def _handle_local_completion(self) -> None:
        if not self._pending_job or not self._pending_sequences:
            return
        seq_count = len(self._pending_sequences)
        self.get_logger().info(
            f"외부 bpp_complete 서비스가 없어 로컬로 결과를 처리합니다. "
            f"robot={self._pending_job.robot_id} order={self._pending_job.order_id} "
            f"(시퀀스={seq_count})"
        )
        bin_lookup = (
            {pose.seq: pose.bin_index for pose in self._latest_plan} if self._latest_plan else {}
        )
        for seq in self._pending_sequences[:20]:
            self.get_logger().info(
                f"  bin={bin_lookup.get(seq.seq, 0)} seq={seq.seq:02d} id={seq.id} "
                f"pos=({seq.x:.3f}, {seq.y:.3f}, {seq.z:.3f}) "
                f"rot=({seq.rx:.2f}, {seq.ry:.2f}, {seq.rz:.2f})"
            )
        self._pending_sequences = None
        self._pending_job = None
        self._latest_plan = None

    def destroy_node(self) -> None:
        self._pool.shutdown(wait=False)
        super().destroy_node()


def _spec_from_msg(msg: ProductInfo) -> ProductSpec:
    return ProductSpec(
        product_id=int(msg.product_id),
        quantity=int(msg.quantity) if msg.quantity != 0 else 1,
        length=int(msg.length),
        width=int(msg.width),
        height=int(msg.height),
        weight=int(msg.weight),
        fragile=bool(msg.fragile),
    )


def _pose_to_msg(pose: PlannedPose) -> Sequence:
    msg = Sequence()
    msg.seq = pose.seq
    msg.id = pose.product_id
    msg.x = float(pose.x) * UNIT_SCALE
    msg.y = float(pose.y) * UNIT_SCALE
    msg.z = float(pose.z) * UNIT_SCALE
    msg.rx = float(pose.rx)
    msg.ry = float(pose.ry)
    msg.rz = float(pose.rz)
    return msg


def main() -> None:
    rclpy.init()
    node = BppOrchestrator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
