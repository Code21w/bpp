from __future__ import annotations

from typing import List

import rclpy
from rclpy.node import Node

from shopee_interfaces.msg import Sequence
from shopee_interfaces.srv import PackeeMainStartMTC


class BppCompleteMockServer(Node):
    """PackeeMainStartMTC 서비스를 흉내 내며 결과 내용을 로그로 확인한다."""

    def __init__(self) -> None:
        super().__init__("bpp_complete_mock_server")
        self._service = self.create_service(
            PackeeMainStartMTC, "/packee/vision/bpp_complete", self._handle_request
        )
        self.get_logger().info("bpp_complete 더미 서버가 준비되었습니다.")

    def _handle_request(
        self,
        request: PackeeMainStartMTC.Request,
        response: PackeeMainStartMTC.Response,
    ) -> PackeeMainStartMTC.Response:
        self.get_logger().info(
            "bpp_complete 요청 수신 robot=%d order=%d (시퀀스=%d)",
            request.robot_id,
            request.order_id,
            len(request.sequences),
        )
        for seq in _format_sequences(request.sequences):
            self.get_logger().info(seq)

        response.success = True
        response.message = "더미 서버가 시퀀스를 수신했습니다."
        return response


def _format_sequences(sequences: List[Sequence]) -> List[str]:
    lines: List[str] = []
    for seq in sequences:
        lines.append(
            f"  seq={seq.seq:02d} id={seq.id} "
            f"pos=({seq.x:.1f}, {seq.y:.1f}, {seq.z:.1f}) "
            f"rot=({seq.rx:.2f}, {seq.ry:.2f}, {seq.rz:.2f})"
        )
    return lines[:20]  # 로그 폭주 방지를 위해 상위 20개까지만 표시


def main() -> None:
    rclpy.init()
    node = BppCompleteMockServer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
