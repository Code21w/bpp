from __future__ import annotations

import itertools
import logging
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

_RECTPACK_TRIED = False
_NEW_PACKER = None
_OPEN_FIGURES: List["matplotlib.figure.Figure"] = []
_DEFAULT_RY = 1.5707
_ROTATE_Z_90 = 1.5707
_GRIPPER_LIMIT = 39
_BIN_LENGTH = 170.0
_BIN_WIDTH = 200.0
# margins are expressed in the same units as product specs (e.g., cm)
_MARGIN_X = 10.0  # 0.25 m after scaling
_MARGIN_Y = 10.0  # 0.30 m after scaling


@dataclass(frozen=True)
class ProductSpec:
    """원시 ProductInfo 데이터를 표현하는 내부 구조."""

    product_id: int
    quantity: int
    length: int
    width: int
    height: int
    weight: int
    fragile: bool


@dataclass(frozen=True)
class _ProductInstance:
    """실제 배치 단위(수량 확장 완료)."""

    product_id: int
    length: int
    width: int
    height: int
    fragile: bool


@dataclass(frozen=True)
class PlannedPose:
    """시퀀스 전송을 위한 6D pose."""

    seq: int
    product_id: int
    bin_index: int
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float


@dataclass(frozen=True)
class Placement:
    bin_index: int
    instance: _ProductInstance
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class PlanningResult:
    poses: List[PlannedPose]
    placements: List[Placement]
    bin_length: float
    bin_width: float
    bin_count: int


def plan_sequences(products: Sequence[ProductSpec]) -> PlanningResult:
    """입력 제품을 받아 배치 및 시퀀스를 계산한다."""

    instances = _expand_products(products)
    if not instances:
        return PlanningResult([], [], 0.0, 0.0)

    plan = _plan_with_rectpack(instances)
    if plan is None:
        plan = _plan_with_grid(instances)

    placements, bin_length, bin_width, bin_count = plan

    planned: List[PlannedPose] = []
    for seq_idx, placement in enumerate(placements, start=1):
        inst = placement.instance
        x = placement.x + inst.length / 2.0
        y = placement.y + inst.width / 2.0
        z = inst.height / 2.0
        planned.append(
            PlannedPose(
                seq=seq_idx,
                product_id=inst.product_id,
                bin_index=placement.bin_index,
                x=x,
                y=y,
                z=z,
                rx=0.0,
                ry=_DEFAULT_RY,
                rz=0.0 if inst.length >= inst.width else _ROTATE_Z_90,
            )
        )
    planned.sort(key=lambda pose: (abs(pose.z), pose.seq))
    planned = [
        replace(
            pose,
            seq=idx,
        )
        for idx, pose in enumerate(planned, start=1)
    ]
    return PlanningResult(planned, placements, bin_length, bin_width, bin_count)


def _expand_products(products: Sequence[ProductSpec]) -> List[_ProductInstance]:
    expanded: List[_ProductInstance] = []
    for spec in products:
        qty = spec.quantity if spec.quantity > 0 else 1
        for _ in range(qty):
            length, width, height = _orient_product(spec)
            expanded.append(
                _ProductInstance(
                    product_id=spec.product_id,
                    length=length,
                    width=width,
                    height=height,
                    fragile=spec.fragile,
                )
            )
    expanded.sort(key=lambda item: item.length * item.width, reverse=True)
    return expanded


def _plan_with_rectpack(
    instances: Sequence[_ProductInstance],
) -> Optional[Tuple[List[Placement], float, float, int]]:
    new_packer = _resolve_new_packer()
    if new_packer is None:
        return None

    bin_length_eff, bin_width_eff = _bin_dimensions(effective=True)
    bin_length, bin_width = _bin_dimensions()
    if bin_length_eff <= 0 or bin_width_eff <= 0:
        LOGGER.error("Effective bin size is zero; margins too large.")
        return None
    remaining = list(range(len(instances)))
    placements: List[Placement] = []
    bin_index = 0

    while remaining:
        packer = new_packer(rotation=False)
        for rid in remaining:
            inst = instances[rid]
            packer.add_rect(int(inst.length), int(inst.width), rid=rid)

        packer.add_bin(int(bin_length_eff), int(bin_width_eff))
        packer.pack()

        placed_this_bin: set[int] = set()
        for _, x, y, w, h, rid in packer.rect_list():
            if rid not in remaining:
                continue
            inst = instances[rid]
            placements.append(
                Placement(
                    bin_index=bin_index,
                    instance=inst,
                    x=float(x) + _MARGIN_X,
                    y=float(y) + _MARGIN_Y,
                    w=float(w),
                    h=float(h),
                )
            )
            placed_this_bin.add(rid)

        if not placed_this_bin:
            LOGGER.error("rectpack가 bin %d에 아무 사각형도 배치하지 못했습니다.", bin_index)
            break

        remaining = [rid for rid in remaining if rid not in placed_this_bin]
        bin_index += 1

    if remaining:
        LOGGER.warning("일부 사각형이 배치되지 않아 그리드 방식으로 전환합니다.")
        return None

    placements.sort(key=lambda data: (data.bin_index, data.y, data.x))
    return placements, bin_length, bin_width, max(bin_index, 1)


def _plan_with_grid(
    instances: Sequence[_ProductInstance],
) -> Tuple[List[Placement], float, float, int]:
    bin_length_eff, bin_width_eff = _bin_dimensions(effective=True)
    bin_length, bin_width = _bin_dimensions()
    if bin_length_eff <= 0 or bin_width_eff <= 0:
        raise ValueError("Effective bin size is zero; margins too large.")
    placements: List[Placement] = []
    cursor_x = 0.0
    cursor_y = 0.0
    row_height = 0.0
    bin_index = 0

    for inst in instances:
        if inst.length > bin_length_eff or inst.width > bin_width_eff:
            LOGGER.warning(
                "제품 %d가 bin 유효 영역(%.1fx%.1f)을 초과합니다. 길이=%.1f 너비=%.1f",
                inst.product_id,
                bin_length_eff,
                bin_width_eff,
                float(inst.length),
                float(inst.width),
            )

        if cursor_x + inst.length > bin_length_eff:
            cursor_x = 0.0
            cursor_y += row_height
            row_height = 0.0

        if cursor_y + inst.width > bin_width_eff:
            bin_index += 1
            cursor_x = 0.0
            cursor_y = 0.0
            row_height = 0.0

        placements.append(
            Placement(
                bin_index=bin_index,
                instance=inst,
                x=float(cursor_x) + _MARGIN_X,
                y=float(cursor_y) + _MARGIN_Y,
                w=float(inst.length),
                h=float(inst.width),
            )
        )
        cursor_x += inst.length
        row_height = max(row_height, float(inst.width))

    placements.sort(key=lambda data: (data.bin_index, data.y, data.x))
    return placements, bin_length, bin_width, bin_index + 1


def _bin_dimensions(effective: bool = False) -> Tuple[float, float]:
    if effective:
        length = max(_BIN_LENGTH - 2 * _MARGIN_X, 0.0)
        width = max(_BIN_WIDTH - 2 * _MARGIN_Y, 0.0)
        return length, width
    return _BIN_LENGTH, _BIN_WIDTH


def _orient_product(spec: ProductSpec) -> Tuple[int, int, int]:
    dims = [
        max(spec.length, 1),
        max(spec.width, 1),
        max(spec.height, 1),
    ]
    orientations: List[Tuple[Tuple[int, int, int], int, int, int]] = []
    eff_length, eff_width = _bin_dimensions(effective=True)
    for idx, vertical in enumerate(dims):
        horizontal = [dims[j] for j in range(3) if j != idx]
        for top in itertools.permutations(horizontal):
            length, width = top
            fits_gripper = min(length, width) <= _GRIPPER_LIMIT
            fits_bin = length <= eff_length and width <= eff_width
            orientations.append(
                (
                    (length, width, vertical),
                    0 if fits_gripper else 1,
                    0 if fits_bin else 1,
                    length * width,
                )
            )

    orientations.sort(key=lambda item: (item[1], item[2], item[3]))
    best, gripper_penalty, bin_penalty, _ = orientations[0]

    if gripper_penalty:
        LOGGER.warning(
            "제품 %d에서 그리퍼 제약(<=%d)을 만족하는 면을 찾지 못했습니다. 가장 작은 면을 사용합니다.",
            spec.product_id,
            _GRIPPER_LIMIT,
        )
    if bin_penalty:
        LOGGER.warning(
            "제품 %d이 bin 유효 영역(%.1fx%.1f)을 초과합니다. 선택된 면: %.1fx%.1f",
            spec.product_id,
            eff_length,
            eff_width,
            best[0],
            best[1],
        )

    return best


def _resolve_new_packer():
    global _RECTPACK_TRIED, _NEW_PACKER
    if _RECTPACK_TRIED:
        return _NEW_PACKER

    _RECTPACK_TRIED = True
    try:
        from rectpack import newPacker

        _NEW_PACKER = newPacker
        return _NEW_PACKER
    except ImportError:
        pass

    for root in _candidate_roots():
        rectpack_dir = root / "rectpack"
        if not rectpack_dir.exists():
            continue
        if str(rectpack_dir) not in sys.path:
            sys.path.append(str(rectpack_dir))
        try:
            from rectpack import newPacker

            _NEW_PACKER = newPacker
            return _NEW_PACKER
        except ImportError:
            continue

    LOGGER.warning("rectpack 모듈을 찾을 수 없어 내부 그리드 배치 알고리즘을 사용합니다.")
    return None


def _candidate_roots() -> Iterable[Path]:
    env_root = os.environ.get("BPP_ALGO_ROOT")
    if env_root:
        yield Path(env_root).expanduser()
    home_root = Path.home() / "BPP--algorithm"
    yield home_root


def visualize_layout(result: PlanningResult, title: Optional[str] = None) -> bool:
    """matplotlib을 이용해 배치 결과를 평면 그래프로 시각화한다."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        LOGGER.warning("matplotlib을 불러올 수 없어 시각화를 건너뜁니다.")
        return False

    if not result.placements:
        LOGGER.info("시각화할 배치 데이터가 없습니다.")
        return False

    bins: dict[int, List[Placement]] = {}
    for placement in result.placements:
        bins.setdefault(placement.bin_index, []).append(placement)

    cmap = plt.cm.get_cmap("tab20")
    for bin_index in sorted(bins.keys()):
        fig, ax = plt.subplots()
        for idx, placement in enumerate(bins[bin_index]):
            color = cmap(idx % cmap.N)
            rect = Rectangle(
                (placement.x, placement.y),
                placement.w,
                placement.h,
                facecolor=color,
                edgecolor="black",
                alpha=0.6,
            )
            ax.add_patch(rect)
            ax.text(
                placement.x + placement.w / 2,
                placement.y + placement.h / 2,
                str(placement.instance.product_id),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

        ax.set_xlim(0, max(result.bin_length, 1))
        ax.set_ylim(0, max(result.bin_width, 1))
        ax.set_aspect("equal")
        ax.set_xlabel("Length (x)")
        ax.set_ylabel("Width (y)")
        bin_title = title or "BPP Layout"
        ax.set_title(f"{bin_title} - Bin {bin_index}")
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        if not plt.isinteractive():
            plt.ion()
        plt.show(block=False)
        plt.pause(0.1)
        _OPEN_FIGURES.append(fig)

    while len(_OPEN_FIGURES) > 5:
        old_fig = _OPEN_FIGURES.pop(0)
        plt.close(old_fig)
    return True
