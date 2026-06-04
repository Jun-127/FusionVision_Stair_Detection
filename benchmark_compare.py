# benchmark_compare.py
#
# FusionVision (ROI 방식) vs Full Frame (전체 이미지 3D 매핑) 비교 벤치마크
#
# 매 프레임마다 동일한 depth/RGB 데이터를 두 방식으로 처리하여
# 처리 시간, FPS, 포인트클라우드 밀도, 계단 측정 정확도를 비교한다.
#
# [A] FusionVision  : YOLO → FastSAM → ROI 마스크 영역만 포인트클라우드 생성
# [B] Full Frame    : YOLO 없이 depth 전체를 포인트클라우드로 변환
#
# 실행:
#   python benchmark_compare.py
#   python benchmark_compare.py --model exp5 --frames 200
#   python benchmark_compare.py --no-display        # 화면 없이 순수 성능만
#
# 결과:
#   benchmark_results/compare_{model}_{timestamp}.csv
#   (터미널에 최종 요약도 출력)

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import time
import csv
import os
from datetime import datetime
from ultralytics import YOLO, FastSAM

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
EXP5_WEIGHTS    = r'C:\Users\Jaejun_Yeo\Documents\detect_stair\runs\detect\exp5_adamw_n\weights\best.pt'
EXP6_WEIGHTS    = r'C:\Users\Jaejun_Yeo\Documents\detect_stair\runs\detect\exp6_adamw_m\weights\best.pt'
FASTSAM_WEIGHTS = r'C:\Users\Jaejun_Yeo\Documents\detect_stair\FastSAM-s.pt'
RESULT_DIR      = r'C:\Users\Jaejun_Yeo\Documents\detect_stair\benchmark_results'

# ─── 공통 파라미터 ────────────────────────────────────────────────────────────
CONF_THRESHOLD  = 0.4
VOXEL_SIZE      = 0.02      # 두 방식 동일하게 적용
NB_NEIGHBORS    = 20
STD_RATIO       = 2.0
DISPLAY_W       = 640
DISPLAY_H       = 480
INFER_SIZE      = 320
BBOX_PADDING    = 30
CAM_HEIGHT      = 0.13
TILT_CORRECTION = 1.0
DEPTH_MAX_M     = 5.0       # 유효 depth 상한 (m)

# ─── 벤치마크 설정 ────────────────────────────────────────────────────────────
BENCHMARK_FRAMES = 300
WARMUP_FRAMES    = 30
DISPLAY_ENABLED  = True
PRINT_EVERY      = 30


# ═══════════════════════════════════════════════════════════════════════════════
#  모델 / 카메라 초기화
# ═══════════════════════════════════════════════════════════════════════════════

def load_yolo(model_name: str):
    weights = EXP5_WEIGHTS if model_name == 'exp5' else EXP6_WEIGHTS
    print(f"[INFO] Loading YOLO : {model_name}")
    m = YOLO(weights)
    m.to('cuda')
    return m


def load_fastsam():
    print("[INFO] Loading FastSAM-s ...")
    m = FastSAM(FASTSAM_WEIGHTS)
    m.to('cuda')
    return m


def init_realsense():
    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    profile  = pipeline.start(cfg)
    align    = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()

    print("[INFO] Warming up camera ...")
    for _ in range(30):
        pipeline.wait_for_frames(timeout_ms=10000)

    print(f"[INFO] RealSense ready. depth_scale={depth_scale:.6f} m/unit")
    return pipeline, align, depth_scale


# ═══════════════════════════════════════════════════════════════════════════════
#  포인트클라우드 공통 유틸
# ═══════════════════════════════════════════════════════════════════════════════

def _depth_to_points(depth_image, mask_bool, intrinsics, depth_scale):
    """
    mask_bool : H×W bool 배열 — True 인 픽셀만 3D 변환
    반환      : (N,3) float32 배열, 또는 None
    """
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy

    ys, xs = np.where(mask_bool)
    if len(ys) == 0:
        return None

    zs    = depth_image[ys, xs] * depth_scale
    valid = (zs > 0) & (zs < DEPTH_MAX_M)
    ys, xs, zs = ys[valid], xs[valid], zs[valid]
    if len(zs) == 0:
        return None

    return np.stack([
        (xs - cx) * zs / fx,
        (ys - cy) * zs / fy,
        zs
    ], axis=1).astype(np.float32)


def _make_pcd(points):
    """(N,3) → open3d PointCloud (컬러는 depth-JET)."""
    if points is None or len(points) == 0:
        return None
    zs     = points[:, 2]
    z_min, z_max = zs.min(), zs.max()
    z_norm = (zs - z_min) / (z_max - z_min + 1e-9)
    z_u8   = (z_norm * 255).astype(np.uint8)
    cmap   = cv2.applyColorMap(z_u8.reshape(-1, 1), cv2.COLORMAP_JET)
    colors = cmap.reshape(-1, 3)[:, ::-1] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def postprocess(pcd):
    """다운샘플 → 디노이즈. (pcd_down, pcd_clean) 반환."""
    pcd_down     = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    pcd_clean, _ = pcd_down.remove_statistical_outlier(
        nb_neighbors=NB_NEIGHBORS, std_ratio=STD_RATIO)
    return pcd_down, pcd_clean


# ═══════════════════════════════════════════════════════════════════════════════
#  계단 측정 (두 방식 공통)
# ═══════════════════════════════════════════════════════════════════════════════

def measure_stairs(pcd):
    if pcd is None or len(pcd.points) < 50:
        return None

    pts    = np.asarray(pcd.points)
    x_vals = pts[:, 0]
    y_vals = pts[:, 1]
    z_vals = pts[:, 2]

    z_min, z_max = z_vals.min(), z_vals.max()
    z_range      = z_max - z_min
    if z_range < 0.05:
        return None

    bin_size    = 0.05
    n_bins      = max(5, int(z_range / bin_size))
    hist, edges = np.histogram(z_vals, bins=n_bins)
    smooth      = np.convolve(hist, np.ones(3)/3, mode='same')
    thr         = smooth.max() * 0.2

    raw_peaks = [
        i for i in range(1, len(smooth)-1)
        if smooth[i] > smooth[i-1]
        and smooth[i] > smooth[i+1]
        and smooth[i] > thr
    ]
    peaks, last = [], -999
    for p in raw_peaks:
        if p - last >= 2:
            peaks.append(p)
            last = p

    n_stairs = len(peaks)
    if n_stairs < 2:
        return None

    peak_zs = sorted([(edges[p]+edges[p+1])/2 for p in peaks])
    margin  = bin_size * 0.6

    heights = []
    for i in range(len(peak_zs)-1):
        m1 = np.abs(z_vals - peak_zs[i])   < margin
        m2 = np.abs(z_vals - peak_zs[i+1]) < margin
        if m1.sum() > 5 and m2.sum() > 5:
            h = abs(y_vals[m2].mean() - y_vals[m1].mean())
            if h > 0.03:
                heights.append(h)

    if not heights:
        return None

    avg_h   = float(np.mean(heights)) * TILT_CORRECTION
    depth_  = abs(peak_zs[1] - peak_zs[0])
    m_first = np.abs(z_vals - peak_zs[0]) < margin
    length_ = 0.0
    dist_   = None

    if m_first.sum() > 5:
        length_ = float(x_vals[m_first].max() - x_vals[m_first].min())
        fz      = float(z_vals[m_first].min())
        if fz > CAM_HEIGHT:
            dist_ = float(np.sqrt(max(0.0, fz**2 - CAM_HEIGHT**2)))

    return {
        'n_stairs'    : n_stairs,
        'height_cm'   : avg_h  * 100,
        'depth_cm'    : depth_ * 100,
        'length_cm'   : length_* 100,
        'total_h_cm'  : avg_h  * n_stairs * 100,
        'dist_cm'     : dist_  * 100 if dist_ else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  [A] FusionVision 방식 — 한 프레임 처리
# ═══════════════════════════════════════════════════════════════════════════════

def process_fusionvision(color_image, depth_image, intrinsics, depth_scale,
                         yolo_model, fastsam_model):
    """
    반환: dict {
        t_yolo, t_fastsam, t_pc_create, t_pc_post, t_measure, t_total (모두 ms),
        yolo_detected, fastsam_ok,
        pts_raw, pts_down, pts_den,
        stair (measure_stairs 결과 or None),
        mask_vis (H×W uint8, 시각화용)
    }
    """
    ih, iw = color_image.shape[:2]
    result = dict(
        t_yolo=0, t_fastsam=0, t_pc_create=0,
        t_pc_post=0, t_measure=0, t_total=0,
        yolo_detected=False, fastsam_ok=False,
        pts_raw=0, pts_down=0, pts_den=0,
        stair=None,
        mask_vis=np.zeros((ih, iw), dtype=np.uint8),
    )

    t_start = time.perf_counter()

    # (1) YOLO
    t0 = time.perf_counter()
    yolo_out = yolo_model(color_image, conf=CONF_THRESHOLD,
                          imgsz=INFER_SIZE, verbose=False)[0]
    boxes    = yolo_out.boxes
    result['t_yolo'] = (time.perf_counter() - t0) * 1000

    if boxes is None or len(boxes) == 0:
        result['t_total'] = (time.perf_counter() - t_start) * 1000
        return result

    result['yolo_detected'] = True
    merged_raw = o3d.geometry.PointCloud()

    for box in boxes:
        rx1, ry1, rx2, ry2 = map(int, box.xyxy[0])
        rx1 = max(0, rx1);  ry1 = max(0, ry1)
        rx2 = min(iw, rx2); ry2 = min(ih, ry2)
        px1 = max(0,  rx1 - BBOX_PADDING)
        py1 = max(0,  ry1 - BBOX_PADDING)
        px2 = min(iw, rx2 + BBOX_PADDING)
        py2 = min(ih, ry2 + BBOX_PADDING)

        # (2) FastSAM
        t1 = time.perf_counter()
        sam_out = fastsam_model(
            color_image,
            bboxes=[[px1, py1, px2, py2]],
            imgsz=INFER_SIZE, verbose=False
        )
        result['t_fastsam'] += (time.perf_counter() - t1) * 1000

        mask = None
        if (sam_out and sam_out[0].masks is not None
                and len(sam_out[0].masks.data) > 0):
            mask = sam_out[0].masks.data[0].cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)
            clip = np.zeros_like(mask)
            clip[ry1:ry2, rx1:rx2] = 1
            mask = mask & clip
            if mask.sum() == 0:
                mask = None

        if mask is None:
            continue

        result['fastsam_ok'] = True
        result['mask_vis']   = cv2.bitwise_or(result['mask_vis'], mask * 255)

        # (3) PC 생성
        t2 = time.perf_counter()
        pts = _depth_to_points(depth_image, mask.astype(bool),
                               intrinsics, depth_scale)
        result['t_pc_create'] += (time.perf_counter() - t2) * 1000

        pcd = _make_pcd(pts)
        if pcd and len(pcd.points) > 10:
            result['pts_raw'] += len(pcd.points)
            merged_raw += pcd

    # (4) 후처리
    merged_clean = o3d.geometry.PointCloud()
    if len(merged_raw.points) > 50:
        t3 = time.perf_counter()
        pcd_down, pcd_clean = postprocess(merged_raw)
        result['t_pc_post'] = (time.perf_counter() - t3) * 1000
        result['pts_down']  = len(pcd_down.points)
        result['pts_den']   = len(pcd_clean.points)
        merged_clean        = pcd_clean

    # (5) 계단 측정
    if len(merged_clean.points) > 50:
        t4 = time.perf_counter()
        result['stair'] = measure_stairs(merged_clean)
        result['t_measure'] = (time.perf_counter() - t4) * 1000

    result['t_total'] = (time.perf_counter() - t_start) * 1000
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  [B] Full Frame 방식 — 한 프레임 처리
# ═══════════════════════════════════════════════════════════════════════════════

def process_fullframe(color_image, depth_image, intrinsics, depth_scale):
    """
    YOLO / FastSAM 없이 전체 depth 이미지를 포인트클라우드로 변환.
    반환: dict {
        t_pc_create, t_pc_post, t_measure, t_total (ms),
        pts_raw, pts_down, pts_den,
        stair (or None)
    }
    """
    ih, iw = depth_image.shape
    result  = dict(
        t_pc_create=0, t_pc_post=0, t_measure=0, t_total=0,
        pts_raw=0, pts_down=0, pts_den=0,
        stair=None,
    )

    t_start = time.perf_counter()

    # (1) 전체 픽셀 → 포인트클라우드
    t0          = time.perf_counter()
    full_mask   = np.ones((ih, iw), dtype=bool)
    pts         = _depth_to_points(depth_image, full_mask, intrinsics, depth_scale)
    result['t_pc_create'] = (time.perf_counter() - t0) * 1000

    pcd = _make_pcd(pts)
    if pcd is None or len(pcd.points) == 0:
        result['t_total'] = (time.perf_counter() - t_start) * 1000
        return result

    result['pts_raw'] = len(pcd.points)

    # (2) 후처리 (동일 파라미터)
    t1 = time.perf_counter()
    pcd_down, pcd_clean = postprocess(pcd)
    result['t_pc_post'] = (time.perf_counter() - t1) * 1000
    result['pts_down']  = len(pcd_down.points)
    result['pts_den']   = len(pcd_clean.points)

    # (3) 계단 측정
    if len(pcd_clean.points) > 50:
        t2 = time.perf_counter()
        result['stair'] = measure_stairs(pcd_clean)
        result['t_measure'] = (time.perf_counter() - t2) * 1000

    result['t_total'] = (time.perf_counter() - t_start) * 1000
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV
# ═══════════════════════════════════════════════════════════════════════════════

# 컬럼: A_ 접두사 = FusionVision, B_ 접두사 = Full Frame
FIELDNAMES = [
    'frame_idx', 'fps',
    # ── [A] 시간 ──
    'A_t_yolo_ms', 'A_t_fastsam_ms', 'A_t_pc_create_ms',
    'A_t_pc_post_ms', 'A_t_measure_ms', 'A_t_total_ms',
    # ── [A] 탐지 ──
    'A_yolo_detected', 'A_fastsam_ok',
    # ── [A] 밀도 ──
    'A_pts_raw', 'A_pts_down', 'A_pts_den',
    # ── [A] 계단 측정 ──
    'A_n_stairs', 'A_height_cm', 'A_depth_cm',
    'A_length_cm', 'A_total_h_cm', 'A_dist_cm',
    # ── [B] 시간 ──
    'B_t_pc_create_ms', 'B_t_pc_post_ms',
    'B_t_measure_ms', 'B_t_total_ms',
    # ── [B] 밀도 ──
    'B_pts_raw', 'B_pts_down', 'B_pts_den',
    # ── [B] 계단 측정 ──
    'B_n_stairs', 'B_height_cm', 'B_depth_cm',
    'B_length_cm', 'B_total_h_cm', 'B_dist_cm',
    # ── 비교 지표 ──
    'speedup_ratio',        # B_total / A_total (클수록 A가 빠름)
    'pts_reduction_ratio',  # 1 - A_pts_den/B_pts_den (포인트 절감률)
]


def _stair_cols(stair, prefix):
    d = {}
    keys = ['n_stairs','height_cm','depth_cm','length_cm','total_h_cm','dist_cm']
    if stair:
        for k in keys:
            v = stair.get(k)
            d[f'{prefix}{k}'] = round(v, 2) if v is not None else ''
    else:
        for k in keys:
            d[f'{prefix}{k}'] = ''
    return d


def build_row(frame_idx, fps, A, B) -> dict:
    r = {'frame_idx': frame_idx, 'fps': round(fps, 2)}

    # A
    r['A_t_yolo_ms']      = round(A['t_yolo'],      3)
    r['A_t_fastsam_ms']   = round(A['t_fastsam'],   3)
    r['A_t_pc_create_ms'] = round(A['t_pc_create'], 3)
    r['A_t_pc_post_ms']   = round(A['t_pc_post'],   3)
    r['A_t_measure_ms']   = round(A['t_measure'],   3)
    r['A_t_total_ms']     = round(A['t_total'],     3)
    r['A_yolo_detected']  = int(A['yolo_detected'])
    r['A_fastsam_ok']     = int(A['fastsam_ok'])
    r['A_pts_raw']        = A['pts_raw']
    r['A_pts_down']       = A['pts_down']
    r['A_pts_den']        = A['pts_den']
    r.update(_stair_cols(A['stair'], 'A_'))

    # B
    r['B_t_pc_create_ms'] = round(B['t_pc_create'], 3)
    r['B_t_pc_post_ms']   = round(B['t_pc_post'],   3)
    r['B_t_measure_ms']   = round(B['t_measure'],   3)
    r['B_t_total_ms']     = round(B['t_total'],     3)
    r['B_pts_raw']        = B['pts_raw']
    r['B_pts_down']       = B['pts_down']
    r['B_pts_den']        = B['pts_den']
    r.update(_stair_cols(B['stair'], 'B_'))

    # 비교 지표
    if A['t_total'] > 0:
        r['speedup_ratio'] = round(B['t_total'] / A['t_total'], 3)
    else:
        r['speedup_ratio'] = ''

    if B['pts_den'] > 0:
        r['pts_reduction_ratio'] = round(
            1 - A['pts_den'] / B['pts_den'], 4)
    else:
        r['pts_reduction_ratio'] = ''

    return r


def open_csv(model_name: str):
    os.makedirs(RESULT_DIR, exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(RESULT_DIR, f'compare_{model_name}_{ts}.csv')
    f    = open(path, 'w', newline='', encoding='utf-8')
    w    = csv.DictWriter(f, fieldnames=FIELDNAMES)
    w.writeheader()
    print(f"[INFO] CSV → {path}")
    return f, w, path


# ═══════════════════════════════════════════════════════════════════════════════
#  요약 출력
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(rows: list[dict], model_name: str):
    def avg(rows, key):
        vals = [r[key] for r in rows
                if r.get(key) not in ('', None)]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (0, 0)

    sep = '=' * 70
    print(f'\n{sep}')
    print(f'  COMPARISON SUMMARY  —  {model_name.upper()}')
    print(f'  Measured frames : {len(rows)}')
    print(sep)

    print(f'\n  {"Metric":<32} {"[A] FusionVision":>18} {"[B] Full Frame":>15}')
    print(f'  {"-"*32} {"-"*18} {"-"*15}')

    metrics = [
        ('Total time / frame (ms)', 'A_t_total_ms',     'B_t_total_ms'),
        ('PC create time (ms)',     'A_t_pc_create_ms', 'B_t_pc_create_ms'),
        ('PC post-proc time (ms)',  'A_t_pc_post_ms',   'B_t_pc_post_ms'),
        ('Stair measure time (ms)', 'A_t_measure_ms',   'B_t_measure_ms'),
        ('Points raw',             'A_pts_raw',         'B_pts_raw'),
        ('Points denoised',        'A_pts_den',         'B_pts_den'),
    ]
    for label, ka, kb in metrics:
        ma, sa = avg(rows, ka)
        mb, sb = avg(rows, kb)
        print(f'  {label:<32} {ma:>9.2f} ±{sa:>6.2f}   {mb:>9.2f} ±{sb:>6.2f}')

    # FusionVision 전용 단계
    print(f'\n  [A] FusionVision 단계별 분해')
    print(f'  {"Stage":<28} {"Mean(ms)":>10} {"Std(ms)":>10}')
    print(f'  {"-"*28} {"-"*10} {"-"*10}')
    for label, key in [
        ('YOLO inference',    'A_t_yolo_ms'),
        ('FastSAM',           'A_t_fastsam_ms'),
        ('PC creation',       'A_t_pc_create_ms'),
        ('PC post-process',   'A_t_pc_post_ms'),
        ('Stair measure',     'A_t_measure_ms'),
        ('TOTAL',             'A_t_total_ms'),
    ]:
        m, s = avg(rows, key)
        print(f'  {label:<28} {m:>10.2f} {s:>10.2f}')

    # 비교 지표
    sr_m, sr_s   = avg(rows, 'speedup_ratio')
    ptr_m, ptr_s = avg(rows, 'pts_reduction_ratio')
    fps_vals = [r['fps'] for r in rows]

    print(f'\n  {"Speedup ratio (B/A)":<32} {sr_m:>8.3f} ±{sr_s:.3f}')
    print(f'  {"  → 1 이상이면 A(FusionVision)가 빠름":<50}')
    print(f'  {"Point reduction ratio":<32} {ptr_m*100:>7.1f}% ±{ptr_s*100:.1f}%')
    print(f'  {"  → A가 B 대비 포인트를 이만큼 절감":<50}')
    print(f'  {"Avg FPS (wall-clock)":<32} {np.mean(fps_vals):>8.2f}')

    # 계단 측정 비교
    a_meas = [r for r in rows if r.get('A_n_stairs') not in ('', None)]
    b_meas = [r for r in rows if r.get('B_n_stairs') not in ('', None)]

    print(f'\n  계단 측정 성공 프레임')
    print(f'    [A] FusionVision : {len(a_meas)} / {len(rows)} 프레임')
    print(f'    [B] Full Frame   : {len(b_meas)} / {len(rows)} 프레임')

    if a_meas and b_meas:
        print(f'\n  {"측정값":<28} {"[A] mean±std":>18} {"[B] mean±std":>18}')
        print(f'  {"-"*28} {"-"*18} {"-"*18}')
        for label, ka, kb in [
            ('Step height (cm)',  'A_height_cm',  'B_height_cm'),
            ('Step depth (cm)',   'A_depth_cm',   'B_depth_cm'),
            ('Step length (cm)',  'A_length_cm',  'B_length_cm'),
            ('Total height (cm)', 'A_total_h_cm', 'B_total_h_cm'),
        ]:
            ma, sa = avg(a_meas, ka)
            mb, sb = avg(b_meas, kb)
            print(f'  {label:<28} {ma:>7.2f} ±{sa:>5.2f}      '
                  f'{mb:>7.2f} ±{sb:>5.2f}')

    print(f'\n{sep}\n')


# ═══════════════════════════════════════════════════════════════════════════════
#  시각화 헬퍼
# ═══════════════════════════════════════════════════════════════════════════════

def _text_block(img, lines, x=10, y0=28, dy=26, scale=0.62,
                color=(0, 255, 255), thick=2):
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (x, y0 + i*dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def make_display(color_image, A_result, B_result,
                 frame_idx, fps, model_name):
    """
    좌: [A] FusionVision 결과 (마스크 오버레이 + 수치)
    우: [B] Full Frame 결과 (수치만)
    하단: 비교 바 차트 (처리 시간)
    """
    h, w = color_image.shape[:2]
    left  = color_image.copy()
    right = color_image.copy()

    # ── 왼쪽: FastSAM 마스크 오버레이 ───────────────────────────────────────
    if A_result['fastsam_ok']:
        overlay       = np.zeros_like(left)
        overlay[:,:,1] = A_result['mask_vis']   # 초록 채널
        left = cv2.addWeighted(left, 1.0, overlay, 0.45, 0)

    # ── 텍스트 오버레이 ──────────────────────────────────────────────────────
    def stair_lines(s, prefix):
        if s is None:
            return [f'{prefix} No stair detected']
        dist = f"{s['dist_cm']:.1f}cm" if s['dist_cm'] else 'N/A'
        return [
            f"{prefix} Steps={s['n_stairs']}  H={s['height_cm']:.1f}cm",
            f"  Depth={s['depth_cm']:.1f}cm  Len={s['length_cm']:.1f}cm",
            f"  TotalH={s['total_h_cm']:.1f}cm  Dist={dist}",
        ]

    _text_block(left, [
        f'[A] FusionVision  {model_name.upper()}',
        f'Total={A_result["t_total"]:.1f}ms  FPS={fps:.1f}',
        f'YOLO={A_result["t_yolo"]:.1f}ms  SAM={A_result["t_fastsam"]:.1f}ms',
        f'PC={A_result["t_pc_create"]:.1f}ms  Post={A_result["t_pc_post"]:.1f}ms',
        f'pts raw={A_result["pts_raw"]}  den={A_result["pts_den"]}',
    ] + stair_lines(A_result['stair'], ''))

    _text_block(right, [
        '[B] Full Frame (no YOLO/SAM)',
        f'Total={B_result["t_total"]:.1f}ms',
        f'PC={B_result["t_pc_create"]:.1f}ms  Post={B_result["t_pc_post"]:.1f}ms',
        f'pts raw={B_result["pts_raw"]}  den={B_result["pts_den"]}',
    ] + stair_lines(B_result['stair'], ''))

    top = np.concatenate([left, right], axis=1)  # 가로 합치기

    # ── 하단: 처리 시간 바 차트 ──────────────────────────────────────────────
    bar_h   = 120
    bar_img = np.full((bar_h, w*2, 3), 30, dtype=np.uint8)

    bar_items = [
        ('YOLO',   A_result['t_yolo'],      0,                      (100,200,100)),
        ('SAM',    A_result['t_fastsam'],   0,                      (100,100,200)),
        ('PC [A]', A_result['t_pc_create'], B_result['t_pc_create'],(200,150, 50)),
        ('Post[A]',A_result['t_pc_post'],   B_result['t_pc_post'],  (200, 80,150)),
        ('Total A',A_result['t_total'],     0,                      (  0,220,220)),
        ('Total B',0,                       B_result['t_total'],    (220,120,  0)),
    ]

    n     = len(bar_items)
    max_t = max(
        max(a for _, a, _, _ in bar_items),
        max(b for _, _, b, _ in bar_items)
    ) or 1
    slot_w = (w*2) // n
    pad    = 8
    max_bar_h = bar_h - 40

    for i, (label, va, vb, color) in enumerate(bar_items):
        x0 = i * slot_w + pad
        # A 막대 (실선)
        if va > 0:
            bh = int(va / max_t * max_bar_h)
            cv2.rectangle(bar_img,
                          (x0, bar_h - 30 - bh),
                          (x0 + slot_w//2 - pad, bar_h - 30),
                          color, -1)
            cv2.putText(bar_img, f'{va:.0f}',
                        (x0, bar_h - 32 - bh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        # B 막대 (반투명 느낌 = 밝은 색)
        if vb > 0:
            bh  = int(vb / max_t * max_bar_h)
            xb0 = x0 + slot_w//2
            cv2.rectangle(bar_img,
                          (xb0, bar_h - 30 - bh),
                          (xb0 + slot_w//2 - pad, bar_h - 30),
                          tuple(min(255, c+80) for c in color), -1)
            cv2.putText(bar_img, f'{vb:.0f}',
                        (xb0, bar_h - 32 - bh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        tuple(min(255, c+80) for c in color), 1)

        cv2.putText(bar_img, label,
                    (x0, bar_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)

    # speedup 표시
    if A_result['t_total'] > 0:
        sp = B_result['t_total'] / A_result['t_total']
        cv2.putText(bar_img,
                    f'Speedup(B/A)={sp:.2f}x  '
                    f'[{frame_idx}/{BENCHMARK_FRAMES}] q=quit',
                    (10, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,200), 1)

    return np.concatenate([top, bar_img], axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
#  메인 루프
# ═══════════════════════════════════════════════════════════════════════════════

def run_compare(model_name: str = 'exp6'):
    yolo    = load_yolo(model_name)
    fastsam = load_fastsam()
    pipeline, align, depth_scale = init_realsense()

    csv_file, csv_writer, csv_path = open_csv(model_name)
    all_rows   = []
    frame_idx  = 0
    measure_idx= 0
    prev_time  = time.perf_counter()

    total_frames = WARMUP_FRAMES + BENCHMARK_FRAMES
    print(f'\n[INFO] Compare benchmark start')
    print(f'       model={model_name}  warmup={WARMUP_FRAMES}  '
          f'measure={BENCHMARK_FRAMES} frames')
    print(f'       Press q to stop early.\n')

    try:
        while frame_idx < total_frames:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=10000)
            except RuntimeError:
                print('[WARN] Frame timeout, retrying...')
                continue

            aligned     = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            is_warmup = frame_idx < WARMUP_FRAMES
            frame_idx += 1

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            intrinsics  = depth_frame.profile.as_video_stream_profile().intrinsics

            # ── 웜업 ─────────────────────────────────────────────────────────
            if is_warmup:
                if DISPLAY_ENABLED:
                    disp = color_image.copy()
                    cv2.putText(disp,
                                f'Warming up... {frame_idx}/{WARMUP_FRAMES}',
                                (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0,200,255), 2)
                    cv2.imshow('Benchmark Compare', disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            measure_idx += 1

            # ── [A] FusionVision ──────────────────────────────────────────────
            A = process_fusionvision(
                color_image, depth_image, intrinsics, depth_scale,
                yolo, fastsam
            )

            # ── [B] Full Frame ────────────────────────────────────────────────
            B = process_fullframe(
                color_image, depth_image, intrinsics, depth_scale
            )

            # ── FPS (두 방식 합산 후 wall-clock) ─────────────────────────────
            curr_time = time.perf_counter()
            fps       = 1.0 / (curr_time - prev_time + 1e-9)
            prev_time = curr_time

            # ── 기록 ─────────────────────────────────────────────────────────
            row = build_row(measure_idx, fps, A, B)
            csv_writer.writerow(row)
            all_rows.append(row)

            # ── 중간 출력 ────────────────────────────────────────────────────
            if measure_idx % PRINT_EVERY == 0:
                sp = (B['t_total']/A['t_total'] if A['t_total'] > 0 else 0)
                pr = (1 - A['pts_den']/B['pts_den']
                      if B['pts_den'] > 0 else 0)
                print(f'[{measure_idx:>4}/{BENCHMARK_FRAMES}] '
                      f'FPS={fps:.1f}  '
                      f'A_total={A["t_total"]:.1f}ms  '
                      f'B_total={B["t_total"]:.1f}ms  '
                      f'Speedup={sp:.2f}x  '
                      f'PtsReduction={pr*100:.1f}%  '
                      f'A_den={A["pts_den"]}  B_den={B["pts_den"]}')

            # ── 시각화 ───────────────────────────────────────────────────────
            if DISPLAY_ENABLED:
                disp = make_display(
                    color_image, A, B, measure_idx, fps, model_name)
                cv2.imshow('Benchmark Compare  [A]FusionVision | [B]FullFrame',
                           disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('[INFO] Early stop by user.')
                    break

    finally:
        pipeline.stop()
        csv_file.flush()
        csv_file.close()
        if DISPLAY_ENABLED:
            cv2.destroyAllWindows()

    if all_rows:
        print_summary(all_rows, model_name)
    print(f'[INFO] CSV saved → {csv_path}')
    return all_rows


# ═══════════════════════════════════════════════════════════════════════════════
#  진입점
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='FusionVision vs Full Frame 비교 벤치마크')
    parser.add_argument(
        '--model', type=str, default='exp6',
        choices=['exp5', 'exp6'],
        help='FusionVision에 사용할 YOLO 모델 (exp5=YOLOv8n, exp6=YOLOv8m)'
    )
    parser.add_argument(
        '--frames', type=int, default=BENCHMARK_FRAMES,
        help=f'측정 프레임 수 (기본: {BENCHMARK_FRAMES})'
    )
    parser.add_argument(
        '--no-display', action='store_true',
        help='화면 출력 없이 순수 성능만 측정 (더 빠름)'
    )
    args = parser.parse_args()

    BENCHMARK_FRAMES = args.frames
    if args.no_display:
        DISPLAY_ENABLED = False

    run_compare(model_name=args.model)