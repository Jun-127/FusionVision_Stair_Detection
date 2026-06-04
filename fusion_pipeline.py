# fusion_pipeline.py
import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import time
from ultralytics import YOLO, FastSAM

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
EXP5_WEIGHTS    = r'C:\Users\Jaejun_Yeo\Documents\detect_stair\runs\detect\exp5_adamw_n\weights\best.pt'
EXP6_WEIGHTS    = r'C:\Users\Jaejun_Yeo\Documents\detect_stair\runs\detect\exp6_adamw_m\weights\best.pt'
FASTSAM_WEIGHTS = r'C:\Users\Jaejun_Yeo\Documents\detect_stair\FastSAM-s.pt'

CONF_THRESHOLD  = 0.4
VOXEL_SIZE      = 0.02
NB_NEIGHBORS    = 20
STD_RATIO       = 2.0
DISPLAY_W       = 640
DISPLAY_H       = 480
INFER_SIZE      = 320
BBOX_PADDING    = 30
GAP             = 20
CAM_HEIGHT      = 0.13
TILT_CORRECTION = 1.0

PANEL_W = DISPLAY_W // 2
PANEL_H = DISPLAY_H // 2
INFO_H  = 320   # 항목 추가로 높이 증가
WIN_W   = GAP + PANEL_W + GAP + PANEL_W + GAP
WIN_H   = GAP + PANEL_H + GAP + PANEL_H + GAP + INFO_H + GAP


# ─── 모델 로드 ────────────────────────────────────────────────────────────────
def load_yolo(model_name):
    weights = EXP5_WEIGHTS if model_name == 'exp5' else EXP6_WEIGHTS
    print(f"[INFO] Loading YOLO: {model_name}")
    model = YOLO(weights)
    model.to('cuda')
    return model


def load_fastsam():
    print("[INFO] Loading FastSAM-s...")
    model = FastSAM(FASTSAM_WEIGHTS)
    model.to('cuda')
    return model


# ─── RealSense 초기화 ─────────────────────────────────────────────────────────
def init_realsense():
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    profile  = pipeline.start(config)
    align    = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()

    print("[INFO] Waiting for camera to stabilize...")
    for _ in range(30):
        pipeline.wait_for_frames(timeout_ms=10000)

    print(f"[INFO] RealSense ready. Depth scale: {depth_scale:.6f} m/unit")
    return pipeline, align, depth_scale


# ─── 3D 포인트클라우드 생성 ───────────────────────────────────────────────────
def create_pointcloud(depth_image, mask, intrinsics, depth_scale):
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None

    zs    = depth_image[ys, xs] * depth_scale
    valid = (zs > 0) & (zs < 5.0)
    ys, xs, zs = ys[valid], xs[valid], zs[valid]

    if len(zs) == 0:
        return None

    points = np.stack([
        (xs - cx) * zs / fx,
        (ys - cy) * zs / fy,
        zs
    ], axis=1)

    z_min, z_max = zs.min(), zs.max()
    if z_max - z_min < 1e-6:
        z_norm = np.zeros_like(zs)
    else:
        z_norm = (zs - z_min) / (z_max - z_min)

    z_uint8  = (z_norm * 255).astype(np.uint8)
    colormap = cv2.applyColorMap(z_uint8.reshape(-1, 1), cv2.COLORMAP_JET)
    colors   = colormap.reshape(-1, 3)[:, ::-1] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# ─── 포인트클라우드 후처리 ────────────────────────────────────────────────────
def postprocess_pointcloud(pcd):
    pcd_down     = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    pcd_clean, _ = pcd_down.remove_statistical_outlier(
        nb_neighbors=NB_NEIGHBORS,
        std_ratio=STD_RATIO
    )
    return pcd_clean


# ─── 3D 바운딩박스 ────────────────────────────────────────────────────────────
def get_3d_bbox(pcd):
    try:
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox.color = (1, 0, 0)
        return bbox
    except Exception:
        return None


# ─── 계단 측정 ────────────────────────────────────────────────────────────────
def measure_stairs(pcd):
    """
    - n_stairs        : 계단 단수
    - avg_step_height : 평균 단 높이 (Y축 차이)
    - first_step_width : 1st step 깊이 (Z 피크 간격, 앞뒤 방향)
    - first_step_length: 1st step 길이 (X축 범위, 좌우 방향)
    - total_height    : 전체 높이
    - dist_to_first   : 카메라 → 1st step 수평 거리
    """
    if pcd is None or len(pcd.points) < 50:
        return None

    points = np.asarray(pcd.points)
    x_vals = points[:, 0]   # 좌우 (카메라 기준)
    y_vals = points[:, 1]   # 수직
    z_vals = points[:, 2]   # depth

    z_min, z_max = z_vals.min(), z_vals.max()
    z_range      = z_max - z_min

    if z_range < 0.05:
        return None

    # ── Z축 히스토그램 (bin 5cm) ──────────────────────────────────────────────
    bin_size    = 0.05
    n_bins      = max(5, int(z_range / bin_size))
    hist, edges = np.histogram(z_vals, bins=n_bins)

    kernel      = np.ones(3) / 3
    hist_smooth = np.convolve(hist, kernel, mode='same')
    threshold   = hist_smooth.max() * 0.2

    raw_peaks = []
    for i in range(1, len(hist_smooth) - 1):
        if (hist_smooth[i] > hist_smooth[i-1] and
                hist_smooth[i] > hist_smooth[i+1] and
                hist_smooth[i] > threshold):
            raw_peaks.append(i)

    # NMS
    peaks = []
    last  = -999
    for p in raw_peaks:
        if p - last >= 2:
            peaks.append(p)
            last = p

    n_stairs = len(peaks)
    if n_stairs < 2:
        return None

    peak_z_vals = sorted([(edges[p] + edges[p+1]) / 2 for p in peaks])
    margin      = bin_size * 0.6

    # ── Step height: 인접 두 피크 사이 Y값 차이 평균 ─────────────────────────
    step_heights = []
    for i in range(len(peak_z_vals) - 1):
        z1 = peak_z_vals[i]
        z2 = peak_z_vals[i + 1]

        mask1 = np.abs(z_vals - z1) < margin
        mask2 = np.abs(z_vals - z2) < margin

        if mask1.sum() > 5 and mask2.sum() > 5:
            y1_mean = float(y_vals[mask1].mean())
            y2_mean = float(y_vals[mask2].mean())
            h       = abs(y2_mean - y1_mean)
            if h > 0.03:
                step_heights.append(h)

    if not step_heights:
        return None

    avg_step_height = float(np.mean(step_heights)) * TILT_CORRECTION
    total_height    = avg_step_height * n_stairs

    # ── 1st step width (깊이, Z 방향): 1st ~ 2nd 피크 간 Z 간격 ──────────────
    first_step_width = abs(peak_z_vals[1] - peak_z_vals[0])

    # ── 1st step length (길이, X 방향): 1st 피크 포인트의 X 범위 ─────────────
    mask_first = np.abs(z_vals - peak_z_vals[0]) < margin
    first_step_length = 0.0
    first_step_z      = None

    if mask_first.sum() > 5:
        x_first           = x_vals[mask_first]
        first_step_length = float(x_first.max() - x_first.min())
        first_step_z      = float(z_vals[mask_first].min())

    # ── 카메라 → 1st step 수평 거리 ─────────────────────────────────────────
    dist_to_first_step = None
    if first_step_z is not None and first_step_z > CAM_HEIGHT:
        dist_to_first_step = float(
            np.sqrt(max(0.0, first_step_z**2 - CAM_HEIGHT**2))
        )

    return {
        'n_stairs'           : n_stairs,
        'avg_step_height'    : avg_step_height,
        'first_step_width'   : first_step_width,
        'first_step_length'  : first_step_length,
        'total_height'       : total_height,
        'dist_to_first_step' : dist_to_first_step,
    }


# ─── 측정 결과 패널 ───────────────────────────────────────────────────────────
def make_info_panel(result, w, h):
    panel = np.full((h, w, 3), 30, dtype=np.uint8)

    cv2.putText(panel, "Stair Measurement",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (0, 220, 255), 2)
    cv2.line(panel, (10, 48), (w - 10, 48), (80, 80, 80), 1)

    cv2.putText(panel,
                f"Tilt correction: x{TILT_CORRECTION:.2f}  "
                f"Cam height: {CAM_HEIGHT*100:.0f}cm",
                (12, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1)

    if result is None:
        cv2.putText(panel, "No stair detected",
                    (12, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (120, 120, 120), 1)
        return panel

    dist_str = (f"{result['dist_to_first_step']*100:.1f} cm"
                if result['dist_to_first_step'] is not None else "N/A")

    items = [
        ("Steps detected",
         str(result['n_stairs']),
         (100, 255, 100)),
        ("Avg step height",
         f"{result['avg_step_height']*100:.1f} cm",
         (100, 200, 255)),
        ("1st step depth\n(front-back)",
         f"{result['first_step_width']*100:.1f} cm",
         (255, 180, 100)),
        ("1st step length\n(left-right)",
         f"{result['first_step_length']*100:.1f} cm",
         (255, 140, 180)),
        ("Total height",
         f"{result['total_height']*100:.1f} cm",
         (200, 150, 255)),
        ("Dist. to 1st step",
         dist_str,
         (80, 230, 180)),
    ]

    n_cols  = 2
    pad     = 10
    card_w  = (w - pad * (n_cols + 1)) // n_cols
    card_h  = 60
    start_y = 78

    for idx, (label, value, color) in enumerate(items):
        col = idx % n_cols
        row = idx // n_cols
        cx  = pad + col * (card_w + pad)
        cy  = start_y + row * (card_h + 8)

        cv2.rectangle(panel,
                      (cx, cy), (cx + card_w, cy + card_h),
                      (50, 50, 50), -1)
        cv2.rectangle(panel,
                      (cx, cy), (cx + card_w, cy + card_h),
                      color, 1)

        # 레이블 (줄바꿈 처리)
        for li, line in enumerate(label.split('\n')):
            cv2.putText(panel, line,
                        (cx + 8, cy + 16 + li * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (170, 170, 170), 1)

        cv2.putText(panel, value,
                    (cx + 8, cy + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                    color, 2)

    return panel


# ─── 패널 레이블 ──────────────────────────────────────────────────────────────
def add_label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 13 + 10, 32), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


# ─── Open3D → BGR ─────────────────────────────────────────────────────────────
def render_pointcloud_to_image(vis3d, width=DISPLAY_W, height=DISPLAY_H):
    img_o3d = vis3d.capture_screen_float_buffer(do_render=True)
    img_np  = (np.asarray(img_o3d) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return cv2.resize(img_bgr, (width, height))


# ─── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    current_model = 'exp6'

    yolo    = load_yolo(current_model)
    fastsam = load_fastsam()
    pipeline, align, depth_scale = init_realsense()

    vis3d = o3d.visualization.Visualizer()
    vis3d.create_window(
        window_name="3D (offscreen)",
        width=DISPLAY_W, height=DISPLAY_H,
        visible=False
    )
    opt = vis3d.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.point_size = 4.0

    pcd_vis      = o3d.geometry.PointCloud()
    bbox_vis     = None
    geo_added    = False
    pcd_img      = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    stair_result = None

    print("\n[INFO] Pipeline running.")
    print(f"       TILT_CORRECTION={TILT_CORRECTION:.2f}  "
          f"CAM_HEIGHT={CAM_HEIGHT*100:.0f}cm")
    print("       Press 'q' to quit, 's' to switch model\n")

    prev_time = time.time()

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=10000)
            except RuntimeError:
                print("[WARN] Frame timeout, retrying...")
                continue

            aligned     = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            intrinsics  = depth_frame.profile.as_video_stream_profile().intrinsics
            ih, iw      = color_image.shape[:2]

            panel_rgb  = color_image.copy()

            yolo_result = yolo(color_image, conf=CONF_THRESHOLD,
                               imgsz=INFER_SIZE, verbose=False)[0]
            boxes = yolo_result.boxes

            panel_yolo = color_image.copy()
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cv2.rectangle(panel_yolo, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)
                    cv2.putText(panel_yolo, f"stairs {conf:.2f}",
                                (x1, max(y1 - 8, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)

            panel_seg  = color_image.copy()
            merged_pcd = o3d.geometry.PointCloud()

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    rx1, ry1, rx2, ry2 = map(int, box.xyxy[0])
                    rx1 = max(0,  rx1); ry1 = max(0,  ry1)
                    rx2 = min(iw, rx2); ry2 = min(ih, ry2)

                    px1 = max(0,  rx1 - BBOX_PADDING)
                    py1 = max(0,  ry1 - BBOX_PADDING)
                    px2 = min(iw, rx2 + BBOX_PADDING)
                    py2 = min(ih, ry2 + BBOX_PADDING)

                    sam_results = fastsam(
                        color_image,
                        bboxes=[[px1, py1, px2, py2]],
                        imgsz=INFER_SIZE,
                        verbose=False
                    )

                    mask = None
                    if (sam_results and
                            sam_results[0].masks is not None and
                            len(sam_results[0].masks.data) > 0):

                        mask = sam_results[0].masks.data[0].cpu().numpy().astype(np.uint8)
                        mask = cv2.resize(mask, (iw, ih),
                                          interpolation=cv2.INTER_NEAREST)

                        bbox_mask = np.zeros_like(mask)
                        bbox_mask[ry1:ry2, rx1:rx2] = 1
                        mask = mask & bbox_mask

                        if mask.sum() == 0:
                            mask = None
                            continue

                        colored = np.zeros_like(panel_seg)
                        colored[mask > 0] = (0, 120, 255)
                        panel_seg = cv2.addWeighted(
                            panel_seg, 1.0, colored, 0.5, 0)

                        contours, _ = cv2.findContours(
                            mask, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(
                            panel_seg, contours, -1, (0, 255, 255), 2)
                        cv2.rectangle(
                            panel_seg, (rx1, ry1), (rx2, ry2),
                            (0, 255, 0), 1)

                    if mask is not None:
                        pcd = create_pointcloud(
                            depth_image, mask, intrinsics, depth_scale)
                        if pcd is not None and len(pcd.points) > 10:
                            pcd_clean = postprocess_pointcloud(pcd)
                            merged_pcd += pcd_clean

            if len(merged_pcd.points) > 50:
                stair_result = measure_stairs(merged_pcd)
                if stair_result:
                    d = stair_result
                    dist_str = (f"{d['dist_to_first_step']*100:.1f}cm"
                                if d['dist_to_first_step'] else "N/A")
                    print(f"[MEASURE] Steps:{d['n_stairs']}  "
                          f"H:{d['avg_step_height']*100:.1f}cm  "
                          f"Depth:{d['first_step_width']*100:.1f}cm  "
                          f"Length:{d['first_step_length']*100:.1f}cm  "
                          f"Dist:{dist_str}")
            else:
                stair_result = None

            if len(merged_pcd.points) > 0:
                pcd_vis.points = merged_pcd.points
                pcd_vis.colors = merged_pcd.colors

                if not geo_added:
                    vis3d.add_geometry(pcd_vis)
                    geo_added = True
                else:
                    vis3d.update_geometry(pcd_vis)

                new_bbox = get_3d_bbox(merged_pcd)
                if new_bbox:
                    if bbox_vis is not None:
                        vis3d.remove_geometry(
                            bbox_vis, reset_bounding_box=False)
                    bbox_vis = new_bbox
                    vis3d.add_geometry(bbox_vis, reset_bounding_box=False)

            else:
                pcd_vis.points = o3d.utility.Vector3dVector(
                    np.empty((0, 3)))
                pcd_vis.colors = o3d.utility.Vector3dVector(
                    np.empty((0, 3)))
                if geo_added:
                    vis3d.update_geometry(pcd_vis)
                if bbox_vis is not None:
                    vis3d.remove_geometry(
                        bbox_vis, reset_bounding_box=False)
                    bbox_vis = None
                pcd_img = np.zeros(
                    (DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)

            vis3d.poll_events()
            vis3d.update_renderer()

            if len(merged_pcd.points) > 0:
                pcd_img = render_pointcloud_to_image(vis3d)

            curr_time = time.time()
            fps       = 1.0 / (curr_time - prev_time + 1e-9)
            prev_time = curr_time

            def resize_panel(img):
                return cv2.resize(img, (PANEL_W, PANEL_H))

            p1 = add_label(resize_panel(panel_rgb),  "1. RGB")
            p2 = add_label(resize_panel(panel_yolo), "2. YOLO BBox")
            p3 = add_label(resize_panel(panel_seg),  "3. FastSAM Segmentation")
            p4 = add_label(resize_panel(pcd_img),    "4. 3D Point Cloud")

            BG = np.full((WIN_H, WIN_W, 3), 40, dtype=np.uint8)

            r1 = GAP
            r2 = GAP + PANEL_H + GAP
            r3 = GAP + PANEL_H + GAP + PANEL_H + GAP
            c1 = GAP
            c2 = GAP + PANEL_W + GAP

            BG[r1:r1+PANEL_H, c1:c1+PANEL_W] = p1
            BG[r1:r1+PANEL_H, c2:c2+PANEL_W] = p2
            BG[r2:r2+PANEL_H, c1:c1+PANEL_W] = p3
            BG[r2:r2+PANEL_H, c2:c2+PANEL_W] = p4

            cv2.putText(BG,
                        f"{current_model.upper()} | FPS: {fps:.1f}  "
                        f"[q] quit  [s] switch model",
                        (GAP, r3 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (180, 180, 180), 1)

            info_w     = WIN_W - GAP * 2
            info_panel = make_info_panel(stair_result, info_w, INFO_H)
            BG[r3:r3+INFO_H, GAP:GAP+info_w] = info_panel

            cv2.imshow("FusionVision - Stair Detection Pipeline", BG)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                current_model = 'exp5' if current_model == 'exp6' else 'exp6'
                yolo = load_yolo(current_model)
                print(f"[INFO] Switched to {current_model.upper()}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        vis3d.destroy_window()
        print("[INFO] Pipeline stopped.")


if __name__ == '__main__':
    main()