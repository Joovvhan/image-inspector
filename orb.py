from glob import glob
import cv2
import numpy as np

def estimate_scale(kp_patch, kp_img, matches):
    """특징점 매칭으로 추정 스케일 계산"""
    if len(matches) < 2:
        return 1.0
    p1, p2 = kp_patch[matches[0].queryIdx].pt, kp_patch[matches[1].queryIdx].pt
    i1, i2 = kp_img[matches[0].trainIdx].pt, kp_img[matches[1].trainIdx].pt
    patch_dist = np.linalg.norm(np.array(p1) - np.array(p2))
    img_dist = np.linalg.norm(np.array(i1) - np.array(i2))
    return img_dist / patch_dist if patch_dist != 0 else 1.0

if __name__ == "__main__":

    # --- 설정 변수 ---
    MIN_SCALE = 0.5
    MAX_SCALE = 2.0
    TEMPLATE_MATCH_THRESHOLD = 0.9
    FINE_SCALE_STEP = 0.01
    FINE_SCALE_MARGIN = 0.1  # ±10% 범위
    filename_width = 30

    img_files = glob("./images/*.jpg") + glob("./images/*.png")
    patch_file = "./images/patch.png"

    # ORB 생성
    orb = cv2.ORB_create(1000)

    # 패치 이미지와 alpha 마스크
    patch = cv2.imread(patch_file, cv2.IMREAD_UNCHANGED)
    alpha = patch[:, :, 3]
    mask_orig = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]
    patch_bgr_orig = patch[:, :, :3]

    # 패치 특징점
    kp_patch, des_patch = orb.detectAndCompute(patch_bgr_orig, None)
    print(f"[PATCH] 특징점 추출 완료, patch kp: {len(kp_patch)}")

    for file in img_files:
        image = cv2.imread(file)
        img_h, img_w = image.shape[:2]
        patch_h, patch_w = patch_bgr_orig.shape[:2]

        # 이미지 특징점
        kp_img, des_img = orb.detectAndCompute(image, None)
        if des_img is None or des_patch is None:
            print(f"{file:<{filename_width}} : 특징점 부족, patch kp: {len(kp_patch)}, image kp: {0 if des_img is None else len(kp_img)}")
            continue
        print(f"{file:<{filename_width}} : 특징점 추출 완료, patch kp: {len(kp_patch)}, image kp: {len(kp_img)}")

        # 특징점 매칭
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_patch, des_img)
        matches = sorted(matches, key=lambda x: x.distance)
        print(f"{file:<{filename_width}} : 특징점 매칭 완료, 후보 매칭 수: {len(matches)}")

        if len(matches) == 0:
            print(f"{file:<{filename_width}} : 일치 후보 없음")
            continue

        # 추정 스케일 계산 + 강제 범위 적용
        est_scale = estimate_scale(kp_patch, kp_img, matches)
        est_scale = max(MIN_SCALE, min(MAX_SCALE, est_scale))
        fine_scales = np.arange(
            max(0.1, est_scale * (1 - FINE_SCALE_MARGIN)),
            est_scale * (1 + FINE_SCALE_MARGIN) + 0.001,
            FINE_SCALE_STEP
        )
        print(f"{file:<{filename_width}} : 추정 스케일 {est_scale:.3f}, fine_scales 후보: {len(fine_scales)}")

        found = False
        max_score_overall = -1.0

        for scale in fine_scales:
            patch_bgr = cv2.resize(patch_bgr_orig, (0, 0), fx=scale, fy=scale)
            mask = cv2.resize(mask_orig, (0, 0), fx=scale, fy=scale)

            # 이미지보다 큰 patch는 스킵
            if patch_bgr.shape[0] > img_h or patch_bgr.shape[1] > img_w:
                continue

            result = cv2.matchTemplate(image, patch_bgr, cv2.TM_CCOEFF_NORMED, mask=mask)
            loc = np.where(result >= TEMPLATE_MATCH_THRESHOLD)
            max_val = result.max()
            max_score_overall = max(max_score_overall, max_val)

            if len(loc[0]) > 0:
                x, y = loc[1][0], loc[0][0]
                bbox = {
                    "x0": int(x),
                    "y0": int(y),
                    "x1": int(x + patch_bgr.shape[1]),
                    "y1": int(y + patch_bgr.shape[0])
                }
                bbox_str = f"x0: {bbox['x0']}, y0: {bbox['y0']}, x1: {bbox['x1']}, y1: {bbox['y1']}"
                print(f"{file:<{filename_width}} : MATCHED, scale: {scale:.3f}, image size: {img_w}x{img_h}, patch size: {patch_w}x{patch_h}, bbox: {bbox_str}")
                found = True
                break

        if not found:
            print(f"{file:<{filename_width}} : threshold 미달, max score: {max_score_overall:.3f}, image size: {img_w}x{img_h}, patch size: {patch_w}x{patch_h}")
