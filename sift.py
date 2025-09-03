from glob import glob
import cv2
import numpy as np

def estimate_scale(kp_patch, kp_img, matches):
    """ 특징점 매칭으로 추정 스케일 계산 """
    if len(matches) < 2:
        return 1.0
    ratios = []
    for i in range(len(matches)):
        for j in range(i+1, len(matches)):
            p1, p2 = kp_patch[matches[i].queryIdx].pt, kp_patch[matches[j].queryIdx].pt
            i1, i2 = kp_img[matches[i].trainIdx].pt, kp_img[matches[j].trainIdx].pt
            patch_dist = np.linalg.norm(np.array(p1) - np.array(p2))
            img_dist = np.linalg.norm(np.array(i1) - np.array(i2))
            if patch_dist > 1e-5:
                ratios.append(img_dist / patch_dist)
    return np.median(ratios) if ratios else 1.0

if __name__ == "__main__":

    img_files = glob("./images/*.jpg") + glob("./images/*.png")
    patch_file = "./images/patch.png"
    filename_width = 30

    min_scale = 0.5
    max_scale = 2.0
    threshold = 0.5  # matchTemplate 최소 점수
    min_knn = 5      # KNN 후보 최소 수
    draw_min_score = 0.1  # 그림 그릴 최소 score

    # ORB 생성
    orb = cv2.ORB_create(2000)

    # 패치 이미지와 alpha 마스크
    patch = cv2.imread(patch_file, cv2.IMREAD_UNCHANGED)
    alpha = patch[:, :, 3]
    mask_orig = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]
    patch_bgr_orig = patch[:, :, :3]

    # 패치 특징점
    kp_patch, des_patch = orb.detectAndCompute(patch_bgr_orig, None)
    patch_kp_count = len(kp_patch)

    for file in img_files:
        image = cv2.imread(file)
        img_h, img_w = image.shape[:2]

        kp_img, des_img = orb.detectAndCompute(image, None)
        image_kp_count = 0 if des_img is None else len(kp_img)

        if des_img is None or des_patch is None:
            print(f"{file:<{filename_width}} : patch_kp:{patch_kp_count}, image_kp:{image_kp_count}, "
                  f"KNN 후보:0, scale:0.0, fine_scales:0, max_score:-1.0, MATCHED:False, bbox:")
            continue

        # KNN 매칭
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des_patch, des_img, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        knn_count = len(good_matches)

        # 추정 스케일 계산
        if knn_count < min_knn:
            est_scale = 1.0
            fine_scales = []
            max_score_overall = -1.0
            found = False
            bbox_dict = {}
        else:
            est_scale = estimate_scale(kp_patch, kp_img, good_matches)
            est_scale = np.clip(est_scale, min_scale, max_scale)
            fine_scales = np.arange(max(0.1, est_scale*0.9), est_scale*1.1+0.001, 0.01)

            found = False
            max_score_overall = -1.0
            bbox_dict = {}

            for scale in fine_scales:
                patch_bgr = cv2.resize(patch_bgr_orig, (0,0), fx=scale, fy=scale)
                mask = cv2.resize(mask_orig, (0,0), fx=scale, fy=scale)
                if patch_bgr.shape[0] > image.shape[0] or patch_bgr.shape[1] > image.shape[1]:
                    continue

                result = cv2.matchTemplate(image, patch_bgr, cv2.TM_CCOEFF_NORMED, mask=mask)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if max_val > max_score_overall:
                    max_score_overall = max_val
                    if max_val >= threshold:
                        x, y = max_loc
                        bbox_dict = {"x0": x, "y0": y, "x1": x+patch_bgr.shape[1], "y1": y+patch_bgr.shape[0]}
                        found = True
                        best_result = result  # 최고점 스케일의 matchTemplate 결과 저장

        bbox_str = ""
        if bbox_dict:
            bbox_str = f"x0:{bbox_dict['x0']},y0:{bbox_dict['y0']},x1:{bbox_dict['x1']},y1:{bbox_dict['y1']}"

        print(f"{file:<{filename_width}} : patch_kp:{patch_kp_count}, image_kp:{image_kp_count}, "
              f"KNN 후보:{knn_count}, scale:{est_scale:.3f}, fine_scales:{len(fine_scales)}, "
              f"max_score:{max_score_overall:.3f}, MATCHED:{found}, bbox:{bbox_str}")

        # 이미지 띄우기: MATCHED거나 최소 점수 이상이면
        if found and bbox_dict:
            display_img = image.copy()
            x0, y0, x1, y1 = bbox_dict["x0"], bbox_dict["y0"], bbox_dict["x1"], bbox_dict["y1"]
            patch_h, patch_w = y1 - y0, x1 - x0

            # BBOX 사각형
            cv2.rectangle(display_img, (x0, y0), (x1, y1), (0, 0, 255), 2)

            # 점수 텍스트
            score_text = f"{max_score_overall:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size, _ = cv2.getTextSize(score_text, font, font_scale, thickness)
            padding = 3
            text_x = x0 + padding
            text_y = y0 + text_size[1] + padding
            cv2.rectangle(display_img,
                        (x0, y0),
                        (x0 + text_size[0] + 2*padding, y0 + text_size[1] + 2*padding),
                        (0, 0, 255), -1)
            cv2.putText(display_img, score_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

            # -----------------------------------------------------------------
            # 히트맵 제거
            # result_resized = cv2.resize(best_result, (patch_w, patch_h))
            # norm_result = np.clip(result_resized, 0, 1)  # 0~1

            # patch 원본 알파 resize
            # patch_alpha_resized = cv2.resize(mask_orig, (patch_w, patch_h))
            # patch_alpha_resized = (patch_alpha_resized > 0).astype(np.uint8)  # 1=유효, 0=무시

            # ggplot heat 스타일 (alpha 포함)
            # heatmap = np.zeros((patch_h, patch_w, 4), dtype=np.uint8)
            # heatmap[..., 0] = (norm_result * 255).astype(np.uint8)          # Blue
            # heatmap[..., 1] = ((1 - norm_result) * 255).astype(np.uint8)    # Green
            # heatmap[..., 2] = 255                                           # Red
            # heatmap[..., 3] = (norm_result * 180 * patch_alpha_resized).astype(np.uint8)  # alpha 적용 + 알파 영역 제외

            # overlay
            # overlay = display_img[y0:y1, x0:x1].astype(np.float32)
            # alpha = heatmap[..., 3:4].astype(np.float32)/255.0
            # display_img[y0:y1, x0:x1] = (1-alpha)*overlay + alpha*heatmap[..., :3].astype(np.float32)
            # display_img = display_img.astype(np.uint8)
            # -----------------------------------------------------------------

            cv2.imshow(f"PATCH MATCH: {file}", display_img)
            cv2.waitKey(1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
