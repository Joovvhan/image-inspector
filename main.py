from glob import glob
import os
import cv2
import numpy as np

if __name__ == "__main__":

    img_files = glob("./images/*.jpg")
    img_files += glob("./images/*.png")

    patch_file = "./images/patch.png"

    filename_width = 30

    print(f"검사할 이미지 목록: {img_files}")

    for file in img_files:

        # 전체 이미지와 패치 이미지 불러오기 (alpha 포함 PNG)
        image = cv2.imread(file)  # 일반 RGB/BGR
        patch = cv2.imread(patch_file, cv2.IMREAD_UNCHANGED)  # RGBA

        # 패치에서 alpha 채널 추출
        alpha = patch[:, :, 3]
        mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]  # alpha > 0 부분만 255

        # BGR 채널만 사용
        patch_bgr = patch[:, :, :3]
        image_bgr = image  # 이미 BGR

        # 템플릿 매칭 수행 (mask 적용)
        result = cv2.matchTemplate(image_bgr, patch_bgr, cv2.TM_CCOEFF_NORMED, mask=mask)

        threshold = 0.9
        loc = np.where(result >= threshold)

        bboxes = [
            {
                "x0": int(x),
                "y0": int(y),
                "x1": int(x + patch_bgr.shape[1]),
                "y1": int(y + patch_bgr.shape[0])
            }
            for (x, y) in zip(*loc[::-1])
        ]

        # 출력
        if bboxes:
            bboxes_str = "  ".join(f"x0: {b['x0']}, y0: {b['y0']}, x1: {b['x1']}, y1: {b['y1']}" for b in bboxes)
            print(f"{file:<{filename_width}} : {bboxes_str}")
        else:
            print(f"{file:<{filename_width}} : 일치하는 부분 없음")