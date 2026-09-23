"""
動態表格對齊工具
------------------------------------------------
背景：
  同一份 PDF 裡，即使是同一種表單，如果是好幾張紙分開掃描後合併(例如一家供應商由
  多個單位分別評核，每個單位各自簽核一張紙，最後疊在一起掃描)，每一頁在掃描機上的
  實際位置、些微歪斜都可能不完全一樣。只用「寫死的像素座標」校正一次，套用到每一頁，
  遇到這種情況容易出現「後面幾頁欄位對不準」的問題。

做法：
  不直接相信 template.json 裡的絕對座標，而是把它當作「預期位置」，針對每一頁實際的
  掃描圖片，在預期位置附近重新尋找真正的格線在哪裡(用同一套抓格線的技術)，抓到兩個
  可靠的錨點(一個區塊的頂端與底端格線)之後，用比例縮放的方式，把區塊內其他欄位的
  座標，等比例對應到這一頁實際的格線位置上——這樣即使每頁的縮放或位移不完全相同，
  也能正確對齊。
------------------------------------------------
"""

import cv2
import numpy as np


def _binary(pil_image_gray_array, thresh=180):
    _, binary = cv2.threshold(pil_image_gray_array, thresh, 255, cv2.THRESH_BINARY_INV)
    return binary


def find_strong_hline(binary, x_range, expected_y, search_radius=130):
    """在 expected_y 附近 ±search_radius 範圍內，找出最像「一條橫線」的位置。"""
    x0, x1 = x_range
    h = binary.shape[0]
    lo = max(0, expected_y - search_radius)
    hi = min(h, expected_y + search_radius)
    if hi <= lo:
        return expected_y
    strip = binary[lo:hi, x0:x1]
    kernel_w = max(int((x1 - x0) * 0.7), 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    lines = cv2.dilate(cv2.erode(strip, kernel), kernel)
    sums = lines.sum(axis=1)
    if sums.max() <= 0:
        return expected_y
    return lo + int(np.argmax(sums))


def find_strong_vline(binary, y_range, expected_x, search_radius=45):
    """在 expected_x 附近 ±search_radius 範圍內，找出最像「一條直線」的位置。"""
    y0, y1 = y_range
    w = binary.shape[1]
    lo = max(0, expected_x - search_radius)
    hi = min(w, expected_x + search_radius)
    if hi <= lo or y1 <= y0:
        return expected_x
    strip = binary[y0:y1, lo:hi]
    kernel_h = max(int((y1 - y0) * 0.7), 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    lines = cv2.dilate(cv2.erode(strip, kernel), kernel)
    sums = lines.sum(axis=0)
    if sums.max() <= 0:
        return expected_x
    return lo + int(np.argmax(sums))


def compute_row_sequence(pil_image, zone):
    """
    針對「一整排列高可能不規則」的表格(例如評分表，某些列因為換行文字而變高)，
    用「逐列往下找」的方式，而不是頭尾兩點抓一個比例套用到全部——
    因為實際列印/掃描出來，不同列的高度變化不一定是等比例縮放，
    有些列可能剛好長一點、有些剛好短一點，用整體比例反而會讓中間對不準。

    做法：從模板記錄的第一條格線位置開始，往下逐條找。每一條的搜尋起點，
    是「上一條實際找到的位置」加上「模板裡這一段的預期間距」，這樣就算某一列
    的實際高度跟模板不完全一樣，下一條線也能就近修正，不會一直沿用錯誤的位置。

    回傳一個 dict，把模板裡每一條格線的位置，對應到這一頁實際找到的位置。
    """
    gray = np.array(pil_image.convert("L"))
    binary = _binary(gray)

    x_range = (zone["anchor_left"], zone["anchor_right"])
    template_ys = zone["row_boundaries"]
    y_search_first = zone.get("y_search_first", 40)
    y_search_step = zone.get("y_search_step", 25)

    y_map = {}
    prev_template_y = template_ys[0]
    prev_real_y = find_strong_hline(binary, x_range, template_ys[0], search_radius=y_search_first)
    y_map[template_ys[0]] = prev_real_y

    for template_y in template_ys[1:]:
        expected_gap = template_y - prev_template_y
        search_center = prev_real_y + expected_gap
        real_y = find_strong_hline(binary, x_range, search_center, search_radius=y_search_step)
        y_map[template_y] = real_y
        prev_template_y, prev_real_y = template_y, real_y

    x_search = zone.get("x_search", 30)
    real_left = find_strong_vline(binary, (y_map[template_ys[0]], y_map[template_ys[-1]]), zone["anchor_left"], search_radius=x_search)
    real_right = find_strong_vline(binary, (y_map[template_ys[0]], y_map[template_ys[-1]]), zone["anchor_right"], search_radius=x_search)
    x_span = zone["anchor_right"] - zone["anchor_left"]
    x_scale = (real_right - real_left) / x_span if x_span else 1.0

    return {
        "y_map": y_map,
        "x0": real_left, "tx0": zone["anchor_left"], "x_scale": x_scale,
    }


def remap_box_row_sequence(box, row_seq):
    """用逐列校正的結果，把某個欄位的座標換算成這一頁實際的座標。box 的 y0/y1 必須剛好對應到 zone 裡列出的格線位置。"""
    x0, y0, x1, y1 = box
    y_map = row_seq["y_map"]
    if y0 not in y_map or y1 not in y_map:
        raise KeyError(f"座標 y0={y0} 或 y1={y1} 沒有出現在這個 zone 的 row_boundaries 清單裡，請檢查 template.json 設定是否一致。")
    ny0, ny1 = y_map[y0], y_map[y1]
    nx0 = row_seq["x0"] + (x0 - row_seq["tx0"]) * row_seq["x_scale"]
    nx1 = row_seq["x0"] + (x1 - row_seq["tx0"]) * row_seq["x_scale"]
    return [int(round(nx0)), int(round(ny0)), int(round(nx1)), int(round(ny1))]


def compute_zone_transform(pil_image, zone):
    """
    針對一個「區塊」(zone)，在這一頁實際圖片上重新找出區塊的頂/底/左/右邊界，
    回傳一個轉換函式所需的參數，用來把模板座標等比例換算成這一頁的實際座標。

    zone 可以用 "y_search" / "x_search" 自訂搜尋半徑：
      - 如果這個區塊內部列高很穩定(例如單行的表頭資訊列)，用小一點的搜尋半徑，
        避免不小心鎖到隔壁、但線條比較粗的格線。
      - 如果這個區塊內部列高可能因為換行而有落差、需要累積校正(例如評分表)，
        底部邊界要給大一點的搜尋半徑。
    """
    gray = np.array(pil_image.convert("L"))
    binary = _binary(gray)

    ax0, ay0, ax1, ay1 = zone["anchor_left"], zone["anchor_top"], zone["anchor_right"], zone["anchor_bottom"]
    y_search = zone.get("y_search", 40)
    x_search = zone.get("x_search", 35)

    real_top = find_strong_hline(binary, (ax0, ax1), ay0, search_radius=y_search)
    real_bottom = find_strong_hline(binary, (ax0, ax1), ay1, search_radius=y_search)
    real_left = find_strong_vline(binary, (real_top, real_bottom), ax0, search_radius=x_search)
    real_right = find_strong_vline(binary, (real_top, real_bottom), ax1, search_radius=x_search)

    y_span = ay1 - ay0
    x_span = ax1 - ax0
    y_scale = (real_bottom - real_top) / y_span if y_span else 1.0
    x_scale = (real_right - real_left) / x_span if x_span else 1.0

    return {
        "tx0": ax0, "ty0": ay0,
        "x0": real_left, "y0": real_top,
        "x_scale": x_scale, "y_scale": y_scale,
    }


def remap_box(box, transform):
    """用某個區塊的轉換參數，把模板裡的座標換算成這一頁實際對應的座標。"""
    x0, y0, x1, y1 = box
    t = transform
    nx0 = t["x0"] + (x0 - t["tx0"]) * t["x_scale"]
    nx1 = t["x0"] + (x1 - t["tx0"]) * t["x_scale"]
    ny0 = t["y0"] + (y0 - t["ty0"]) * t["y_scale"]
    ny1 = t["y0"] + (y1 - t["ty0"]) * t["y_scale"]
    return [int(round(nx0)), int(round(ny0)), int(round(nx1)), int(round(ny1))]
