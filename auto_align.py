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


def find_strong_hline(binary, x_range, expected_y, search_radius=130, backward=None, forward=None, prefer_near=True):
    """
    在 expected_y 附近找出最像「一條橫線」的位置。
    預設往前、往後都用 search_radius 這個範圍(對稱)；
    如果有另外指定 backward/forward，就用不對稱的範圍——
    例如「這一段內容只可能變長、不可能變短」的情況，backward 給小一點，
    避免大範圍搜尋時反而往回找到前面已經找過的格線。

    prefer_near=True(預設)時，不是單純找範圍內「最粗/最強」的線，而是用
    「線條強度 ÷ 跟預期位置的距離」加權評分——範圍開大時，常常會有別的地方
    剛好有一條更粗、但其實距離預期位置很遠的線(例如表格另一個區塊的粗框線)，
    單純比強度會誤判成那條；加上距離的懲罰之後，優先選「夠強、而且離預期
    位置最近」的線，才不會因為搜尋範圍開大就跳到不相干的地方。
    """
    x0, x1 = x_range
    h = binary.shape[0]
    back = backward if backward is not None else search_radius
    fwd = forward if forward is not None else search_radius
    lo = max(0, expected_y - back)
    hi = min(h, expected_y + fwd)
    if hi <= lo:
        return expected_y
    strip = binary[lo:hi, x0:x1]
    kernel_w = max(int((x1 - x0) * 0.7), 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    lines = cv2.dilate(cv2.erode(strip, kernel), kernel)
    sums = lines.sum(axis=1).astype(np.float64)
    if sums.max() <= 0:
        return expected_y

    if not prefer_near:
        return lo + int(np.argmax(sums))

    # 只看「夠強」的候選位置(至少達到該窗口最強線條的一半)，在這些候選裡面挑離 expected_y 最近的一個；
    # 這樣既不會漏掉真正的格線(只要夠強就算數)，也不會被遠處更粗、但不相關的線搶走。
    threshold = sums.max() * 0.5
    candidates = np.where(sums >= threshold)[0]
    distances = np.abs((lo + candidates) - expected_y)
    best = candidates[np.argmin(distances)]
    return lo + int(best)


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


def ink_ratio(binary, y0, y1, x_range):
    """計算某個區域裡「有墨色(文字/線條)的像素」佔比，用來判斷這個區域是不是空白的。"""
    x0, x1 = x_range
    y0, y1 = max(0, y0), min(binary.shape[0], y1)
    if y1 <= y0:
        return 0.0
    region = binary[y0:y1, x0:x1]
    return float((region > 0).mean())


def find_first_anchor_with_content_check(binary, x_range, expected_y, next_gap, search_radius, min_ink_ratio=0.05, max_retries=4):
    """
    專門給「一整個區塊的第一條格線」用：因為後面每一步都是從前一步的實際位置接著找，
    如果第一條就找錯，後面會整個跟著錯，所以這裡額外做一層驗證——
    找到候選線之後，檢查它後面(到下一條預期格線之間)是不是真的有內容(文字/墨色)，
    沒有的話，代表大概率是誤判到一條不相干的線(例如版面上剛好有一條粗分隔線，
    但後面其實是空白)，就排除掉這個候選，再往更遠的地方重新找一次。
    """
    tried_positions = []
    radius = search_radius
    for _ in range(max_retries):
        candidate = find_strong_hline(binary, x_range, expected_y, search_radius=radius)
        if any(abs(candidate - t) < 5 for t in tried_positions):
            # 搜尋範圍擴大後還是找到同一個位置，表示已經沒有更好的候選了，直接採用
            return candidate
        ratio = ink_ratio(binary, candidate, candidate + next_gap, x_range)
        if ratio >= min_ink_ratio:
            return candidate
        # 這個候選後面是空的，排除掉，往更遠的地方找
        tried_positions.append(candidate)
        radius = radius + max(next_gap, 60)
        expected_y = candidate + next_gap  # 從這個(可能是假的)候選點之後，繼續往前找下一個
    return candidate


def find_first_anchor_two_stage(binary, x_range, pre_anchor_y, pre_anchor_search, table_top_y, next_gap, min_gap_from_pre=15, search_radius=200, min_ink_ratio=0.05):
    """
    專門解決「表格正上方的小方框底線，容易被誤判成表格頂端」的狀況。

    不直接在 table_top_y 附近找答案(因為那個位置本身就可能是造成混淆的那條線)，
    而是分兩步：
      1. 先找到 pre_anchor_y 附近那條很穩定、幾乎每份文件都在的線(例如「文件編號」
         小方框的底線)，當作確定的起點。
      2. 從這個確定的起點，往下(只往下，不回頭)搜尋，找到第一條「後面有實際內容」
         的線，中間至少要隔 min_gap_from_pre 像素(避免又抓回同一條線)。

    這樣即使兩條線之間的空白間距，每份文件大小不一樣，也能正確分辨。
    """
    pre_anchor = find_strong_hline(binary, x_range, pre_anchor_y, search_radius=pre_anchor_search)

    search_start = pre_anchor + min_gap_from_pre
    h = binary.shape[1]
    x0, x1 = x_range
    strip = binary[search_start:search_start + search_radius, x0:x1]
    kernel_w = max(int((x1 - x0) * 0.7), 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    lines = cv2.dilate(cv2.erode(strip, kernel), kernel)
    sums = lines.sum(axis=1).astype(np.float64)
    if sums.max() <= 0:
        return table_top_y

    threshold = sums.max() * 0.5
    candidates = np.where(sums >= threshold)[0]
    # 由近到遠，找到第一個「後面有實際內容」的候選位置
    for c in sorted(candidates):
        y = search_start + c
        if ink_ratio(binary, y, y + next_gap, x_range) >= min_ink_ratio:
            return y
    return search_start + int(candidates[0]) if len(candidates) else table_top_y


def compute_row_sequence(pil_image, zone):
    """
    針對「一整排列高可能不規則」的表格(例如評分表某些列因為換行文字而變高、
    或供應商資料表的「服務範疇」欄位因為填寫內容長短不一而變高)，
    用「逐列往下找」的方式，而不是頭尾兩點抓一個比例套用到全部——
    因為實際列印/掃描出來，不同列的高度變化不一定是等比例縮放，
    有些列可能剛好長一點、有些剛好短一點，用整體比例反而會讓中間對不準。

    做法：從模板記錄的第一條格線位置開始，往下逐條找。每一條的搜尋起點，
    是「上一條實際找到的位置」加上「模板裡這一段的預期間距」，這樣就算某一列
    的實際高度跟模板不完全一樣，下一條線也能就近修正，不會一直沿用錯誤的位置。

    每一步的搜尋半徑可以各自設定(row_search_radii，長度要跟 row_boundaries 少1，
    對應每一段的搜尋範圍)——大部分列高很固定的地方用小範圍，避免誤鎖到隔壁較粗的
    格線；已知內容長度可能落差很大的地方(例如服務範疇)用大範圍。沒有個別設定的話，
    就都用 y_search_step 這個預設值。

    回傳一個 dict，把模板裡每一條格線的位置，對應到這一頁實際找到的位置。
    """
    gray = np.array(pil_image.convert("L"))
    binary = _binary(gray)

    x_range = (zone["anchor_left"], zone["anchor_right"])
    template_ys = zone["row_boundaries"]
    y_search_first = zone.get("y_search_first", 40)
    default_step = zone.get("y_search_step", 25)
    per_step_radii = zone.get("row_search_radii")  # 可選：長度應為 len(template_ys)-1

    y_map = {}
    prev_template_y = template_ys[0]
    first_gap = template_ys[1] - template_ys[0] if len(template_ys) > 1 else 80
    pre_anchor = zone.get("pre_anchor")  # 可選：{"template_y": ..., "search_radius": ...} 指向更穩定的參考線
    if pre_anchor:
        prev_real_y = find_first_anchor_two_stage(
            binary, x_range,
            pre_anchor_y=pre_anchor["template_y"],
            pre_anchor_search=pre_anchor.get("search_radius", 40),
            table_top_y=template_ys[0],
            next_gap=first_gap,
            min_gap_from_pre=pre_anchor.get("min_gap", 15),
            search_radius=y_search_first,
        )
    else:
        prev_real_y = find_first_anchor_with_content_check(
            binary, x_range, template_ys[0], next_gap=first_gap, search_radius=y_search_first
        )
    y_map[template_ys[0]] = prev_real_y

    for i, template_y in enumerate(template_ys[1:]):
        expected_gap = template_y - prev_template_y
        search_center = prev_real_y + expected_gap
        step_setting = per_step_radii[i] if per_step_radii else default_step
        next_gap = template_ys[i + 2] - template_y if i + 2 < len(template_ys) else expected_gap

        if isinstance(step_setting, (list, tuple)):
            back, fwd = step_setting
            real_y = find_strong_hline(binary, x_range, search_center, backward=back, forward=fwd)
            # 這一步是刻意放寬搜尋範圍的(例如服務範疇這種內容長度不固定的欄位)，
            # 一樣順便驗證一下找到的位置後面是不是真的有內容，避免跳到不相干的粗線。
            if ink_ratio(binary, real_y, real_y + next_gap, x_range) < 0.03:
                real_y = find_strong_hline(binary, x_range, search_center + next_gap, backward=5, forward=fwd)
        else:
            real_y = find_strong_hline(binary, x_range, search_center, search_radius=step_setting)
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


def refine_column_divider(pil_image, box, search_radius=40):
    """
    有些欄位(例如統一編號/身份證號)是「標籤在上、數字在下」的結構，
    但它內部這條分隔線的位置，不一定跟同一列左邊其他欄位的分隔線完全對齊
    (可能因為標籤文字换行方式不同，跟左邊欄位差個幾十像素)。

    這個函式只用「這個欄位自己的 x 範圍」，在預期的 y 位置附近重新找一次
    真正的分隔線，獨立修正，不受同一列其他欄位影響。
    """
    gray = np.array(pil_image.convert("L"))
    binary = _binary(gray)
    x0, y0, x1, y1 = box
    height = y1 - y0
    real_top = find_strong_hline(binary, (x0, x1), y0, search_radius=search_radius)
    return [x0, real_top, x1, real_top + height]


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
