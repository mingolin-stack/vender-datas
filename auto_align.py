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


def find_strong_hline(binary, x_range, expected_y, search_radius=130, backward=None, forward=None, prefer_near=True, excluded=None):
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

    # 格線連續度要求，由嚴格到寬鬆依序嘗試：掃描品質好的文件，格線通常很完整連續，
    # 嚴格的要求(70%寬度)比較不會誤判到不相干的短筆畫；但有些文件掃描出來碳粉較淡、
    # 壓縮較重，格線會斷斷續續，太嚴格反而完全偵測不到任何線、整段直接放棄校正。
    # 所以先用嚴格的試，真的完全找不到才逐步放寬，兩邊兼顧。
    sums = None
    for pct in (0.7, 0.5, 0.35, 0.15):
        kernel_w = max(int((x1 - x0) * pct), 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
        lines = cv2.dilate(cv2.erode(strip, kernel), kernel)
        candidate_sums = lines.sum(axis=1).astype(np.float64)
        if candidate_sums.max() > 0:
            sums = candidate_sums
            break
    if sums is None:
        return expected_y

    if not prefer_near:
        return lo + int(np.argmax(sums))

    # 只看「夠強」的候選位置(至少達到該窗口最強線條的一半)，在這些候選裡面挑離 expected_y 最近的一個；
    # 這樣既不會漏掉真正的格線(只要夠強就算數)，也不會被遠處更粗、但不相關的線搶走。
    threshold = sums.max() * 0.5
    candidates = np.where(sums >= threshold)[0]
    abs_positions = lo + candidates
    if excluded:
        keep = np.array([all(abs(p - e) >= 5 for e in excluded) for p in abs_positions])
        if keep.any():
            candidates = candidates[keep]
            abs_positions = abs_positions[keep]
    distances = np.abs(abs_positions - expected_y)
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


def has_crossing_line(binary, y0, gap, x_range, thresh=0.8):
    """
    檢查某個候選位置「後面一整列」的範圍裡，中段有沒有藏著另一條貫穿整列寬度的粗格線。
    如果有，代表這個候選位置還太早——它跟真正的格線之間，其實還夾著另一條格線，
    範圍內量到的「內容」很可能只是不小心框到了那條格線，不是真正的文字。
    真正該停下來的位置，後面應該是乾淨的文字內容，不會再有另一條貫穿的粗線。
    """
    x0, x1 = x_range
    inner_top = y0 + int(gap * 0.35)
    inner_bottom = y0 + int(gap * 0.75)
    if inner_bottom <= inner_top:
        return False
    region = binary[inner_top:inner_bottom, x0:x1]
    row_ratio = (region > 0).mean(axis=1)
    return row_ratio.max() > thresh if len(row_ratio) else False


def find_first_anchor_with_content_check(binary, x_range, expected_y, next_gap, search_radius, min_ink_ratio=0.05, max_retries=4, content_check_x_range=None):
    """
    專門給「一整個區塊的第一條格線」用：因為後面每一步都是從前一步的實際位置接著找，
    如果第一條就找錯，後面會整個跟著錯，所以這裡額外做一層驗證——
    找到候選線之後，檢查它後面(到下一條預期格線之間)是不是真的有內容(文字/墨色)，
    沒有的話，代表大概率是誤判到一條不相干的線(例如版面上剛好有一條粗分隔線，
    但後面其實是空白)，就排除掉這個候選，再往更遠的地方重新找一次。

    content_check_x_range：可選，只檢查這個較窄的 x 範圍裡有沒有內容，而不是整列寬度。
    用來避開「隔壁欄位剛好文字比較密」反而誤判成正確答案的狀況(例如「統一編號」這種
    文字較密集的標籤，可能讓它隔壁、其實是空白的候選行也被誤判成「有內容」)。
    """
    check_x_range = content_check_x_range or x_range
    original_expected_y = expected_y
    excluded = []  # 已經驗證過、判定是假訊號的位置，往後的搜尋都要排除掉這些
    radius = search_radius
    candidate = None
    for _ in range(max_retries):
        candidate = find_strong_hline(
            binary, x_range, original_expected_y, search_radius=radius, excluded=excluded
        )
        if candidate in excluded:
            # 搜尋範圍擴大後，已經沒有其他候選可排除了，直接採用目前找到的
            return candidate
        ratio = ink_ratio(binary, candidate, candidate + next_gap, check_x_range)
        crosses = has_crossing_line(binary, candidate, next_gap, check_x_range)
        if ratio >= min_ink_ratio and not crosses:
            return candidate
        # 這個候選不合格，排除掉它，但搜尋中心仍然維持在原本預期的位置附近，
        # 只是擴大搜尋範圍、並且跳過剛剛排除的這個位置，避免搜尋中心亂跳到不相關的地方。
        excluded.append(candidate)
        radius = min(radius + max(next_gap, 60), 160)  # 範圍最多擴大到這裡，避免搜尋到不相關的標題/LOGO區域
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


def find_row_boundary_robust(binary, x_range, expected_y, next_gap, backward=30, forward=30,
                              min_ink_ratio=0.03, max_retries=4, content_check_x_range=None,
                              max_radius=200):
    """
    穩健版的格線搜尋，套用在逐列搜尋的每一步：
      - 找到候選後，檢查後面是不是真的有內容(不是空白)
      - 檢查候選後面的範圍裡，中段有沒有藏著另一條貫穿整列的粗格線(代表還太早)
      - 不合格的候選會被排除掉，繼續在原本預期的位置附近擴大搜尋，而不是把搜尋中心
        跳到候選點之後，避免搜尋方向跑偏。
    """
    check_x_range = content_check_x_range or x_range
    excluded = []
    back, fwd = backward, forward
    candidate = None
    for _ in range(max_retries):
        candidate = find_strong_hline(binary, x_range, expected_y, backward=back, forward=fwd, excluded=excluded)
        if candidate in excluded:
            return candidate
        ratio = ink_ratio(binary, candidate, candidate + next_gap, check_x_range)
        crosses = has_crossing_line(binary, candidate, next_gap, check_x_range)
        if ratio >= min_ink_ratio and not crosses:
            return candidate
        excluded.append(candidate)
        back = min(back + max(next_gap, 60), max_radius)
        fwd = min(fwd + max(next_gap, 60), max_radius)
    return candidate


def compute_row_sequence(pil_image, zone, ocr_y_seed=None):
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

    ocr_y_seed = ocr_y_seed or {}

    y_map = {}
    if template_ys[0] in ocr_y_seed:
        # 這一條格線已經用 OCR 標籤定位出來了，直接採用，不用再猜格線在哪裡。
        prev_real_y = ocr_y_seed[template_ys[0]]
    else:
        first_gap = template_ys[1] - template_ys[0] if len(template_ys) > 1 else 80
        content_check_x = zone.get("content_check_x_range")
        content_check_x = tuple(content_check_x) if content_check_x else None
        prev_real_y = find_first_anchor_with_content_check(
            binary, x_range, template_ys[0], next_gap=first_gap, search_radius=y_search_first,
            content_check_x_range=content_check_x,
        )
    y_map[template_ys[0]] = prev_real_y
    prev_template_y = template_ys[0]

    for i, template_y in enumerate(template_ys[1:]):
        if template_y in ocr_y_seed:
            # 這一條也已經用 OCR 標籤定位出來了，直接採用；
            # 後面還沒用 OCR 定位到的格線，會接著從這一條「校正過的」位置繼續往下推算，
            # 這樣前面的誤差就不會帶到後面去。
            real_y = ocr_y_seed[template_y]
        else:
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



def _nearest_line_above(binary, x_range, y_top, max_up):
    """在 y_top 上方 max_up 像素內，找出「最靠近 y_top」的一條橫線，回傳該線的中心 y；找不到回傳 None。"""
    x0, x1 = x_range
    lo, hi = max(0, y_top - max_up), max(0, y_top - 2)
    if hi <= lo:
        return None
    strip = binary[lo:hi, x0:x1]
    for pct in (0.6, 0.4, 0.25):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int((x1 - x0) * pct), 10), 1))
        rows = np.where(cv2.erode(strip, kernel).any(axis=1))[0]
        if len(rows):
            end = rows[-1]
            start = end
            while start - 1 in rows:
                start -= 1
            return lo + int((start + end) // 2)
    return None


def ocr_labels_to_line_seed(pil_image, zone, label_tops, max_up=120):
    """
    把 OCR 找到的「印刷標籤文字頂端 y 座標」換算成「這一列上方格線」的實際 y 座標。

    舊寫法直接把標籤文字頂端當成格線位置，但標籤是印在格子「裡面」的，
    文字頂端比上方格線低 20~70 像素(服務範疇這種高的格子差最多)。一旦差距超過
    後續逐列搜尋的範圍(±30)，之後每一列都找不到格線、只能沿用推算值，
    整張表就固定往下偏，裁到下一列(例如聯絡人裁到匯款戶名)。

    新寫法：從標籤頂端往上找「最靠近的那條橫線」，那就是這一列的上框線。
    只看填寫欄位那一段(content_check_x_range)的橫線，避免被左邊合併儲存格
    (例如「公司全名」跨兩列)的框線干擾。往上找不到線的標籤就不採用，
    讓那一條退回原本的格線偵測。
    """
    binary = _binary(np.array(pil_image.convert("L")))
    xr = zone.get("content_check_x_range") or [zone["anchor_left"], zone["anchor_right"]]
    x_range = (int(xr[0]), int(xr[1]))
    seed = {}
    for template_y, label_text in sorted(zone.get("ocr_anchors", {}).items(), key=lambda kv: int(kv[0])):
        if label_text not in label_tops:
            continue
        top = int(label_tops[label_text])
        y = _nearest_line_above(binary, x_range, top, max_up)
        if y is None:         # 範圍內沒有任何橫線 -> 不採用
            continue
        if seed and y <= max(seed.values()):   # 順序錯亂(OCR 找錯標籤)就不採用
            continue
        seed[int(template_y)] = y
    return seed


def snap_cell_x(pil_image, box, search_radius=90, min_cover=0.85):
    """
    把欄位框的左右邊，吸附到「這一列」裡最靠近的直線(儲存格框線)。

    有些供應商是拿電子檔自己填寫，會不小心拉動欄寬，例如標籤欄變寬、
    電話欄右邊多一格，整張表外框還是同樣大小，但中間的直線位置跟模板不同。
    只靠外框換算會裁到隔壁格(例如多裁到標籤的「名」、或切掉電話第一碼)。

    判斷直線的條件：從這一列上框線一路連到下框線(覆蓋 min_cover 以上的列高)，
    手寫或印刷的字不會這麼長，不會被誤認。找不到就維持原本位置。
    """
    x0, y0, x1, y1 = box
    gray = np.array(pil_image.convert("L"))
    binary = _binary(gray)
    h, w = binary.shape
    top, bottom = max(0, y0 + 3), min(h, y1 - 3)
    if bottom - top < 20:
        return box
    band = binary[top:bottom, :]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(int((bottom - top) * min_cover), 10)))
    cols = cv2.erode(band, kernel).any(axis=0)

    def nearest(x):
        lo, hi = max(0, x - search_radius), min(w, x + search_radius)
        xs = np.where(cols[lo:hi])[0]
        if not len(xs):
            return x
        return int(lo + xs[np.argmin(np.abs(lo + xs - x))])

    nx0, nx1 = nearest(x0), nearest(x1)
    if nx1 - nx0 < (x1 - x0) * 0.5:      # 吸附後寬度縮太多，判定為誤判，維持原框
        return box
    return [nx0, y0, nx1, y1]


def _row_vlines(binary, y0, y1, min_cover=0.85):
    """回傳這一列(y0~y1)裡，從上框連到下框的直線所在的 x 位置(布林陣列)。"""
    h = binary.shape[0]
    top, bottom = max(0, y0 + 3), min(h, y1 - 3)
    if bottom - top < 15:
        return None
    band = binary[top:bottom, :]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(int((bottom - top) * min_cover), 10)))
    return cv2.erode(band, kernel).any(axis=0)


def _nearest_true(cols, x, radius):
    w = len(cols)
    lo, hi = max(0, x - radius), min(w, x + radius)
    xs = np.where(cols[lo:hi])[0]
    if not len(xs):
        return None
    return int(lo + xs[np.argmin(np.abs(lo + xs - x))])


def remap_x_row_local(pil_image, template_box, mapped_box, zone, radius=110):
    """
    用「這一列自己的左右外框」重新換算欄位的 x 座標。

    有些掃描檔不是單純位移或縮放，而是歪斜變形(橫線是平的，但直線往一邊斜，
    表格上下兩端的左框線差十幾像素)。整張表只算一組 x 位移/縮放的話，
    越往下越偏，勾選框、附件欄就會裁到隔壁格。改成每一列各自找左右外框，
    就不受歪斜影響。找不到外框的列，維持原本的換算結果。
    """
    binary = _binary(np.array(pil_image.convert("L")))
    cols = _row_vlines(binary, mapped_box[1], mapped_box[3])
    if cols is None:
        return mapped_box
    al, ar = zone["anchor_left"], zone["anchor_right"]
    # 以整張表的換算結果當作預期位置，在附近找這一列的左右外框
    exp_l = mapped_box[0] - (template_box[0] - al) * ((mapped_box[2] - mapped_box[0]) / max(1, template_box[2] - template_box[0]))
    exp_r = mapped_box[2] + (ar - template_box[2]) * ((mapped_box[2] - mapped_box[0]) / max(1, template_box[2] - template_box[0]))
    L = _nearest_true(cols, int(round(exp_l)), radius)
    R = _nearest_true(cols, int(round(exp_r)), radius)
    if L is None or R is None or R - L < (ar - al) * 0.7:
        return mapped_box
    sx = (R - L) / (ar - al)
    nx0 = L + (template_box[0] - al) * sx
    nx1 = L + (template_box[2] - al) * sx
    return [int(round(nx0)), mapped_box[1], int(round(nx1)), mapped_box[3]]
