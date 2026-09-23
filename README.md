# 供應商表單 PDF 掃描辨識工具

## 這是什麼

同仁把供應商填好、掃描成 PDF 的表格上傳到這個網頁，程式會：

1. 上方先選擇表單類型：**供應商資料表** 或 **供應商評核表**
2. 把 PDF 轉成圖片(評核表若同一家供應商由多個單位評核、掃在同一份 PDF 裡，會自動依每 2 頁切成好幾份分開處理)
3. 依照對應的 `templates/*.json` 裡校正好的座標裁出每個欄位；供應商評核表額外會針對每一頁**動態微調**格線位置(因為不同單位各自簽核的紙張，掃描時的實際位置多少有落差，固定死的座標沒辦法保證每一頁都準)
4. 文字欄位送 Google Cloud Vision API 辨識(支援手寫)；勾選框用「黑色像素比例」判斷有沒有打勾
5. 畫面顯示「裁切小圖 + 辨識建議值」，同仁核對、修正
6. 按確認後，寫入 Google Drive：
   - **供應商資料表**：`供應商主檔/統編_公司名稱.xlsx`(每次提交新增一列)，`彙總總表.xlsx`(每家供應商固定一列，重複提交覆蓋更新)
   - **供應商評核表**：`供應商評核紀錄/供應商名稱_評核紀錄.xlsx`(每個單位各自新增一列)，`評核彙總表.xlsx`(同步新增)；**如果一份 PDF 裡有多個單位的評核，全部確認存檔後，畫面上會出現「多單位評分彙整平均」，把各單位填的分數取平均、算出總分與等級，另外存一列彙總紀錄**

## 檔案說明

| 檔案 | 用途 |
|---|---|
| `streamlit_app.py` | 主程式(含表單類型切換、多份評核拆分、彙總平均) |
| `drive_utils.py` | 存取 Google Drive |
| `vision_utils.py` | 呼叫 Vision API 做文字辨識；用像素比例判斷勾選框 |
| `master_utils.py` | 主檔/評核紀錄/彙總表的讀寫邏輯；`aggregate_score()` 是多單位評分平均的計算邏輯 |
| `auto_align.py` | 供應商評核表專用：在每一頁實際掃描圖片上動態尋找格線、微調欄位座標，因應不同頁面掃描位置的落差 |
| `templates/supplier_form.json` | 供應商資料表的欄位座標校正檔 |
| `templates/eval_form.json` | 供應商評核表的欄位座標校正檔(2頁、44個欄位，含動態校正用的 zones 設定) |
| `requirements.txt` | 套件清單(新增了 `opencv-python-headless`，動態校正需要用到) |

## ⚠️ 目前的測試狀況

- ✅ 供應商評核表的座標已改用**真實掃描檔**(汎嘉工程，含 3 個不同單位個別掃描、合併成一份 6 頁 PDF 的情況)重新校正，並針對「不同頁面掃描位置略有落差」加上了動態微調機制，用這 3 份實測過，包含最容易出錯的評分欄位都已核對過不再跨列/跨欄
- ✅ 多單位評分彙整平均的計算邏輯，已用這 3 份真實資料的實際分數測試過，結果符合預期(缺填的單位不會拉低平均)
- ⚠️ **「評核總分」的計算公式是我依表單上印的說明文字("評核總分，依該次評核、所有評核人員的評分，彙總計算")推測的**：目前做法是每個評核項目取平均、10項加總(滿分100)，加分項目另外平均、每項上限10分後加進總分。**這個公式請務必幫我確認是否符合貴公司實際想要的算法**，如果不對，告訴我正確的計算方式，我再調整。
- ✅ 供應商資料表的座標、勾選偵測、存檔邏輯，都已用真實掃描檔驗證過
- ⏳ Vision API 實際辨識手寫字的準確率、Google Drive 實際讀寫，都需要部署後在有網路的正式環境測試(我開發的環境沒有對外網路)。

## 部署方式

跟之前一樣，用 Streamlit Community Cloud(免費)：

1. 建立 GitHub repository(建議設為 **Private**)
2. 把這個資料夾裡的所有檔案(含 `templates/` 子資料夾)全部上傳上去，保持資料夾結構一致
3. 到 https://share.streamlit.io 用 GitHub 帳號登入，建立新 App，Main file path 填 `streamlit_app.py`
4. 設定 Secrets — 到 App 的設定(Settings) → Secrets，貼入：

```toml
drive_folder_id = "你的 Google Drive 資料夾 ID"

[gcp_service_account]
type = "service_account"
project_id = "你的專案ID"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...(很長一串)...\n-----END PRIVATE KEY-----\n"
client_email = "...@....iam.gserviceaccount.com"
client_id = "..."
token_uri = "https://oauth2.googleapis.com/token"
```

上面這些值，除了 `drive_folder_id`，其他都可以直接從您下載的那個 `.json` 金鑰檔裡複製對應欄位過來(打開 .json 檔案用文字編輯器看，裡面每個欄位名稱都對得起來)。

**`drive_folder_id` 怎麼找**：打開您要存放供應商資料的 Google Drive 資料夾，網址列會長得像
`https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrStUvWxYz`
最後那一串 `1AbCdEfGhIjKlMnOpQrStUvWxYz` 就是資料夾 ID。

5. 存檔後，Streamlit 會自動重新部署套用新的 Secrets，就可以開始使用了。

## 如果模板改版了，或要新增第三種表單

用人工校正的方式重新框出欄位座標，更新 `templates/` 底下對應的 json 檔(格式：`{"name":欄位名, "type":"text"/"checkbox_single"/"checkbox_group", "box":[x0,y0,x1,y1], "page":第幾頁(可省略，預設第1頁)}`)，上傳更新到 GitHub 上的同名檔案即可自動生效。

如果要新增第三種表單類型，在 `streamlit_app.py` 開頭的 `FORM_TYPES` 字典裡，比照現有兩種表單的寫法加一組設定即可(模板路徑、建檔規則、彙總表要用覆蓋更新還是每次新增列)。

如果掃描解析度(DPI)跟目前的 200 不一樣，座標需要等比例換算，或是統一約定所有掃描都固定用 200 DPI，比較不容易出錯。
