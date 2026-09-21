# 供應商表單 PDF 掃描辨識工具

## 這是什麼

同仁把供應商填好、掃描成 PDF 的表格上傳到這個網頁，程式會：

1. 上方先選擇表單類型：**供應商資料表** 或 **供應商評核表**
2. 把 PDF 轉成圖片
3. 依照對應的 `templates/*.json` 裡校正好的座標，裁出每個欄位
4. 文字欄位送 Google Cloud Vision API 辨識(支援手寫)；勾選框用「黑色像素比例」判斷有沒有打勾
5. 畫面顯示「裁切小圖 + 辨識建議值」，同仁核對、修正
6. 按確認後，寫入 Google Drive：
   - **供應商資料表**：`供應商主檔/統編_公司名稱.xlsx`(每次提交新增一列)，`彙總總表.xlsx`(每家供應商固定一列，重複提交覆蓋更新)
   - **供應商評核表**：`供應商評核紀錄/供應商名稱_評核紀錄.xlsx`(每次評核新增一列)，`評核彙總表.xlsx`(每次評核也新增一列，保留所有供應商所有評核的完整歷史，方便日後統計)

## 檔案說明

| 檔案 | 用途 |
|---|---|
| `streamlit_app.py` | 主程式(含表單類型切換) |
| `drive_utils.py` | 存取 Google Drive |
| `vision_utils.py` | 呼叫 Vision API 做文字辨識；用像素比例判斷勾選框 |
| `master_utils.py` | 主檔/評核紀錄/彙總表的讀寫邏輯 |
| `templates/supplier_form.json` | 供應商資料表的欄位座標校正檔 |
| `templates/eval_form.json` | 供應商評核表的欄位座標校正檔(2頁、44個欄位) |
| `requirements.txt` | 套件清單 |

## ⚠️ 目前的測試狀況

- ✅ 兩份模板的欄位座標都已對照實際文件(或轉出的乾淨掃描版面)逐一目視驗證，位置精準
- ✅ 供應商資料表的勾選偵測、供應商評核表的評核類別勾選偵測，都已用模擬/真實資料測試
- ✅ 兩種表單各自的存檔邏輯(新增列/覆蓋更新列)都已用模擬資料完整測試
- ⏳ **供應商評核表的座標校正，是用電腦直接轉出的乾淨 PDF 做的，不是真正掃描機掃出來的紙本**，跟供應商資料表當初的校正方式不同(供應商資料表當初有拿到您真實的掃描檔驗證過，評核表目前還沒有)。正式使用前，務必先用一份真正手寫填寫、掃描過的評核表測試，確認校正在真實掃描狀況下仍然準確，如果有明顯偏移，需要重新校正 `templates/eval_form.json` 裡的座標。
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
