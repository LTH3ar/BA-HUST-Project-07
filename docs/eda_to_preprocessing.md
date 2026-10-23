# Từ EDA đến Preprocessing: Các phát hiện và quyết định xử lý dữ liệu

**Dự án:** Profit-Aware Telecom Churn & Retention Targeting
**Dữ liệu:** IBM Telco Customer Churn, 7,043 khách × 21 cột

**Tài liệu liên quan:**
- [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb): phân tích khám phá
- [`notebooks/02_preprocessing.ipynb`](../notebooks/02_preprocessing.ipynb): chạy và kiểm tra preprocessing
- [`src/preprocessing.py`](../src/preprocessing.py): toàn bộ logic xử lý
- [`src/config.yaml`](../src/config.yaml): tham số

---

## 1. Mục đích tài liệu

Tài liệu này giải thích **vì sao** dữ liệu được xử lý như hiện tại. Mỗi quyết định preprocessing đều xuất phát từ một quan sát cụ thể trong EDA. Mỗi quyết định được trình bày theo cùng một khung:

> **Quan sát** (EDA thấy gì, kèm con số) → **Quyết định** (xử lý thế nào) → **Lý do** và **các phương án đã cân nhắc**

Phạm vi của bước preprocessing gồm: làm sạch dữ liệu, feature engineering, chia train/val/test, kiểm tra tuyến tính với log-odds, và định nghĩa sẵn encoding/scaling. Hai việc **cố ý để lại cho bước mô hình** là xử lý mất cân bằng lớp và chọn feature (xem mục 6).

---

## 2. Tóm tắt nhanh

| # | Phát hiện từ EDA | Quyết định preprocessing |
|---|---|---|
| 1 | `TotalCharges` có kiểu chuỗi, chứa 11 giá trị rỗng, tất cả ở khách `tenure = 0` | Ép sang số, điền **0** |
| 2 | `SeniorCitizen` mã hoá 0/1, các cột nhị phân khác lại dùng Yes/No | Đổi tất cả cột Yes/No sang 0/1 |
| 3 | `"No internet service"` / `"No phone service"` trùng 100% với cột dịch vụ gốc | Gộp về `"No"` |
| 4 | Không có ngoại lai, không có nhóm hiếm, 22 dòng trùng là khách khác nhau | **Không** xử lý ngoại lai, không gộp nhóm, không xoá dòng trùng |
| 5 | Churn tập trung ở khách mới (0–12 tháng), giảm không đều theo tenure | Giữ `tenure` **dạng số**; tạo cờ nghiệp vụ `is_new_customer`. *(`tenure_group` đã thử và bị bỏ, xem mục 4.1)* |
| 6 | Dịch vụ bảo vệ (OnlineSecurity / TechSupport) gắn với churn thấp | Tạo `has_protection`, `num_addons`, `has_streaming` |
| 7 | Thanh toán tự động churn thấp hơn nhiều so với Electronic check | Tạo `auto_payment` |
| 8 | `TotalCharges` ≈ `tenure × MonthlyCharges` (r = 0.9996), gây đa cộng tuyến | Tạo `avg_past_charge`, `charge_change` làm phương án thay thế |
| 9 | Hai tổ hợp M2M × Fiber và M2M × Electronic check có churn hơn 50% | Tạo biến tương tác `m2m_fiber`, `m2m_echeck` |
| 10 | Churn tăng theo `MonthlyCharges` đến khoảng $100 rồi **giảm lại** ở decile cao nhất, quan hệ không đơn điệu | Tạo `monthly_charge_band` (4 nhóm) |
| 11 | `TotalCharges` lệch phải, churn dồn ở vùng thấp | Tạo `log_TotalCharges` |
| 12 | Mất cân bằng lớp ở mức vừa phải (2.8 : 1) | Chia dữ liệu **stratified** 60/20/20 |
| 13 | Có 4 biến phân loại nhiều mức (cùng `gender` 2 mức), biến số có thang đo rất khác nhau | One-hot và scaling đặt **trong pipeline**, fit trên train |

Các dòng 5, 10 và 11 được **kiểm tra lại bằng kiểm định tuyến tính với log-odds** trên tập train (mục 4.6). Kết quả: dòng 10 khắc phục được gần hoàn toàn, dòng 11 khắc phục được một phần. Với dòng 5, `tenure` gần thẳng trên thang log-odds, nên biến chia khoảng `tenure_group` không giúp được gì và đã bị bỏ.

---

## 3. Phát hiện về chất lượng dữ liệu → Làm sạch

### 3.1 `TotalCharges` bị đọc thành chuỗi và có 11 giá trị rỗng

**Quan sát (EDA mục 2.1–2.2)**
- `TotalCharges` có kiểu `object`. Không có giá trị NaN, nhưng có **11 chuỗi rỗng `" "`**.
- Cả 11 dòng đều là khách **`tenure = 0`**, và đây cũng là tất cả khách có `tenure = 0` trong dữ liệu.
- Cả 11 khách đều `Churn = No`; 10 khách có hợp đồng Two year, 1 khách One year.

Diễn giải: đây là khách **vừa ký hợp đồng, chưa có hoá đơn nào**. Giá trị rỗng không phải lỗi ngẫu nhiên mà có lý do nghiệp vụ.

**Quyết định:** ép sang số và **điền 0**.

**Lý do và các phương án đã cân nhắc**

| Phương án | Đánh giá |
|---|---|
| **Điền 0** ✅ | Đúng về nghiệp vụ: chưa có hoá đơn thì tổng tiền đã trả bằng 0. Đây là quy tắc cố định, không tính từ dữ liệu, nên không gây leakage. Giữ đủ khách cho bảng targeting |
| Xoá 11 dòng | Chỉ mất 0.16% dữ liệu, nhưng làm mất khách khỏi bảng targeting cuối cùng mà không có lý do |
| Điền `MonthlyCharges` | Giả định khách đã trả 1 tháng, mâu thuẫn với `tenure = 0` |
| Điền trung bình hoặc trung vị | Sai về nghiệp vụ (trung vị là $1,397 cho khách chưa trả đồng nào). Ngoài ra phải tính từ tập train nên dễ gây leakage |

**Biện pháp an toàn:** `clean_data()` **báo lỗi** nếu gặp `TotalCharges` thiếu ở khách có `tenure > 0`. Như vậy quy tắc "điền 0" không âm thầm áp dụng sai nếu dữ liệu thay đổi.

### 3.2 Mã hoá nhị phân không đồng nhất

**Quan sát (EDA mục 1.2, 2.2):** `SeniorCitizen` dùng 0/1, trong khi 11 cột nhị phân khác (`Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, các cột dịch vụ…) dùng `"Yes"` / `"No"`.

**Quyết định:** đổi tất cả cột Yes/No sang **1/0**. `Churn` được đổi thành cột nhãn `y` (0/1).

**Lý do:** đồng nhất kiểu dữ liệu, giúp tính tỉ lệ và tạo biến mới (tổng, max) trực tiếp. Hàm báo lỗi nếu gặp giá trị ngoài Yes/No.

### 3.3 Nhãn `"No internet service"` và `"No phone service"` trùng thông tin

**Quan sát (EDA mục 2.4b, 5.4)**
- 6 cột dịch vụ bổ sung có mức thứ ba `"No internet service"` (1,526 khách). `MultipleLines` có mức `"No phone service"` (682 khách).
- Kiểm tra logic cho thấy **0 vi phạm**: các mức này trùng **100%** với `InternetService = No` / `PhoneService = No`.
- Trong bảng xếp hạng driver, 7 nhóm `"No internet service"` có cùng churn rate 7.4% và cùng số khách, tức chỉ là **một thông tin lặp lại 7 lần**.

**Quyết định:** gộp `"No internet service"` và `"No phone service"` về `"No"` (sau đó đổi thành 0).

**Lý do**
- Thông tin "không có internet" đã nằm trong `InternetService`, nên không mất gì khi gộp.
- Nếu giữ lại, khi one-hot sẽ sinh ra **7 cột dummy giống hệt nhau** và trùng với `InternetService_No`. Điều này gây **đa cộng tuyến hoàn toàn** cho logistic regression (hệ số không xác định được).
- Các cột dịch vụ trở thành nhị phân, nên dễ diễn giải hơn: "khách có dùng OnlineSecurity không".

### 3.4 Những thứ quyết định **không** xử lý

| Quan sát (EDA) | Quyết định | Lý do |
|---|---|---|
| Không có ngoại lai theo quy tắc 1.5×IQR ở `tenure`, `MonthlyCharges`, `TotalCharges` (mục 2.5) | Không cắt hay winsorize | Không có gì để xử lý. Các giá trị lớn là khách lâu năm và trả cước cao, đúng là nhóm có giá trị |
| 22 dòng trùng toàn bộ đặc trưng (42 dòng liên quan), nhưng `customerID` khác nhau, đều là hợp đồng Month-to-month (mục 2.3) | Giữ nguyên | Là các khách khác nhau có profile đơn giản giống nhau, không phải lỗi nhập liệu |
| Nhóm nhỏ nhất trong các biến phân loại có 682 khách (mục 1.2) | Không gộp nhóm hiếm | Không có nhóm nào quá hiếm |
| `customerID` là cột định danh (mục 1) | Giữ làm **khoá**, không dùng làm feature | Cần để ghép điểm rủi ro với bảng targeting |

---

## 4. Phát hiện về driver churn → Feature engineering

**Nguyên tắc:** chỉ tạo biến khi EDA cho thấy **một tín hiệu cụ thể** mà các biến gốc thể hiện chưa trọn. Bước này tạo **tất cả biến ứng viên**; việc chọn dùng biến nào để lại cho bước mô hình.

Mỗi mục dưới đây gồm tín hiệu quan sát được trong EDA và kết quả kiểm tra lại sau khi tạo biến (notebook 02, mục 2.2 và 4).

### 4.1 Vòng đời khách: `tenure` dạng số, `is_new_customer` (đã bỏ `tenure_group`)

**Quan sát (EDA mục 4.2, 5.2, 6.3)**
- Khách churn có median tenure là **10 tháng**, so với **38 tháng** ở khách ở lại (Mann-Whitney, r = −0.48).
- Churn giảm **không đều** theo tenure: 52.9% (0–6 tháng), 35.9% (7–12), 28.7% (13–24), 20.4% (25–48), 9.5% (49–72).
- Phân phối `tenure` có dạng **hai đỉnh**, tập trung ở khách rất mới và khách rất lâu năm.

**Hướng xử lý ban đầu**
- Tạo `tenure_group` gồm 5 khoảng 0-6 / 7-12 / 13-24 / 25-48 / 49-72 tháng, với ý định giúp logistic regression nắm được quan hệ phi tuyến.
- `is_new_customer = tenure ≤ 12`: cờ đơn giản cho giai đoạn rủi ro nhất.

**Kiểm tra sau khi tạo** (notebook 02 mục 4.1, AIC của LR trên train)

| Dạng | AIC | So với `tenure` dạng số |
|---|---|---|
| `tenure` dạng số | 4,315 | — |
| `tenure_group` (thay cho `tenure`) | 4,343 | **+28, kém hơn** |
| `tenure` + `tenure_group` | 4,301 | −14 (cần thêm 4 cột dummy) |
| `tenure` + `is_new_customer` | 4,314 | −1, gần như không thêm gì |
| 10 decile (mức tham chiếu) | 4,291 | −24 |

Đường churn rate trong EDA trông phi tuyến vì nó nằm trên **thang xác suất**. Trên **thang log-odds**, thang mà LR thực sự làm việc, `tenure` gần như thẳng; chỉ đoạn 65–72 tháng dốc mạnh hơn.

**Quyết định cuối cùng**
- **Bỏ `tenure_group`.** Thay cho `tenure` thì kém hơn, dùng kèm thì chỉ cải thiện −14 với 4 cột dummy thêm. Mô hình cây tự tìm ngưỡng cắt nên cũng không cần. Lý do "dễ đọc cho playbook" không đủ: phân khúc trong EDA dùng cách chia khác (0–12 / 13–24 / 25–72), còn "khách năm đầu" đã có `is_new_customer`.
- **Giữ `tenure` dạng số.**
- **Giữ `is_new_customer` làm cờ nghiệp vụ.** Đứng riêng thì có tín hiệu: churn **47.4%** so với **17.1%** (lift 1.79×). Nhưng thông tin đã nằm hết trong `tenure` (dùng kèm chỉ cải thiện −1). Cờ này phục vụ phân khúc và playbook; khi chọn feature cho LR **không dùng chung** với `tenure`. Quan hệ này được ghi trong `BUSINESS_FLAG_COLS` của `preprocessing.py`.

### 4.2 Dịch vụ: `has_protection`, `num_addons`, `has_streaming`

**Quan sát (EDA mục 5.1, 5.3)**
- `OnlineSecurity` và `TechSupport` là hai driver mạnh nhất trong nhóm dịch vụ (Cramér's V = 0.35 và 0.34). Khách không có các dịch vụ này churn khoảng **42%**, khách có chỉ khoảng **15%**.
- `OnlineBackup` và `DeviceProtection` có xu hướng tương tự nhưng yếu hơn. `StreamingTV` và `StreamingMovies` có liên hệ yếu (V = 0.23) và hành vi gần như giống nhau.

**Quyết định**
- `has_protection`: có `OnlineSecurity` **hoặc** `TechSupport`. Gộp hai driver mạnh nhất thành một cờ dễ hiểu.
- `num_addons`: số dịch vụ bổ sung đang dùng (0–6), đo mức độ gắn bó của khách.
- `has_streaming`: có `StreamingTV` **hoặc** `StreamingMovies`. Gộp hai biến có cùng hành vi.

**Kiểm tra sau khi tạo**
- `has_protection = 1` có churn 17.1%, so với 33.4%.
- `has_streaming = 1` có churn 30.3%, so với 22.8%.
- `num_addons` **không đơn điệu**: 0 → 21.4%, 1 → 45.8%, 2 → 35.8%, … 6 → 5.3%. Nhóm 0 chứa 1,526 khách không dùng internet (churn 7.4%). Điểm này cần lưu ý khi làm mô hình (xem mục 7).

### 4.3 Thanh toán: `auto_payment`

**Quan sát (EDA mục 5.1, 7.2)**
- Electronic check có churn **45.3%**, lift cao nhất trong bảng xếp hạng driver của EDA (1.71×), và chiếm **61% MRR bị mất**.
- Hai hình thức thanh toán tự động có churn thấp: Bank transfer 16.7%, Credit card 15.2%. Mailed check là 19.1%.

**Quyết định:** `auto_payment` = 1 nếu thanh toán bằng Bank transfer hoặc Credit card (automatic).

**Lý do:** điều quan trọng là **khách thanh toán tự động hay thủ công**, không nằm ở việc chọn ngân hàng hay thẻ. Cờ này cũng gợi ý một hành động giữ chân cụ thể: khuyến khích chuyển sang thanh toán tự động.

**Kiểm tra sau khi tạo:** `auto_payment = 1` có churn 16.0%, so với 34.7%.

### 4.4 Cước phí: `avg_past_charge`, `charge_change`

**Quan sát (EDA mục 2.4a, 6.1)**
- `TotalCharges` gần như bằng `tenure × MonthlyCharges` (Pearson r = **0.9996**).
- VIF: `TotalCharges` 9.53, `tenure` 5.84, `MonthlyCharges` 3.23. Đây là **đa cộng tuyến** rõ rệt, gây bất ổn cho hệ số của logistic regression.
- Tuy vậy, khoảng **5.4% khách** có `TotalCharges` lệch hơn 10% so với `tenure × MonthlyCharges`, nghĩa là họ đã **thay đổi gói cước** trong quá khứ. Đây là thông tin mà `TotalCharges` mang thêm so với hai biến kia.

**Quyết định**
- `avg_past_charge = TotalCharges / tenure`: cước trung bình trong quá khứ. Khách `tenure = 0` lấy bằng `MonthlyCharges`, vì chưa có lịch sử.
- `charge_change = MonthlyCharges − avg_past_charge`: cước hiện tại cao hơn hay thấp hơn trước đây.

**Lý do:** ở bước mô hình, nếu cần bỏ `TotalCharges` cho logistic regression, `charge_change` giữ lại phần thông tin riêng ("có tăng hoặc giảm cước") mà không gây đa cộng tuyến.

**Kiểm tra sau khi tạo**
- So sánh trung bình thì `charge_change` gần 0 ở cả hai nhóm; khách churn chỉ phân tán rộng hơn (std 3.33 so với 2.30). Kết luận ban đầu "tín hiệu yếu" là **sai**.
- Kiểm tra tuyến tính (mục 4.6) cho thấy biến này có **tín hiệu mạnh dạng chữ V**: cả tăng cước lẫn giảm cước đều gắn với churn cao (ΔAIC −228 khi chia decile). Phép so sánh trung bình đã che mất hình dạng này.
- Điểm nhô tại `charge_change = 0` là 387 khách trên train: 379 khách tenure = 1 và 7 khách tenure = 0, vì khi đó `TotalCharges = MonthlyCharges`. Nhóm này churn 62%. Như vậy biến này một phần lẫn với `tenure`.
- Xử lý dạng chữ V không xuất phát từ EDA, nên chưa làm ở bước này. Đây là ghi chú cho bước mô hình (mục 7).

### 4.5 Tương tác: `m2m_fiber`, `m2m_echeck`

**Quan sát (EDA mục 6.2, 7.3)**
- Hợp đồng Month-to-month **và** Fiber optic có churn **55%** (n = 2,128). Riêng nhóm Fiber với hợp đồng 2 năm chỉ có 7%.
- Hợp đồng Month-to-month **và** Electronic check có churn **54%** (n = 1,850).
- Ba profile M2M × Fiber chiếm **72% MRR bị mất**, là phân khúc ưu tiên số 1.

**Quyết định:** tạo hai cờ tương tác `m2m_fiber` và `m2m_echeck`.

**Lý do:** logistic regression chỉ cộng dồn tác động của từng biến, nên **không tự học được** tương tác. Mô hình cây thì tự học được, nên hai cờ này chủ yếu phục vụ LR. Cả hai cũng khớp trực tiếp với phân khúc trong playbook.

**Kiểm tra sau khi tạo:** `m2m_fiber = 1` có churn 54.6%, so với 14.4% (lift 2.06×). `m2m_echeck = 1` có churn 53.7%, so với 16.8% (lift 2.02×). Đây là hai biến có lift cao nhất trong toàn bộ dữ liệu.

### 4.6 Quan hệ phi tuyến với log-odds: `monthly_charge_band`, `log_TotalCharges`

Logistic regression giả định mỗi biến số **tuyến tính với log-odds** của churn. Tương quan Pearson với nhãn 0/1 không kiểm tra được giả định này. Mô hình cây không cần giả định này.

**Quan sát (EDA)**
- **`MonthlyCharges` (mục 7.1, biểu đồ f16):** churn theo decile tăng từ 8.5% ($20) lên khoảng 41% ($98), rồi **giảm còn 24.7%** ở decile cao nhất ($107). KDE ở mục 5.2 có hai đỉnh: khách ở lại tập trung ở khoảng $20, khách churn ở khoảng $80. Nếu chỉ chia 3 mức Low/Med/High thì churn trông như tăng đều (15.9% → 29.7% → 34.1%), vì cách chia này **che mất** đoạn giảm ở nhóm cước cao nhất. Insight #6 và ghi chú #12 của EDA đã được cập nhật theo điểm này.
- **`TotalCharges` (mục 4.2, 5.2):** lệch phải (skew 0.96), khách churn dồn ở vùng thấp. EDA mục 8.2 #5 đã gợi ý log-transform.

**Quyết định**
- `monthly_charge_band`: chia `MonthlyCharges` thành 4 khoảng **≤25 / 25-70 / 70-100 / >100**. Ngưỡng làm tròn từ các mốc decile trong EDA (25.05 / 70.35 / 102.6) và đặt trong `config.yaml`. Churn theo nhóm: 9.0% / 23.1% / 37.8% / **28.0%**, đúng dạng tăng rồi giảm.
- `log_TotalCharges = log(1 + TotalCharges)`. Khách `tenure = 0` (`TotalCharges = 0`) cho giá trị 0.

**Kiểm tra tuyến tính** (notebook 02 mục 4, **chỉ trên tập train**). Mỗi biến được kiểm tra bằng:
- **Empirical logit:** log-odds thực tế theo decile, so với đường thẳng của LR một biến. Xem biểu đồ [`p02_linearity_logit.png`](../outputs/figures/p02_linearity_logit.png).
- **So sánh AIC:** dạng tuyến tính so với dạng 10 decile (đường cong tự do).
- **Kiểm định Box-Tidwell:** thành phần `x·ln(x)`.

| Biến | AIC tuyến tính | AIC 10 decile | ΔAIC | Box-Tidwell p | Hình dạng log-odds |
|---|---|---|---|---|---|
| `tenure` | 4,315 | 4,291 | −24 | 6e-4 | Gần thẳng, chỉ dốc mạnh ở 65–72 tháng |
| `MonthlyCharges` | 4,726 | 4,618 | −108 | 2e-15 | Tăng đến khoảng $98 rồi giảm |
| `TotalCharges` | 4,702 | 4,641 | −61 | 6e-8 | Giảm nhanh ở vùng thấp rồi chậm dần |
| `avg_past_charge` | 4,727 | 4,639 | −88 | 9e-15 | Giống `MonthlyCharges` |
| `charge_change` | 4,893 | 4,665 | −228 | 4e-10 | Hình chữ V quanh 0 |
| `num_addons` | 4,859 | 4,651 | −208 | 2e-38 | Đỉnh tại 1 rồi giảm dần; nhóm 0 lẫn khách không có internet |

Kết luận: **cả 6 biến số đều không tuyến tính với log-odds.**

**Biến đổi từ EDA có khắc phục được không?**

| Biến gốc | Dạng | AIC | So với tuyến tính | Đánh giá |
|---|---|---|---|---|
| `MonthlyCharges` | `monthly_charge_band` (4 nhóm) | 4,619 | −106 | ✅ Gần như bằng mức 10 decile (4,618), chỉ với 4 nhóm |
| `TotalCharges` | `log_TotalCharges` | 4,668 | −33 | ⚠️ Khắc phục khoảng một nửa (decile là 4,641) |
| `tenure` | `tenure_group` (5 nhóm) | 4,343 | **+28** | ❌ Kém hơn dạng tuyến tính, **đã bỏ** (chi tiết ở mục 4.1) |

**Lưu ý**
- Đây là kiểm tra **từng biến riêng lẻ**. Khi có mô hình LR đầy đủ, cần kiểm tra lại bằng partial residual plot.
- Ngưỡng của `monthly_charge_band` được chọn từ EDA trên toàn bộ dữ liệu. Vì đây là các mốc làm tròn và mô hình không học trực tiếp từ đó, rủi ro leakage là nhỏ. Dù vậy, kiểm tra AIC ở trên chỉ dùng tập train.

---

## 5. Phát hiện về cấu trúc dữ liệu → Chia tập và encoding

### 5.1 Chia train / val / test: stratified 60 / 20 / 20

**Quan sát (EDA mục 3):** tỉ lệ churn là 26.5% (1,869 / 7,043), tức khoảng 2.8 khách ở lại cho mỗi khách churn.

**Quyết định:** chia **stratified theo `y`**, tỉ lệ 60 / 20 / 20, `random_state = 42`. Kết quả:

| Tập | Số khách | Churn rate | Dùng cho |
|---|---|---|---|
| train | 4,225 | 26.53% | Fit preprocessor và mô hình, cross-validation |
| val | 1,409 | 26.54% | Calibration, chọn ngưỡng, chạy thử simulator targeting |
| test | 1,409 | 26.54% | Chỉ dùng **một lần** để báo cáo kết quả cuối |

**Lý do**
- **Stratify** để tỉ lệ churn ở mọi tập đều giống tổng thể. Với 1,409 khách, chia ngẫu nhiên thông thường có độ lệch chuẩn khoảng 1.2 điểm %, tức có thể lệch 2–3 điểm %. Đây **không** phải là xử lý mất cân bằng.
- **Cần tập val riêng** vì dự án yêu cầu **calibration** (công thức EV dùng trực tiếp `p_churn`) và chọn ngưỡng. Nếu calibrate trên test thì kết quả cuối sẽ bị lạc quan quá mức.
- Bảng `customerID → split` được lưu ra `data/processed/splits.csv`, để cả nhóm dùng **cùng một cách chia**.

### 5.2 Encoding và scaling: đặt trong pipeline, không lưu sẵn vào CSV

**Quan sát (EDA mục 1, 4)**
- Có 5 biến phân loại: `gender`, `Contract`, `InternetService`, `PaymentMethod`, và `monthly_charge_band` mới tạo.
- Các biến số có thang đo rất khác nhau: `tenure` từ 0 đến 72, `MonthlyCharges` từ 18 đến 119, `TotalCharges` từ 0 đến 8,685. `TotalCharges` lệch phải (skew = 0.96).

**Quyết định:** hàm `build_preprocessor(num_cols, cat_cols, bin_cols, scale, drop_first)` tạo một `ColumnTransformer`. Biến số đi qua `StandardScaler`, biến phân loại đi qua `OneHotEncoder`, biến nhị phân giữ nguyên. Preprocessor được đặt trong `Pipeline` cùng mô hình và **chỉ fit trên train**.

**Lý do không one-hot sẵn rồi lưu vào CSV**
1. **Mỗi mô hình cần một kiểu encoding khác nhau.** Logistic regression cần `drop_first=True` để tránh dummy trap. Mô hình cây nên giữ đủ các mức. Lưu sẵn một kiểu sẽ khiến kiểu kia phải sửa lại.
2. **Chưa chọn feature.** Bỏ một biến gốc chỉ là bỏ 1 tên cột khỏi danh sách, thay vì phải tìm và bỏ nhiều cột dummy.
3. **Scaling bắt buộc phải fit trên train**, nếu không sẽ leakage. Đặt one-hot và scaling chung trong pipeline thì cả hai được fit lại đúng cách trong từng fold cross-validation.
4. **Các bước sau cần dữ liệu dạng dễ đọc.** Value proxy, bảng targeting, phân khúc và dashboard cần `Contract = "Month-to-month"` dạng chữ, không phải các cột dummy.

**Kiểm tra (notebook 02 mục 5.1):** dùng toàn bộ cột ứng viên, cấu hình kiểu LR cho ra 36 cột, cấu hình kiểu mô hình cây cho ra 41 cột. Số cột giống nhau giữa train, val và test, không có NaN. Trên train, biến số sau khi scale có mean = 0 và std = 1.

**Mức tham chiếu khi `drop_first=True`:** `OneHotEncoder` bỏ mức đứng đầu theo thứ tự chữ cái. Vì vậy hệ số LR được đọc so với: `Female`, `Month-to-month`, `DSL`, `Bank transfer (automatic)`, và `25-70` (với `monthly_charge_band`, vì chuỗi `"25-70"` đứng trước `"<=25"`).

---

## 6. Những việc cố ý **chưa** làm, và lý do

| Việc | Vì sao để lại cho bước mô hình |
|---|---|
| **Xử lý mất cân bằng lớp** | Mức mất cân bằng chỉ vừa phải (2.8 : 1). Quan trọng hơn, `class_weight` hay resample (SMOTE) đều **làm lệch xác suất dự đoán**, trong khi công thức EV cần xác suất đúng. Phải chọn cách xử lý cùng lúc với chiến lược calibration |
| **Chọn feature** | Còn phụ thuộc loại mô hình: LR cần loại các cặp biến tương quan cao, mô hình cây thì không. Bước preprocessing giữ lại **tất cả** 30 cột ứng viên, gồm cả dạng gốc và dạng đã biến đổi của cùng một biến |
| Quyết định về `gender` | Không có tín hiệu (p = 0.49) và có rủi ro về fairness. Đây là quyết định chọn feature, sẽ chốt cùng phần thảo luận fairness |

---

## 7. Ghi chú chuyển tiếp cho bước mô hình

**Các cặp biến tương quan cao (|r| > 0.7)**, cần cân nhắc khi chọn feature cho logistic regression:

| Cặp | \|r\| | Gợi ý |
|---|---|---|
| `MonthlyCharges` – `avg_past_charge` | 0.996 | Gần như trùng nhau, chỉ giữ một biến |
| `tenure` – `log_TotalCharges` / `TotalCharges` | 0.829 / 0.826 | Với LR cân nhắc bỏ nhóm `TotalCharges` |
| `StreamingMovies` / `StreamingTV` – `has_streaming` | 0.80 / 0.80 | Chọn biến gốc **hoặc** biến gộp |
| `log_TotalCharges` – `is_new_customer` | 0.797 | Cùng mang thông tin về độ "mới" của khách |
| `tenure` – `is_new_customer` | 0.755 | Với LR dùng `tenure`; `is_new_customer` là cờ nghiệp vụ |
| `TechSupport` / `OnlineSecurity` – `has_protection` | 0.75 / 0.74 | Chọn biến gốc **hoặc** biến gộp |
| `num_addons` – `TotalCharges` / `MonthlyCharges` / `has_streaming` | 0.70–0.75 | Lưu ý khi dùng cùng nhau |

**Dạng biến nên dùng cho logistic regression** (từ kiểm tra tuyến tính, mục 4.6)

| Biến | Nên dùng cho LR | Mô hình cây |
|---|---|---|
| `MonthlyCharges` | `monthly_charge_band` | Dạng gốc |
| `TotalCharges` | `log_TotalCharges` (nếu giữ nhóm biến này) | Dạng gốc |
| `tenure` | `tenure` dạng số, không dùng chung với `is_new_customer` | Dạng gốc |
| `num_addons` | Dạng phân loại, hoặc dùng kèm `InternetService` | Dạng gốc |
| `charge_change` | Cần biến đổi, ví dụ lấy độ lớn \|charge_change\|, rồi kiểm tra lại cùng `tenure`. Chưa làm vì không xuất phát từ EDA | Dạng gốc |

**Các lưu ý khác**
- **`num_addons` không đơn điệu**, vì nhóm 0 dịch vụ bổ sung lẫn 1,526 khách không dùng internet (churn 7.4%).
- **`charge_change` có tín hiệu dạng chữ V**, không phải tín hiệu yếu như đánh giá ban đầu. Giá trị 0 gần như chỉ gồm khách tenure ≤ 1.
- **`SeniorCitizen`** có tín hiệu mạnh (churn 41.7%, lift 1.57×), nhưng cần **kiểm tra fairness**: tỉ lệ nhóm này được nhận ưu đãi so với nhóm khác.
- **Giới hạn chung** (từ EDA mục 8.3): các quan hệ trên chỉ là **tương quan, không phải nhân quả**. Dữ liệu là snapshot của một bộ dữ liệu mẫu, hư cấu, nên mọi kết luận có giới hạn về khả năng áp dụng cho nhà mạng thực tế (external validity).

---

## 8. Đầu ra của bước preprocessing

| File | Nội dung |
|---|---|
| `data/processed/telco_clean.csv` | 7,043 dòng × 32 cột: `customerID`, `y`, 19 cột gốc đã làm sạch, 11 biến mới. **Chưa scale, chưa encode** |
| `data/processed/splits.csv` | `customerID → train / val / test` |
| `src/preprocessing.py` | `clean_data()`, `add_features()`, `assign_splits()`, `build_preprocessor()`, các danh sách `NUM_COLS` / `CAT_COLS` / `BIN_COLS`, và `BUSINESS_FLAG_COLS` |
| `src/config.yaml` | `random_state`, tỉ lệ chia, ngưỡng `is_new_customer`, ngưỡng chia `MonthlyCharges` |
| `outputs/figures/p01_engineered_flags.png`, `p02_linearity_logit.png` | Churn theo các cờ mới; empirical logit của các biến số |

**Cách dùng ở bước mô hình**

```python
from src import preprocessing as pp

config = pp.load_config()
data = pp.load_processed(config)                    # có cột `split`
train = data[data["split"] == "train"]

prep = pp.build_preprocessor(num_cols=[...], cat_cols=[...], bin_cols=[...],
                             scale=True, drop_first=True)
# Pipeline([("prep", prep), ("clf", LogisticRegression())]).fit(train, train[pp.TARGET])
```
