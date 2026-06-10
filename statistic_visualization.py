# ============================================================
# STATISTIC & VISUALIZATION CHO DỮ LIỆU REVIEW PHIM
# Dùng cho báo cáo / slide thuyết trình
#
# Yêu cầu:
# - pandas
# - matplotlib
# - seaborn
# - wordcloud, nếu muốn tạo word cloud
#
# Cài thư viện nếu cần:
# pip install pandas matplotlib seaborn wordcloud
#
# Giả định chính: dữ liệu đã được lưu trong DataFrame tên là df.
# Tuy nhiên, file này cũng có phần load trực tiếp từ:
# - data_global(1).db hoặc data_global.db
# - data_local(1).db hoặc data_local.db
# ============================================================

import sqlite3
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CẤU HÌNH CHUNG
# ============================================================

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11

OUTPUT_DIR = Path("statistic_visualization_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# HÀM HỖ TRỢ LOAD DATABASE
# ============================================================

def find_existing_file(possible_paths):
    """
    Tìm file tồn tại đầu tiên trong danh sách đường dẫn.
    """
    for path in possible_paths:
        path = Path(path)
        if path.exists():
            return path
    return None


def read_sqlite_table(db_path):
    """
    Đọc bảng đầu tiên trong SQLite database.
    Ưu tiên bảng imdb_reviews hoặc youtube_comments nếu có.
    """
    conn = sqlite3.connect(db_path)

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';",
        conn
    )

    if tables.empty:
        conn.close()
        raise ValueError(f"Database {db_path} không có bảng nào.")

    table_names = tables["name"].tolist()

    if "imdb_reviews" in table_names:
        table_name = "imdb_reviews"
    elif "youtube_comments" in table_names:
        table_name = "youtube_comments"
    else:
        table_name = table_names[0]

    data = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)
    conn.close()

    return data, table_name


def build_dataframe_from_databases():
    """
    Hàm này dùng khi bạn muốn chạy trực tiếp từ hai file database.
    Nếu bạn đã có DataFrame df rồi, có thể bỏ qua phần này.
    """

    global_db = find_existing_file([
        "data_global(1).db",
        "data_global.db",
        r"D:\Convert_Data\data_global(1).db",
        r"D:\Convert_Data\data_global.db"
    ])

    local_db = find_existing_file([
        "data_local(1).db",
        "data_local.db",
        r"D:\Convert_Data\data_local(1).db",
        r"D:\Convert_Data\data_local.db"
    ])

    if global_db is None and local_db is None:
        raise FileNotFoundError(
            "Không tìm thấy file database. Nếu bạn đã có DataFrame df, "
            "hãy chạy code từ phần BƯỚC 1 trở đi trong notebook/script của bạn."
        )

    frames = []

    if global_db is not None:
        imdb_df, imdb_table = read_sqlite_table(global_db)
        print(f"Đã đọc global database: {global_db}, bảng: {imdb_table}")

        temp = pd.DataFrame()
        temp["movie_name"] = imdb_df["movie_name"] if "movie_name" in imdb_df.columns else np.nan
        temp["review"] = imdb_df["review_text"] if "review_text" in imdb_df.columns else imdb_df.get("clean_text", "")
        temp["clean_text"] = imdb_df["clean_text"] if "clean_text" in imdb_df.columns else temp["review"]
        temp["rating"] = pd.to_numeric(imdb_df["rating"], errors="coerce") if "rating" in imdb_df.columns else np.nan
        temp["sentiment"] = np.nan
        temp["source_platform"] = "IMDb"
        temp["language"] = imdb_df["language"] if "language" in imdb_df.columns else np.nan
        frames.append(temp)

    if local_db is not None:
        yt_df, yt_table = read_sqlite_table(local_db)
        print(f"Đã đọc local database: {local_db}, bảng: {yt_table}")

        temp = pd.DataFrame()
        temp["movie_name"] = yt_df["movie_name"] if "movie_name" in yt_df.columns else np.nan
        temp["review"] = yt_df["comment_text"] if "comment_text" in yt_df.columns else yt_df.get("clean_text", "")
        temp["clean_text"] = yt_df["clean_text"] if "clean_text" in yt_df.columns else temp["review"]
        temp["rating"] = np.nan
        temp["sentiment"] = np.nan
        temp["source_platform"] = "YouTube"
        temp["language"] = yt_df["language"] if "language" in yt_df.columns else np.nan
        frames.append(temp)

    return pd.concat(frames, ignore_index=True)


# ============================================================
# HÀM HỖ TRỢ PHÂN TÍCH
# ============================================================

def get_mode(series):
    """
    Tính mode, tức giá trị xuất hiện nhiều nhất.
    Nếu có nhiều mode, lấy giá trị đầu tiên.
    """
    mode_values = series.dropna().mode()
    if len(mode_values) == 0:
        return np.nan
    return mode_values.iloc[0]


def tokenize_text(text):
    """
    Tách từ đơn giản, dùng được cho tiếng Anh và tiếng Việt.
    """
    text = str(text).lower()
    return re.findall(r"[a-zA-ZÀ-ỹ0-9']+", text)


def assign_sentiment_from_rating(rating):
    """
    Tạo sentiment đơn giản từ rating.
    Quy ước:
    - 1 đến 4: Negative
    - 5 đến 6: Neutral
    - 7 đến 10: Positive
    """
    if pd.isna(rating):
        return np.nan
    if rating <= 4:
        return "Negative"
    elif rating <= 6:
        return "Neutral"
    else:
        return "Positive"


def save_plot(filename):
    """
    Lưu biểu đồ vào thư mục output.
    """
    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    print(f"Đã lưu biểu đồ: {path}")


def interpret_skewness(mean, median):
    """
    Nhận xét nhanh về độ lệch phân phối dựa trên Mean và Median.
    """
    if mean > median:
        return "Phân phối có xu hướng lệch phải vì Mean lớn hơn Median."
    elif mean < median:
        return "Phân phối có xu hướng lệch trái vì Mean nhỏ hơn Median."
    else:
        return "Phân phối khá cân bằng vì Mean gần bằng Median."


# ============================================================
# LOAD DỮ LIỆU
# ============================================================
# Nếu df đã tồn tại trong môi trường làm việc, script sẽ dùng df đó.
# Nếu chưa có df, script sẽ thử load từ file database.

try:
    df
    print("Đã phát hiện DataFrame df có sẵn. Sử dụng df hiện tại để phân tích.")
except NameError:
    print("Chưa có DataFrame df. Tiến hành load từ database...")
    df = build_dataframe_from_databases()


# ============================================================
# BƯỚC 1: KIỂM TRA NHANH CẤU TRÚC DỮ LIỆU SAU KHI LÀM SẠCH
# ============================================================

print("\n" + "=" * 80)
print("BƯỚC 1: KIỂM TRA NHANH CẤU TRÚC DỮ LIỆU")
print("=" * 80)

print(f"Số dòng: {df.shape[0]}")
print(f"Số cột: {df.shape[1]}")

print("\nTên các cột:")
print(df.columns.tolist())

print("\nKiểu dữ liệu của từng cột:")
print(df.dtypes)

print("\n5 dòng đầu tiên:")
print(df.head())

print("\nÝ nghĩa các cột quan trọng:")
print("- review / clean_text: nội dung nhận xét, bình luận hoặc review phim.")
print("- rating: điểm đánh giá phim nếu có, thường dùng để phân tích mức độ tích cực/tiêu cực.")
print("- sentiment: nhãn cảm xúc nếu có; nếu chưa có, có thể suy ra đơn giản từ rating.")
print("- movie_name: tên phim tương ứng với review.")
print("- source_platform: nguồn dữ liệu, ví dụ IMDb hoặc YouTube.")


# ============================================================
# BƯỚC 2: TẠO THÊM ĐẶC TRƯNG THỐNG KÊ TỪ REVIEW
# ============================================================

print("\n" + "=" * 80)
print("BƯỚC 2: TẠO THÊM ĐẶC TRƯNG TEXT")
print("=" * 80)

# Xác định cột chứa nội dung review.
if "clean_text" in df.columns:
    text_col = "clean_text"
elif "review" in df.columns:
    text_col = "review"
elif "review_text" in df.columns:
    text_col = "review_text"
elif "comment_text" in df.columns:
    text_col = "comment_text"
else:
    raise ValueError("Không tìm thấy cột chứa nội dung review/comment.")

# Đảm bảo text không bị NaN.
df[text_col] = df[text_col].fillna("").astype(str)

# Tạo đặc trưng độ dài review theo số ký tự.
df["review_length"] = df[text_col].str.len()

# Tạo đặc trưng số lượng từ.
df["word_count"] = df[text_col].apply(lambda x: len(tokenize_text(x)))

# Chuẩn hóa rating nếu có.
if "rating" in df.columns:
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_valid"] = df["rating"].where(df["rating"].between(1, 10))
else:
    df["rating_valid"] = np.nan

# Nếu chưa có sentiment, tạo sentiment từ rating.
if "sentiment" not in df.columns:
    df["sentiment"] = np.nan

if df["sentiment"].isna().all() and df["rating_valid"].notna().sum() > 0:
    df["sentiment"] = df["rating_valid"].apply(assign_sentiment_from_rating)

print("Đã tạo/cập nhật các cột:")
print("- review_length: độ dài review tính theo số ký tự.")
print("- word_count: số lượng từ trong review.")
print("- rating_valid: rating hợp lệ trong khoảng 1 đến 10.")
print("- sentiment: nhãn cảm xúc, nếu chưa có thì được suy ra từ rating.")

print("\nMột số dòng sau khi tạo đặc trưng:")
cols_to_show = [col for col in ["movie_name", "source_platform", "rating_valid", "sentiment", "review_length", "word_count"] if col in df.columns]
print(df[cols_to_show].head())


# ============================================================
# BƯỚC 3: TÍNH THỐNG KÊ MÔ TẢ
# ============================================================

print("\n" + "=" * 80)
print("BƯỚC 3: TÍNH THỐNG KÊ MÔ TẢ")
print("=" * 80)

numeric_vars = ["review_length", "word_count"]
if df["rating_valid"].notna().sum() > 0:
    numeric_vars.append("rating_valid")

stats_rows = []

for col in numeric_vars:
    series = df[col].dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    stats_rows.append({
        "Variable": col,
        "Count": series.count(),
        "Mean": series.mean(),
        "Median": series.median(),
        "Mode": get_mode(series),
        "Standard Deviation": series.std(),
        "Min": series.min(),
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Max": series.max()
    })

stats_df = pd.DataFrame(stats_rows)
print(stats_df.round(2))

stats_df.to_csv(OUTPUT_DIR / "descriptive_statistics.csv", index=False, encoding="utf-8-sig")

print("\nGiải thích ý nghĩa:")
print("- Mean: giá trị trung bình, dễ bị ảnh hưởng bởi outlier.")
print("- Median: trung vị, phản ánh giá trị ở giữa dữ liệu sau khi sắp xếp.")
print("- Mode: giá trị xuất hiện nhiều nhất.")
print("- Standard Deviation: độ lệch chuẩn, cho biết dữ liệu phân tán mạnh hay yếu.")
print("- Q1, Q3: mốc 25% và 75% của dữ liệu.")
print("- IQR = Q3 - Q1, dùng để đánh giá độ phân tán và phát hiện outlier.")


# ============================================================
# BƯỚC 4.1: HISTOGRAM PHÂN BỐ ĐỘ DÀI REVIEW
# ============================================================

print("\n" + "=" * 80)
print("BƯỚC 4.1: HISTOGRAM PHÂN BỐ ĐỘ DÀI REVIEW")
print("=" * 80)

sns.histplot(data=df, x="review_length", bins=50, kde=True)
plt.title("Histogram: Phân bố độ dài review")
plt.xlabel("Độ dài review, tính theo số ký tự")
plt.ylabel("Số lượng review")
save_plot("01_histogram_review_length.png")

review_mean = df["review_length"].mean()
review_median = df["review_length"].median()
review_q1 = df["review_length"].quantile(0.25)
review_q3 = df["review_length"].quantile(0.75)
review_iqr = review_q3 - review_q1
upper_bound = review_q3 + 1.5 * review_iqr
outlier_count = (df["review_length"] > upper_bound).sum()

print("Nhận xét:")
print(f"- Độ dài review trung bình: {review_mean:.2f} ký tự.")
print(f"- Trung vị độ dài review: {review_median:.2f} ký tự.")
print(f"- {interpret_skewness(review_mean, review_median)}")
print(f"- Số review có khả năng là outlier theo ngưỡng Q3 + 1.5*IQR: {outlier_count}.")


# ============================================================
# BƯỚC 4.2: HISTOGRAM PHÂN BỐ SỐ TỪ
# ============================================================

print("\n" + "=" * 80)
print("BƯỚC 4.2: HISTOGRAM PHÂN BỐ SỐ TỪ")
print("=" * 80)

sns.histplot(data=df, x="word_count", bins=50, kde=True)
plt.title("Histogram: Phân bố số từ trong review")
plt.xlabel("Số lượng từ")
plt.ylabel("Số lượng review")
save_plot("02_histogram_word_count.png")

word_mean = df["word_count"].mean()
word_median = df["word_count"].median()

print("Nhận xét:")
print(f"- Số từ trung bình: {word_mean:.2f}.")
print(f"- Trung vị số từ: {word_median:.2f}.")
print(f"- {interpret_skewness(word_mean, word_median)}")


# ============================================================
# BƯỚC 4.3: BAR CHART PHÂN BỐ SENTIMENT HOẶC RATING
# ============================================================

print("\n" + "=" * 80)
print("BƯỚC 4.3: BAR CHART PHÂN BỐ SENTIMENT / RATING")
print("=" * 80)

if df["sentiment"].notna().sum() > 0:
    sentiment_order = ["Negative", "Neutral", "Positive"]
    sentiment_counts = df["sentiment"].value_counts().reindex(sentiment_order)

    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title("Bar chart: Phân bố sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Số lượng review")
    save_plot("03_bar_sentiment_distribution.png")

    print("Nhận xét:")
    print(sentiment_counts)
    dominant_sentiment = sentiment_counts.idxmax()
    print(f"- Nhóm sentiment xuất hiện nhiều nhất là: {dominant_sentiment}.")
    print("- Biểu đồ này giúp kiểm tra dữ liệu có bị mất cân bằng nhãn hay không.")

elif df["rating_valid"].notna().sum() > 0:
    rating_counts = df["rating_valid"].value_counts().sort_index()

    sns.barplot(x=rating_counts.index.astype(str), y=rating_counts.values)
    plt.title("Bar chart: Phân bố rating")
    plt.xlabel("Rating")
    plt.ylabel("Số lượng review")
    save_plot("03_bar_rating_distribution.png")

    print("Nhận xét:")
    print(rating_counts)
    print("- Biểu đồ cho biết rating tập trung ở mức thấp, trung bình hay cao.")
else:
    print("Không có sentiment hoặc rating để vẽ bar chart.")


# ============================================================
# BƯỚC 4.4: BOXPLOT SO SÁNH ĐỘ DÀI REVIEW THEO SENTIMENT
# ============================================================

print("\n" + "=" * 80)
print("BƯỚC 4.4: BOXPLOT ĐỘ DÀI REVIEW THEO SENTIMENT")
print("=" * 80)

if df["sentiment"].notna().sum() > 0:
    box_df = df.dropna(subset=["sentiment", "review_length"]).copy()

    sns.boxplot(data=box_df, x="sentiment", y="review_length", order=["Negative", "Neutral", "Positive"], showfliers=False)
    plt.title("Boxplot: So sánh độ dài review theo sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Độ dài review, tính theo số ký tự")
    save_plot("04_boxplot_review_length_by_sentiment.png")

    group_stats = box_df.groupby("sentiment")["review_length"].agg(["count", "mean", "median", "std"]).round(2)
    print("Thống kê độ dài review theo sentiment:")
    print(group_stats)

    print("Nhận xét:")
    print("- Boxplot giúp so sánh median và độ phân tán độ dài review giữa các nhóm sentiment.")
    print("- Nếu hộp của nhóm nào cao hơn hoặc rộng hơn, nhóm đó có review dài hơn hoặc phân tán mạnh hơn.")
else:
    print("Không có sentiment để vẽ boxplot.")


# ============================================================
# BƯỚC 4.5: SCATTER PLOT RATING VÀ ĐỘ DÀI REVIEW
# ============================================================

print("\n" + "=" * 80)
print("BƯỚC 4.5: SCATTER PLOT RATING VÀ ĐỘ DÀI REVIEW")
print("=" * 80)

if df["rating_valid"].notna().sum() > 0:
    scatter_df = df.dropna(subset=["rating_valid", "review_length"]).copy()

    # Lấy mẫu nếu dữ liệu nhiều để biểu đồ dễ nhìn.
    if len(scatter_df) > 1500:
        plot_df = scatter_df.sample(1500, random_state=42)
    else:
        plot_df = scatter_df

    sns.scatterplot(data=plot_df, x="review_length", y="rating_valid", alpha=0.45)
    plt.title("Scatter plot: Mối quan hệ giữa rating và độ dài review")
    plt.xlabel("Độ dài review, tính theo số ký tự")
    plt.ylabel("Rating")
    save_plot("05_scatter_rating_vs_review_length.png")

    corr = scatter_df[["rating_valid", "review_length"]].corr().iloc[0, 1]

    print("Nhận xét:")
    print(f"- Hệ số tương quan giữa rating và độ dài review: {corr:.3f}.")
    if abs(corr) < 0.1:
        print("- Tương quan gần 0, chưa thấy quan hệ tuyến tính rõ ràng giữa rating và độ dài review.")
    elif corr > 0:
        print("- Tương quan dương, review dài hơn có xu hướng đi kèm rating cao hơn.")
    else:
        print("- Tương quan âm, review dài hơn có xu hướng đi kèm rating thấp hơn.")
else:
    print("Không có rating để vẽ scatter plot.")


# ============================================================
# BƯỚC 4.6: WORD CLOUD VÀ TOP TỪ PHỔ BIẾN
# ============================================================

print("\n" + "=" * 80)
print("BƯỚC 4.6: WORD CLOUD VÀ TOP TỪ PHỔ BIẾN")
print("=" * 80)

stopwords = {
    # English
    "the", "and", "for", "that", "this", "with", "was", "were", "are", "but",
    "not", "you", "your", "have", "has", "had", "his", "her", "they", "them",
    "from", "all", "one", "two", "can", "could", "would", "should", "about",
    "what", "when", "where", "who", "why", "how", "there", "their", "than",
    "then", "into", "out", "get", "got", "just", "really", "very", "more",
    "some", "which", "well", "even", "will", "most", "also", "much", "only",
    "because", "other", "it's", "its",
    # Movie-specific generic words
    "movie", "film", "movies", "films", "character", "characters", "story",
    "watch", "seen", "like", "time", "good", "great",
    # Vietnamese
    "và", "là", "của", "có", "cho", "một", "những", "các", "đã", "đang",
    "này", "kia", "thì", "mà", "với", "trong", "khi", "để", "tôi", "bạn",
    "nó", "phim", "xem", "rất", "hay", "quá", "thật", "nên", "cũng"
}

tokens = []

for text in df[text_col].dropna().astype(str):
    words = tokenize_text(text)
    words = [word for word in words if len(word) >= 3 and word not in stopwords]
    tokens.extend(words)

word_counts = Counter(tokens)

top_words = pd.DataFrame(
    word_counts.most_common(30),
    columns=["word", "count"]
)

print("Top 30 từ phổ biến:")
print(top_words)

top_words.to_csv(OUTPUT_DIR / "top_words.csv", index=False, encoding="utf-8-sig")

sns.barplot(data=top_words.head(20), y="word", x="count")
plt.title("Top 20 từ phổ biến nhất trong review")
plt.xlabel("Tần suất")
plt.ylabel("Từ")
save_plot("06_bar_top_words.png")

print("Nhận xét:")
print("- Các từ phổ biến cho biết người xem thường nhắc đến chủ đề hoặc yếu tố nào của phim.")
print("- Nếu các từ liên quan đến diễn xuất, cảnh phim, cảm xúc xuất hiện nhiều, dữ liệu có thể hữu ích cho sentiment analysis.")

try:
    from wordcloud import WordCloud

    wordcloud = WordCloud(
        width=1100,
        height=650,
        background_color="white",
        max_words=150
    ).generate_from_frequencies(word_counts)

    plt.figure(figsize=(11, 6.5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud: Từ phổ biến trong review")
    save_plot("07_wordcloud.png")

except ImportError:
    print("Chưa cài wordcloud. Cài bằng lệnh: pip install wordcloud")


# ============================================================
# BƯỚC 5: KẾT LUẬN TỔNG QUAN
# ============================================================

print("\n" + "=" * 80)
print("BƯỚC 5: KẾT LUẬN TỔNG QUAN")
print("=" * 80)

total_reviews = len(df)
rating_available = df["rating_valid"].notna().sum()
sentiment_available = df["sentiment"].notna().sum()

summary_lines = []

summary_lines.append(f"- Tổng số review/comment sau xử lý: {total_reviews}.")
summary_lines.append(f"- Số review có rating hợp lệ: {rating_available}.")
summary_lines.append(f"- Số review có sentiment: {sentiment_available}.")
summary_lines.append(f"- Độ dài review trung bình là {review_mean:.2f} ký tự, median là {review_median:.2f} ký tự.")
summary_lines.append(f"- Số từ trung bình là {word_mean:.2f}, median là {word_median:.2f}.")
summary_lines.append(f"- {interpret_skewness(review_mean, review_median)}")

if df["rating_valid"].notna().sum() > 0:
    rating_mean = df["rating_valid"].mean()
    rating_median = df["rating_valid"].median()
    rating_mode = get_mode(df["rating_valid"])
    summary_lines.append(f"- Rating trung bình là {rating_mean:.2f}, median là {rating_median:.2f}, mode là {rating_mode}.")
    summary_lines.append("- Nếu rating tập trung ở mức cao, dữ liệu có xu hướng nghiêng về review tích cực.")

if df["sentiment"].notna().sum() > 0:
    sentiment_counts = df["sentiment"].value_counts()
    summary_lines.append(f"- Nhóm sentiment phổ biến nhất là {sentiment_counts.idxmax()}.")

if df["rating_valid"].notna().sum() > 0:
    summary_lines.append("- Scatter plot và hệ số tương quan giúp đánh giá rating có liên quan đến độ dài review hay không.")
    summary_lines.append("- Nếu tương quan gần 0, độ dài review không nên được kỳ vọng là biến dự báo mạnh cho rating.")

summary_lines.append("- Kết quả thống kê giúp lựa chọn đặc trưng phù hợp cho bước sentiment analysis hoặc machine learning.")
summary_lines.append("- Các outlier về độ dài review nên được xem xét trước khi vector hóa văn bản hoặc huấn luyện mô hình.")

for line in summary_lines:
    print(line)

with open(OUTPUT_DIR / "summary_for_report.txt", "w", encoding="utf-8") as f:
    f.write("KẾT LUẬN TỔNG QUAN\n")
    f.write("\n".join(summary_lines))

print(f"\nĐã lưu toàn bộ biểu đồ, bảng thống kê và nhận xét trong thư mục: {OUTPUT_DIR.resolve()}")
