import time
import re
import os
import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException, InvalidSessionIdException
from webdriver_manager.chrome import ChromeDriverManager


# =========================
# CONFIG
# =========================
TOP_MOVIE_LIMIT = 250
MAX_REVIEWS_PER_MOVIE = 100

OUTPUT_FILE = "film_reviews_dataset_top250.csv"
MOVIE_LIST_FILE = "imdb_top250_movie_list.csv"
CHECKPOINT_FILE = "scrape_checkpoint.txt"

RESTART_BROWSER_EVERY = 10
SLEEP_BETWEEN_MOVIES = 5

# Chrome profile riêng để lưu đăng nhập
CHROME_PROFILE_DIR = "/Users/kietnguyen/Downloads/Movies scraping/chrome_profile"


# =========================
# DRIVER
# =========================
driver = None


def create_driver():
    options = webdriver.ChromeOptions()

    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")

    # Giữ đăng nhập sau khi restart Chrome
    options.add_argument(f"--user-data-dir={CHROME_PROFILE_DIR}")

    new_driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    new_driver.set_page_load_timeout(45)

    return new_driver


def restart_driver():
    global driver

    try:
        if driver:
            driver.quit()
    except:
        pass

    print("Restart Chrome...")
    driver = create_driver()
    time.sleep(2)


# =========================
# CHECKPOINT
# =========================
def save_checkpoint(index):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        f.write(str(index))


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return 0

    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except:
        return 0


# =========================
# TOP 250 MOVIES
# =========================
def get_top250_movie_pages(limit=250):
    global driver

    if os.path.exists(MOVIE_LIST_FILE):
        movie_df = pd.read_csv(MOVIE_LIST_FILE)

        movie_pages = dict(
            zip(movie_df["movie_name"], movie_df["review_url"])
        )

        print(f"Loaded movie list: {len(movie_pages)} phim")
        return movie_pages

    driver.get("https://www.imdb.com/chart/top/")
    time.sleep(5)

    movie_pages = {}

    links = driver.find_elements(
        By.CSS_SELECTOR,
        "a.ipc-title-link-wrapper"
    )

    for link in links:
        title = link.text.strip()
        href = link.get_attribute("href")

        if not title or not href:
            continue

        if "/title/" not in href:
            continue

        clean_title = re.sub(r"^\d+\.\s*", "", title)
        imdb_id = href.split("/title/")[1].split("/")[0]

        review_url = f"https://www.imdb.com/title/{imdb_id}/reviews/"

        movie_pages[clean_title] = review_url

        if len(movie_pages) >= limit:
            break

    movie_df = pd.DataFrame([
        {
            "movie_name": movie_name,
            "review_url": review_url
        }
        for movie_name, review_url in movie_pages.items()
    ])

    movie_df.to_csv(
        MOVIE_LIST_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Saved movie list: {len(movie_pages)} phim")

    return movie_pages


# =========================
# RATING
# =========================
def extract_rating(text):
    match = re.search(r"(\d{1,2})/10", str(text))

    if match:
        return int(match.group(1))

    return None


# =========================
# SAFE TEXT
# =========================
def safe_inner_text(element):
    global driver

    try:
        text = driver.execute_script(
            "return arguments[0].innerText;",
            element
        )
        return text.strip()

    except:
        return ""


# =========================
# CLICK LOAD MORE
# =========================
def click_load_more():
    global driver

    try:
        buttons = driver.find_elements(By.TAG_NAME, "button")

        for btn in buttons:
            try:
                btn_text = btn.text.lower()

                if "load more" in btn_text or "more" in btn_text:
                    driver.execute_script(
                        "arguments[0].scrollIntoView({block:'center'});",
                        btn
                    )

                    time.sleep(0.5)

                    driver.execute_script(
                        "arguments[0].click();",
                        btn
                    )

                    time.sleep(3)

                    return True

            except:
                pass

    except:
        pass

    return False


# =========================
# SCRAPE ONE MOVIE
# =========================
def scrape_reviews(movie_name, url):
    global driver

    driver.get(url)
    time.sleep(5)

    reviews = []
    seen_reviews = set()
    no_progress_rounds = 0

    while len(reviews) < MAX_REVIEWS_PER_MOVIE:
        elements = driver.find_elements(
            By.CSS_SELECTOR,
            '[data-testid="review-overflow"]'
        )

        previous_len = len(reviews)

        print(
            f"{movie_name}: elements={len(elements)}, collected={len(reviews)}"
        )

        for element in elements:
            review_text = safe_inner_text(element)

            if not review_text or len(review_text) < 20:
                continue

            if review_text in seen_reviews:
                continue

            seen_reviews.add(review_text)

            try:
                full_card_text = driver.execute_script(
                    """
                    let el = arguments[0];
                    let parent = el.closest('article, li, div');
                    return parent ? parent.innerText : el.innerText;
                    """,
                    element
                )
            except:
                full_card_text = ""

            rating = extract_rating(full_card_text)

            reviews.append({
                "movie_name": movie_name,
                "review_id": len(reviews) + 1,
                "rating": rating,
                "review_text": review_text,
                "review_length": len(review_text),
                "source_url": url
            })

            if len(reviews) >= MAX_REVIEWS_PER_MOVIE:
                break

        if len(reviews) >= MAX_REVIEWS_PER_MOVIE:
            break

        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);"
        )

        time.sleep(2)

        clicked = click_load_more()

        if len(reviews) == previous_len and not clicked:
            no_progress_rounds += 1
        else:
            no_progress_rounds = 0

        if no_progress_rounds >= 3:
            print(f"{movie_name}: không lấy thêm được review mới")
            break

    return reviews


# =========================
# SAVE CSV
# =========================
def append_to_csv(records):
    if not records:
        return

    df_temp = pd.DataFrame(records)

    file_exists = os.path.exists(OUTPUT_FILE)

    df_temp.to_csv(
        OUTPUT_FILE,
        mode="a",
        index=False,
        header=not file_exists,
        encoding="utf-8-sig"
    )


# =========================
# MAIN
# =========================
def main():
    global driver

    restart_driver()

    # Mở IMDb để bạn login lần đầu
    driver.get("https://www.imdb.com/")

    print("Nếu đây là lần đầu chạy, hãy đăng nhập IMDb trên Chrome vừa mở.")
    print("Nếu đã đăng nhập rồi thì chỉ cần nhấn ENTER.")
    input("Nhấn ENTER để bắt đầu scrape...")

    movie_pages = get_top250_movie_pages(
        limit=TOP_MOVIE_LIMIT
    )

    movie_items = list(movie_pages.items())

    start_index = load_checkpoint()

    print("=" * 60)
    print(f"Resume từ phim số: {start_index + 1}")
    print(f"Tổng phim: {len(movie_items)}")

    for idx in range(start_index, len(movie_items)):
        movie_name, url = movie_items[idx]

        print("=" * 60)
        print(f"[{idx + 1}/{len(movie_items)}] Đang cào phim: {movie_name}")

        # Restart định kỳ để tránh crash nhưng vẫn giữ login nhờ profile
        if idx > start_index and idx % RESTART_BROWSER_EVERY == 0:
            restart_driver()

        try:
            movie_reviews = scrape_reviews(movie_name, url)

            append_to_csv(movie_reviews)

            print(f"{movie_name}: lấy được {len(movie_reviews)} reviews")
            print(f"Đã lưu vào {OUTPUT_FILE}")

            # Chỉ lưu checkpoint sau khi phim đã xử lý xong
            save_checkpoint(idx + 1)

        except (InvalidSessionIdException, WebDriverException) as e:
            print(f"Chrome bị crash/session chết ở phim {movie_name}")
            print("Dừng tại đây. Chạy lại script để tiếp tục phim này.")

            # Không save checkpoint để không miss phim
            break

        except KeyboardInterrupt:
            print("Bạn đã dừng chương trình bằng Ctrl + C")
            print("Checkpoint giữ ở phim trước đó. Chạy lại sẽ tiếp tục.")
            break

        except Exception as e:
            print(f"Lỗi khi cào phim {movie_name}: {e}")
            print("Bỏ qua phim này và lưu checkpoint.")
            save_checkpoint(idx + 1)

        time.sleep(SLEEP_BETWEEN_MOVIES)

    print("=" * 60)
    print("HOÀN TẤT / ĐÃ DỪNG AN TOÀN")

    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        print("Tổng số phim:", df["movie_name"].nunique())
        print("Tổng số reviews:", len(df))
        print(df.head())

    try:
        driver.quit()
    except:
        pass


if __name__ == "__main__":
    main()