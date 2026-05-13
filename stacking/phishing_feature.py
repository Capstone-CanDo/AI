import pandas as pd
import re
from urllib.parse import urlparse
import tldextract

# ===============================
# 설정
# ===============================
DATE_COLUMNS = ["timestamp", "date", "datetime", "created_at"]
TEST_SIZE_PER_CLASS = 125

# ===============================
# 여행 관련 키워드
# ===============================
TRAVEL_KEYWORDS = [
    "travel", "trip", "tour", "flight", "airline", "airways", "airport",
    "plane", "boarding", "departure", "arrival", "gate", "baggage",
    "checkin", "check-in", "pilot",
    "booking", "hotel", "hostel", "resort", "bnb", "rooms", "stay",
    "vacation", "holiday", "package", "cruise",
    "visa", "passport", "itinerary", "reservation", "luggage", "ticket",
    "embassy", "consulate", "immigration", "customs",
    "tourist", "guide", "excursion",
    "booking.com", "agoda", "expedia", "airbnb", "tripadvisor",
    "hotels.com", "trivago", "kayak", "priceline",
    "delta", "americanairlines", "united", "emirates",
    "qatarairways", "singaporeair", "lufthansa", "airfrance",
    "koreanair", "asiana", "jinair", "jejuair",
    "orbitz", "travelocity", "ctrip", "cafe", "shop", "resturant"
]

# 단축 URL 도메인 리스트
SHORT_DOMAINS = [
    "bit.ly", "t.co", "goo.gl", "tinyurl.com", "is.gd",
    "ow.ly", "buff.ly", "cutt.ly", "bit.do", "rebrand.ly",
    "t.ly", "shrtco.de", "s.id"
]

# ===============================
# 유틸리티 함수
# ===============================
def ensure_https(url):
    if isinstance(url, str) and not url.startswith(("http://", "https://")):
        return "https://" + url.strip()
    return url.strip()


def is_travel_related(url):
    lower = url.lower()
    return any(keyword in lower for keyword in TRAVEL_KEYWORDS)


def is_ip_address(host):
    return bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host))


def count_subdomains(ext):
    if ext.subdomain == "":
        return 0
    return len(ext.subdomain.split("."))


def is_short_url(host):
    return int(any(short in host for short in SHORT_DOMAINS))


# ===============================
# Lexical Feature 추출
# ===============================
def extract_lexical_features(url):
    parsed = urlparse(url)
    ext = tldextract.extract(url)

    host = parsed.netloc or ""
    path = parsed.path or ""
    tld = ext.suffix or ""
    lower = url.lower()

    feats = {
        "url": url,
        "len_url": len(url),
        "len_hostname": len(host),
        "len_TLD": len(tld),
        "len_path": len(path),
        "url_depth": path.count("/"),
        "len_first_dir": len(path.split("/")[1]) if path.count("/") >= 1 else 0,
        "num_http": lower.count("http"),
        "num_https": lower.count("https"),
        "num_www": lower.count("www"),
        "num_@": lower.count("@"),
        "num_?": lower.count("?"),
        "num_&": lower.count("&"),
        "num_%": lower.count("%"),
        "num_#": lower.count("#"),
        "num_.": lower.count("."),
        "num_=": lower.count("="),
        "num__": lower.count("_"),
        "num_-": lower.count("-"),
        "num_hostname_-": host.count("-"),
        "num_subdomains": count_subdomains(ext),
        "num_digits": sum(c.isdigit() for c in lower),
        "num_letters": sum(c.isalpha() for c in lower),
        "is_ip": int(is_ip_address(host)),
        "is_short_url": is_short_url(host),
    }

    return feats


# ===============================
# 최신 데이터 정렬
# ===============================
def sort_by_latest(df):
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.sort_values(by=col, ascending=False)
            print(f"'{col}' 기준으로 최신 순 정렬 완료")
            return df.reset_index(drop=True)

    print("날짜 컬럼이 없어 파일의 상단을 최신 데이터로 간주합니다.")
    return df.reset_index(drop=True)


# ===============================
# 메인 함수
# ===============================
def sample_and_preprocess(
    input_csv,
    sample_size,
    travel_sample_csv,
    non_travel_sample_csv,
    travel_output_csv,
    non_travel_output_csv
):
    # CSV 읽기
    df = pd.read_csv(input_csv, encoding="utf-8")

    if "url" not in df.columns:
        raise ValueError("CSV 파일에 'url' 컬럼이 필요합니다.")

    # 최신 순 정렬
    df = sort_by_latest(df)

    # HTTPS 추가
    df["url"] = df["url"].astype(str).apply(ensure_https)

    # 여행 관련 여부 라벨 추가
    df["is_travel"] = df["url"].apply(is_travel_related)

    # 데이터 분리
    df_travel = df[df["is_travel"]].copy()
    df_non_travel = df[~df["is_travel"]].copy()

    # ==========================================
    # 1. 테스트용 균형 데이터셋 생성 (125 + 125)
    # ==========================================
    balanced_travel = df_travel.head(min(TEST_SIZE_PER_CLASS, len(df_travel)))
    balanced_non_travel = df_non_travel.head(min(TEST_SIZE_PER_CLASS, len(df_non_travel)))

    balanced_df = pd.concat(
        [balanced_travel, balanced_non_travel],
        ignore_index=True
    )

    balanced_df.to_csv(
        "phishing_250_urls.csv",
        index=False,
        encoding="utf-8"
    )

    # 피처 추출
    balanced_features = [
        extract_lexical_features(url)
        for url in balanced_df["url"]
    ]

    balanced_features_df = pd.DataFrame(balanced_features)
    balanced_features_df["is_travel"] = balanced_df["is_travel"].values

    balanced_features_df.to_csv(
        "phishing_250_features.csv",
        index=False,
        encoding="utf-8"
    )


    print(f"균형 테스트 데이터 저장 완료: {len(balanced_df)}개")

    # ==========================================
    # 2. 테스트 데이터 제거
    # ==========================================
    test_urls = set(balanced_df["url"])
    df_travel = df_travel[~df_travel["url"].isin(test_urls)]
    df_non_travel = df_non_travel[~df_non_travel["url"].isin(test_urls)]

    # ==========================================
    # 3. 학습 데이터 생성 (최신 순)
    # ==========================================
    df_travel_sample = df_travel.head(min(sample_size, len(df_travel)))
    df_non_travel_sample = df_non_travel.head(min(sample_size, len(df_non_travel)))

    df_travel_sample.to_csv(travel_sample_csv, index=False)
    df_non_travel_sample.to_csv(non_travel_sample_csv, index=False)

    # ==========================================
    # 4. 피처 추출
    # ==========================================
    travel_features = [
        extract_lexical_features(url)
        for url in df_travel_sample["url"]
    ]

    non_travel_features = [
        extract_lexical_features(url)
        for url in df_non_travel_sample["url"]
    ]

    travel_feature_df = pd.DataFrame(travel_features)
    non_travel_feature_df = pd.DataFrame(non_travel_features)

    travel_feature_df["is_travel"] = 1
    non_travel_feature_df["is_travel"] = 0

    travel_feature_df.to_csv(travel_output_csv, index=False)
    non_travel_feature_df.to_csv(non_travel_output_csv, index=False)

    print(f"여행 관련 URL 수: {len(travel_feature_df)}")
    print(f"비여행 관련 URL 수: {len(non_travel_feature_df)}")

    # ==========================================
    # 추가: 테스트셋 + 학습셋 제외 정상 URL 250개 저장
    # ==========================================

    # 학습셋에 사용된 URL 제거
    used_non_travel_urls = set(df_non_travel_sample["url"])

    remaining_non_travel = df_non_travel[
        ~df_non_travel["url"].isin(used_non_travel_urls)
    ]

    # 250개 선택
    normal_250 = remaining_non_travel.head(250)

    # URL CSV 저장
    normal_250.to_csv(
        "remaining_normal_250_urls.csv",
        index=False,
        encoding="utf-8"
    )

    # Feature 추출
    normal_250_features = [
        extract_lexical_features(url)
        for url in normal_250["url"]
    ]

    normal_250_feature_df = pd.DataFrame(normal_250_features)
    normal_250_feature_df["is_travel"] = 0

    # Feature CSV 저장
    normal_250_feature_df.to_csv(
        "remaining_normal_250_features.csv",
        index=False,
        encoding="utf-8"
    )

    print(f"남은 정상 URL 250개 저장 완료")


# ===============================
# 실행
# ===============================
if __name__ == "__main__":
    sample_and_preprocess(
        input_csv="verified_online.csv",
        sample_size=8000,
        travel_sample_csv="travel_phishing_sample.csv",
        non_travel_sample_csv="non_travel_phishing_sample.csv",
        travel_output_csv="travel_phishing_output.csv",
        non_travel_output_csv="non_travel_phishing_output.csv"
    )