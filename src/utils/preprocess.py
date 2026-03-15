import pandas as pd
import re
from urllib.parse import urlparse
import tldextract

# 여행 관련 키워드 리스트
TRAVEL_KEYWORDS = [
    "travel", "trip", "tour", "flight", "airline", "airport", "booking",  "hotel", 
    "stay",  "visa", "passport", "luggage", "ticket", "itinerary", "reservation", 
]

# 단축 URL 도메인 리스트
SHORT_DOMAINS = [
    "bit.ly", "t.co", "goo.gl", "tinyurl.com", "is.gd",
    "ow.ly", "buff.ly", "cutt.ly", "bit.do", "rebrand.ly",
    "t.ly", "shrtco.de", "s.id"
]

# URL 앞에 https:// 자동 부착
def ensure_https(url):
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return "https://" + url.strip()

# 여행 관련 URL인지 판별
def is_travel_related(url):
    lower = url.lower()
    return any(keyword in lower for keyword in TRAVEL_KEYWORDS)

# IP 주소 판별
def is_ip_address(host):
    return bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host))

# subdomain 개수
def count_subdomains(ext):
    if ext.subdomain == "":
        return 0
    return len(ext.subdomain.split("."))

# 단축 URL 판별
def is_short_url(host):
    return int(any(short in host for short in SHORT_DOMAINS))

# URL → lexical feature (24개)
def extract_lexical_features(url):
    parsed = urlparse(url)
    ext = tldextract.extract(url)

    host = parsed.netloc or ""
    path = parsed.path or ""
    tld = ext.suffix or ""
    lower = url.lower()

    feats = {}
    feats["url"] = url

    # Length-based (6)
    feats["len_url"] = len(url)
    feats["len_hostname"] = len(host)
    feats["len_TLD"] = len(tld)
    feats["len_path"] = len(path)
    feats["url_depth"] = path.count("/")
    feats["len_first_dir"] = len(path.split("/")[1]) if path.count("/") >= 1 else 0

    # Count-based (16)
    feats["num_http"] = lower.count("http")
    feats["num_https"] = lower.count("https")
    feats["num_www"] = lower.count("www")
    feats["num_@"] = lower.count("@")
    feats["num_?"] = lower.count("?")
    feats["num_&"] = lower.count("&")
    feats["num_%"] = lower.count("%")
    feats["num_#"] = lower.count("#")
    feats["num_."] = lower.count(".")
    feats["num_="] = lower.count("=")
    feats["num__"] = lower.count("_")
    feats["num_-"] = lower.count("-")
    feats["num_hostname_-"] = host.count("-")
    feats["num_subdomains"] = count_subdomains(ext)
    feats["num_digits"] = sum(c.isdigit() for c in lower)
    feats["num_letters"] = sum(c.isalpha() for c in lower)

    # Existence-based (2)
    feats["is_ip"] = int(is_ip_address(host))
    feats["is_short_url"] = is_short_url(host)

    return feats


# 전체 파이프라인: 필터링 → 샘플링 → 전처리 → 저장
def sample_and_preprocess(
    input_csv, 
    sample_size, 
    sample_csv, 
    output_csv  
):
    #CSV 읽기
    df = pd.read_csv(input_csv)

    #여행 관련 url 필터링
    df_filtered = df[~df["url"].astype(str).apply(is_travel_related)]

    #1000개 랜덤 샘플링
    df_sample = df_filtered.sample(n=sample_size, random_state=42)

    #https:// 자동 부착
    df_sample["url"] = df_sample["url"].astype(str).apply(ensure_https)
    df_sample.to_csv(sample_csv, index=False)

    #문자열기반피처로 전처리
    feature_rows = []
    for url in df_sample["url"]:
        feature_rows.append(extract_lexical_features(url))

    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(output_csv, index=True)



if __name__ == "__main__":
    sample_and_preprocess(
        input_csv="top-1m.csv",
        sample_size=1000,
        sample_csv="normal_1000.csv",
        output_csv="normal_1000_preprocessed.csv"
    )
