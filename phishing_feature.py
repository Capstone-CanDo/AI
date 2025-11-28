import pandas as pd
import re
from urllib.parse import urlparse
import tldextract

# �ы뻾 愿��� �ㅼ썙�� 由ъ뒪��
TRAVEL_KEYWORDS = [
    # 기본 여행 키워드
    "travel", "trip", "tour", "flight", "airline", "airways", "airport",
    "plane", "boarding", "departure", "arrival", "gate", "baggage",
    "checkin", "check-in", "pilot",
    "booking", "hotel", "hostel", "resort", "bnb", "rooms", "stay",
    "vacation", "holiday", "package", "cruise",
    "visa", "passport", "itinerary", "reservation", "luggage", "ticket",
    "embassy", "consulate", "immigration", "customs",
    "tourist", "guide", "excursion",

    # 여행사/OTA/호텔 브랜드
    "booking", "booking.com", "agoda", "expedia", "airbnb", "tripadvisor",
    "hotels", "hotels.com", "trivago", "kayak", "priceline",

    # 항공사 브랜드
    "delta", "americanairlines", "aa.com", "united", "emirates",
    "qatarairways", "singaporeair", "lufthansa", "airfrance",
    "koreanair", "korean-air", "asiana", "jinair", "jejuair",

    # 기타 여행 브랜드
    "orbitz", "travelocity", "ctrip"
]


# �⑥텞 URL �꾨찓�� 由ъ뒪��
SHORT_DOMAINS = [
    "bit.ly", "t.co", "goo.gl", "tinyurl.com", "is.gd",
    "ow.ly", "buff.ly", "cutt.ly", "bit.do", "rebrand.ly",
    "t.ly", "shrtco.de", "s.id"
]

# URL �욎뿉 https:// �먮룞 遺�李�
def ensure_https(url):
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return "https://" + url.strip()

# �ы뻾 愿��� URL�몄� �먮퀎
def is_travel_related(url):
    lower = url.lower()
    return any(keyword in lower for keyword in TRAVEL_KEYWORDS)

# IP 二쇱냼 �먮퀎
def is_ip_address(host):
    return bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host))

# subdomain 媛쒖닔
def count_subdomains(ext):
    if ext.subdomain == "":
        return 0
    return len(ext.subdomain.split("."))

# �⑥텞 URL �먮퀎
def is_short_url(host):
    return int(any(short in host for short in SHORT_DOMAINS))

# URL �� lexical feature (24媛�)
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


# �꾩껜 �뚯씠�꾨씪��: �꾪꽣留� �� �섑뵆留� �� �꾩쿂由� �� ����
def sample_and_preprocess(
    input_csv, 
    sample_size, 
    sample_csv, 
    output_csv  
):
    #CSV �쎄린
    df = pd.read_csv(input_csv, encoding='cp949', usecols=["url"])   


    #�ы뻾 愿��� url �꾪꽣留�
    df_filtered = df[~df["url"].astype(str).apply(is_travel_related)]

    #1000媛� �쒕뜡 �섑뵆留�
    df_sample = df_filtered.sample(n=sample_size, random_state=42)

    #https:// �먮룞 遺�李�
    df_sample["url"] = df_sample["url"].astype(str).apply(ensure_https)
    df_sample.to_csv(sample_csv, index=False)

    #臾몄옄�닿린諛섑뵾泥섎줈 �꾩쿂由�
    feature_rows = []
    for url in df_sample["url"]:
        feature_rows.append(extract_lexical_features(url))

    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(output_csv, index=True)




if __name__ == "__main__":
    sample_and_preprocess(
        input_csv="test_phishing_url.csv",
        sample_size=400,
        sample_csv="test_phishing_sample.csv",
        output_csv="test_phishing_output.csv"
    )