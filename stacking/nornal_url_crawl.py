import requests
import gzip
import json
import csv

r = requests.get(wat_url)
with open('wat.paths.gz', 'wb') as f:
    f.write(r.content)

# 2. 저장할 CSV 경로
csv_file_path = 'commoncrawl_urls_wat.csv'

# 3. URL 저장 리스트
url_list = []

# 4. WAT 파일 스트리밍
response = requests.get(wat_url, stream=True)
with gzip.open('temp.wat.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        try:
            record = json.loads(line)
            # 링크 정보 접근
            links = record.get('Envelope', {}) \
                          .get('Payload-Metadata', {}) \
                          .get('HTTP-Response-Metadata', {}) \
                          .get('HTML-Metadata', {}) \
                          .get('Links', [])
            for link in links:
                if 'url' in link:
                    url_list.append(link['url'])
                    if len(url_list) >= 400:
                        break  # 400개 모이면 멈춤
            if len(url_list) >= 400:
                break  # 바깥 반복도 멈춤
        except Exception:
            continue

# 5. CSV로 저장
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['url'])
    for url in url_list:
        writer.writerow([url])

print(f"총 {len(url_list)}개의 URL을 CSV로 저장했습니다: {csv_file_path}")
