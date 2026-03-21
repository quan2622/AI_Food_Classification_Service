import os
import time
import random
import pandas as pd
import requests
import matplotlib.pyplot as plt
from ddgs import DDGS

def get_ddg_img_urls(query, max_results=50):
    urls = []
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries and len(urls) == 0:
        try:
            # Thêm ngẫu nhiên delay 2-4 giây giữa các request
            time.sleep(random.uniform(2, 4))
            
            with DDGS() as ddgs:
                results = ddgs.images(
                    query,
                    region="wt-wt",
                    safesearch="off",
                    max_results=max_results
                )
                for r in results:
                    if 'image' in r:
                        urls.append(r['image'])
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 5 + (retry_count * 5)  # 5s, 10s, 15s
                print(f"    Lỗi khi tìm kiếm '{query}': {e}")
                print(f"    Thử lại sau {wait_time} giây...")
                time.sleep(wait_time)
            else:
                print(f"    Lỗi khi tìm kiếm '{query}' sau {max_retries} lần thử: {e}")
    
    return urls

def download_images(query, urls, output_dir):
    """Download ảnh từ danh sách URL và lưu vào folder"""
    
    # Tạo folder cho food dựa vào tên query
    safe_query = query.replace(' ', '_').replace('/', '-').replace('\\', '-')
    image_output_dir = os.path.join(output_dir, "Google", safe_query)
    os.makedirs(image_output_dir, exist_ok=True)
    
    start_index = 1
    downloaded_count = 0
    
    for i, url in enumerate(urls):
        try:
            # Thêm delay 0.5-1.5 giây giữa các lần download
            time.sleep(random.uniform(0.5, 1.5))
            
            image_name = f'{safe_query}_{start_index}.jpg'
            print(f'  [GET] Downloading {image_name} - {url}')
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            result = requests.get(url, timeout=10, headers=headers)
            
            if result.status_code == 200:
                image_path = os.path.join(image_output_dir, image_name)
                with open(image_path, 'wb') as f:
                    f.write(result.content)
                
                # Kiểm tra xem file có hợp lệ không
                try:
                    plt.imread(image_path)
                    start_index += 1
                    downloaded_count += 1
                except:
                    print(f'  [DELETE] Image has no contents - {image_name}')
                    os.remove(image_path)
            else:
                print(f'  [ERROR] HTTP {result.status_code} - {url}')
        except Exception as e:
            print(f'  [ERROR] Failed to download - {url}: {str(e)[:50]}')
    
    return downloaded_count

# --- CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    input_file = "food_list.xlsx"
    output_dir = "images/urls"
    
    
    os.makedirs(output_dir, exist_ok=True)

    
    try:
        df = pd.read_excel(input_file, header=None)
        queries = df.iloc[:, 0].dropna().tolist()
    except Exception as e:
        print(f"Không thể đọc file Excel: {e}")
        exit()

    print(f"Bắt đầu quét {len(queries)} món ăn từ DuckDuckGo...")

    for i, query in enumerate(queries):
        
        safe_query = query.replace(' ', '_').replace('/', '-').replace('\\', '-')
        filename = os.path.join(output_dir, f"urls_{safe_query}.txt")

        
        if os.path.exists(filename):
            print(f"Đã có dữ liệu cho [{query}], bỏ qua.")
            continue

        print(f"[{i+1}/{len(queries)}] Đang tìm ảnh cho: {query}...")
        
        # Lấy link ảnh (mặc định lấy 50 ảnh mỗi món)
        image_links = get_ddg_img_urls(query, max_results=50)

        if image_links:
            # Lưu danh sách URL
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(image_links))
            print(f"Đã lưu {len(image_links)} link vào: {filename}")
            
            # Download ảnh về máy
            print(f"Bắt đầu download ảnh cho: {query}...")
            downloaded_count = download_images(query, image_links, "images")
            print(f"✓ Đã download {downloaded_count} ảnh cho: {query}")
        else:
            print(f"Không tìm thấy ảnh nào cho món: {query}")

        # Nghỉ 5-10 giây giữa các query để tránh bị rate limit
        time.sleep(random.uniform(5, 10))

    print("\nHoàn thành! Toàn bộ link ảnh đã nằm trong thư mục images/urls.")