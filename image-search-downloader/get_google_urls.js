(async () => {
    console.log("Đang khởi động trình quét ảnh nâng cao...");

    const downloadAsFile = (content, fileName) => {
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fileName;
        a.click();
        URL.revokeObjectURL(url);
    };

    // Tìm tất cả các link trên trang có chứa tham số 'imgurl'
    const allLinks = Array.from(document.querySelectorAll('a[href*="imgurl"]'));
    const urls = [];

    allLinks.forEach(link => {
        try {
            const href = link.href;
            const urlParams = new URLSearchParams(href.substring(href.indexOf('?')));
            const imgUrl = urlParams.get('imgurl');
            
            if (imgUrl && !urls.includes(imgUrl)) {
                urls.push(decodeURIComponent(imgUrl));
            }
        } catch (e) {
            // Bỏ qua các link lỗi
        }
    });

    if (urls.length > 0) {
        console.log(`Thành công! Tìm thấy ${urls.length} link ảnh.`);
        console.log("Đang chuẩn bị tải file...");
        downloadAsFile(urls.join('\n'), 'google_images_final.txt');
    } else {
        console.error("Vẫn không tìm thấy link nào.");
        console.log("HƯỚNG DẪN KHẮC PHỤC:");
        console.log("1. Hãy cuộn chuột xuống để ảnh tải ra (Lazy load).");
        console.log("2. Thử nhấn vào 1-2 ảnh bất kỳ để Google kích hoạt dữ liệu.");
        console.log("3. Sau đó chạy lại code này.");
    }
})();