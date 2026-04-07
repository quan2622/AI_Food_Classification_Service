# Cơ sở lý thuyết cho hệ thống phân loại ảnh món ăn Việt Nam

## 1. Bối cảnh bài toán và mục tiêu nhận diện ảnh món ăn

Trong bài toán phân loại ảnh món ăn, mục tiêu của hệ thống là ánh xạ một ảnh đầu vào vào một trong các lớp món ăn đã biết trong tập dữ liệu huấn luyện. Về mặt toán học, với ảnh đầu vào \(x\), mô hình học hàm \(f_\theta(x)\) để dự đoán nhãn \(y \in \{1,2,\dots,C\}\), trong đó \(C\) là số lớp món ăn. Với dự án này, tập lớp được suy ra tự động từ cấu trúc thư mục dữ liệu huấn luyện, do đó số lớp có thể thay đổi theo dữ liệu thực tế. Cách thiết kế này phù hợp với bối cảnh triển khai luận văn vì giảm thao tác cấu hình thủ công và tăng tính tái sử dụng của pipeline.

Không giống các bài toán ảnh vật thể đơn giản, ảnh món ăn có độ biến thiên lớn theo ánh sáng, góc chụp, bố cục đĩa, nền bàn, dụng cụ ăn uống, cũng như phong cách trình bày theo vùng miền. Đồng thời, nhiều món ăn có đặc trưng thị giác gần nhau nên ranh giới giữa các lớp không luôn rõ ràng. Vì vậy, một hệ thống nhận diện món ăn hiệu quả cần kết hợp xử lý ảnh số, học biểu diễn bằng học sâu và cơ chế đánh giá toàn diện để đảm bảo khả năng tổng quát hóa khi gặp dữ liệu thực tế.

## 2. Cơ sở xử lý ảnh số trong hệ thống

Xử lý ảnh số (Digital Image Processing) cung cấp các phép biến đổi đầu vào nhằm chuẩn hóa dữ liệu trước khi đưa vào mô hình học sâu. Ảnh màu RGB được biểu diễn dưới dạng tensor ba kênh, trong đó mỗi điểm ảnh mang thông tin màu sắc và cường độ. Nếu không chuẩn hóa, sự khác biệt về điều kiện chụp có thể khiến mô hình học lệch theo nhiễu thay vì học đặc trưng bản chất của món ăn.

Trong pipeline của dự án, bước tiền xử lý cơ bản gồm thay đổi kích thước ảnh theo từng kiến trúc mạng, chuyển ảnh sang tensor, và chuẩn hóa theo thống kê chuẩn ImageNet với trung bình \([0.485, 0.456, 0.406]\) và độ lệch chuẩn \([0.229, 0.224, 0.225]\). Việc dùng cùng chuẩn hóa cho cả huấn luyện và suy luận là yêu cầu quan trọng để giữ tính nhất quán phân phối dữ liệu giữa hai giai đoạn. Nếu phân phối đầu vào khi suy luận khác phân phối huấn luyện, độ chính xác có thể suy giảm đáng kể dù kiến trúc mạng không thay đổi.

Một phép toán cốt lõi xuyên suốt từ xử lý ảnh cổ điển đến mạng CNN là tích chập rời rạc:
\[
S(i,j)=\sum_m\sum_n I(i-m,j-n)K(m,n),
\]
trong đó \(I\) là ảnh đầu vào và \(K\) là kernel. Trong xử lý ảnh truyền thống, kernel được thiết kế thủ công để làm trơn, tăng biên hoặc lọc nhiễu; trong học sâu, kernel được học tự động từ dữ liệu để tối ưu trực tiếp cho mục tiêu phân loại.

## 3. Thị giác máy tính và hướng tiếp cận phân loại ảnh

Thị giác máy tính (Computer Vision) nghiên cứu các phương pháp giúp máy hiểu ngữ nghĩa từ dữ liệu thị giác. Trong dự án này, bài toán được mô hình hóa dưới dạng phân loại ảnh một nhãn (single-label image classification), nghĩa là mỗi ảnh được gán một nhãn món ăn chính. Đây là lựa chọn phù hợp với tập dữ liệu tổ chức theo cấu trúc thư mục lớp, đồng thời tối ưu cho kịch bản API dự đoán nhanh theo yêu cầu người dùng.

Ở mức hệ thống, bài toán không chỉ dừng ở mô hình học sâu. Dự án triển khai đầy đủ chu trình từ nạp dữ liệu, huấn luyện, đánh giá, suy luận đến phục vụ qua FastAPI. Dự đoán trả về top-3 lớp có xác suất cao nhất thay vì chỉ top-1, giúp tăng tính hỗ trợ quyết định trong thực tế khi các lớp gần nhau dễ gây nhầm lẫn. Cách thiết kế này phản ánh tư duy thị giác máy tính ứng dụng: đầu ra mô hình cần có ý nghĩa sử dụng, không chỉ là một chỉ số tối ưu trong thí nghiệm.

## 4. Học máy, học sâu và nguyên lý tối ưu hóa mô hình

Học máy (Machine Learning) cho phép mô hình trích xuất quy luật từ dữ liệu thay vì dùng luật viết tay. Với ảnh, cách tiếp cận truyền thống yêu cầu thiết kế đặc trưng thủ công nên khó bao quát sự biến thiên phức tạp của món ăn. Học sâu (Deep Learning) khắc phục điểm này thông qua học đặc trưng phân cấp trực tiếp từ ảnh đầu vào.

Trong mô hình phân loại nhiều lớp, tầng cuối thường tạo vector logit \(z\), sau đó dùng hàm Softmax để chuyển sang phân phối xác suất:
\[
P(y=k|x)=\frac{e^{z_k}}{\sum_{c=1}^{C}e^{z_c}}.
\]
Mất mát được tối ưu là cross-entropy:
\[
\mathcal{L}=-\sum_{k=1}^{C} y_k\log P(y=k|x).
\]
Trong dự án, hàm mất mát được mở rộng bằng kỹ thuật label smoothing (\(\epsilon = 0.1\)) nhằm giảm hiện tượng mô hình quá tự tin vào nhãn cứng, từ đó cải thiện khả năng tổng quát trên dữ liệu mới. Đây là lựa chọn phù hợp cho bài toán món ăn vốn có ranh giới thị giác mờ giữa một số lớp.

Bộ tối ưu sử dụng AdamW, tức tách suy giảm trọng số (weight decay) khỏi cập nhật gradient thích nghi, thường cho tính ổn định tốt trong fine-tuning mạng tiền huấn luyện. Learning rate được điều chỉnh theo Cosine Annealing trong suốt quá trình huấn luyện 30 epoch, giúp mô hình giảm tốc học dần về cuối, hạn chế dao động quanh cực tiểu và cải thiện chất lượng hội tụ.

## 5. Mạng nơ-ron tích chập và các kiến trúc được dùng trong dự án

Mạng nơ-ron tích chập (CNN) là nền tảng chính cho nhận diện ảnh. Đặc trưng quan trọng của CNN gồm kết nối cục bộ, chia sẻ trọng số và khả năng học đặc trưng không gian nhiều tầng. Những tầng đầu học biên và kết cấu, tầng giữa học motif thị giác phức tạp hơn, còn tầng sâu học đặc trưng ngữ nghĩa cấp cao liên quan trực tiếp tới lớp món ăn.

Dự án triển khai ba kiến trúc tiêu biểu. Thứ nhất, EfficientNet-B3 được chọn làm cấu hình mặc định vì cân bằng tốt giữa số tham số và độ chính xác nhờ cơ chế compound scaling theo chiều sâu, chiều rộng và độ phân giải. Trong quá trình fine-tuning, một phần các khối đặc trưng đầu được đóng băng để giữ tri thức nền, trong khi bộ phân loại cuối được thay bằng head mới gồm Dropout và các lớp tuyến tính phù hợp số lớp món ăn thực tế.

Thứ hai, ResNet50 sử dụng residual connection để duy trì luồng gradient ở mạng sâu. Chiến lược huấn luyện của dự án đóng băng các tầng sớm, chỉ mở các tầng sâu hơn và fully connected head, từ đó giảm chi phí tính toán và hạn chế quá khớp khi dữ liệu không quá lớn.

Thứ ba, InceptionV3 khai thác nhiều nhánh tích chập trong cùng một block để học đặc trưng đa tỉ lệ. Mô hình này dùng thêm nhánh phụ (auxiliary logits) trong huấn luyện; hàm mất mát tổng hợp được viết dưới dạng:
\[
\mathcal{L}_{total}=\mathcal{L}_{main}+0.4\mathcal{L}_{aux}.
\]
Thiết kế này hỗ trợ lan truyền gradient hiệu quả hơn trong huấn luyện mạng sâu.

Điểm chung của cả ba kiến trúc là sử dụng pretrained weights cục bộ và thay classifier theo số lớp hiện tại. Đây là hình thức transfer learning điển hình, phù hợp trong luận văn ứng dụng khi cần tận dụng tri thức từ tập dữ liệu lớn để tăng hiệu quả học trên dữ liệu chuyên biệt món ăn Việt Nam.

## 6. Tiền xử lý và tăng cường dữ liệu trong huấn luyện

Trong giai đoạn huấn luyện, pipeline tăng cường dữ liệu gồm Resize mở rộng, RandomCrop, RandomHorizontalFlip, RandomRotation 15 độ và ColorJitter theo ba thành phần sáng, tương phản, bão hòa. Về bản chất, các phép biến đổi này mô phỏng biến động tự nhiên khi người dùng chụp ảnh món ăn ở điều kiện thực tế khác nhau. Nhờ đó, mô hình học các đặc trưng ổn định hơn thay vì ghi nhớ vị trí hoặc bố cục cố định trong dữ liệu gốc.

Với tập validation và test, dự án không dùng augmentation ngẫu nhiên mà chỉ resize và chuẩn hóa. Quy tắc này rất quan trọng trong đánh giá học thuật: dữ liệu kiểm định phải phản ánh đúng khả năng tổng quát của mô hình, tránh việc nhiễu ngẫu nhiên làm sai lệch so sánh giữa các lần huấn luyện.

Ngoài ra, dữ liệu được tổ chức rõ ràng theo ba tập Train/Validate/Test, và hệ thống có cơ chế thống kê số mẫu từng lớp ở mỗi tập. Thông tin phân bố lớp là cơ sở để nhận diện nguy cơ mất cân bằng dữ liệu và giải thích hiện tượng mô hình thiên lệch về các lớp xuất hiện nhiều.

## 7. Quy trình xây dựng hệ thống từ huấn luyện đến triển khai dịch vụ

Về kỹ thuật hệ thống, quy trình được thiết kế theo hướng end-to-end. Ở pha huấn luyện, hệ thống chọn kiến trúc theo cấu hình, tải dữ liệu theo kích thước ảnh tương ứng, huấn luyện nhiều epoch và lưu checkpoint tốt nhất theo tiêu chí validation accuracy. Đồng thời, lịch sử huấn luyện (loss, accuracy, learning rate, thời gian mỗi epoch) được ghi lại để phục vụ phân tích thực nghiệm.

Ở pha đánh giá, mô hình tốt nhất được nạp lại trên tập test để sinh classification report và confusion matrix. Đây là bước chuyển từ theo dõi tiến trình tối ưu sang đo lường hiệu năng độc lập, giúp kết luận khách quan hơn cho phần kết quả luận văn.

Ở pha triển khai, FastAPI cung cấp các endpoint chính gồm dự đoán ảnh, tra cứu lớp, kiểm tra sức khỏe dịch vụ và ghi nhận phản hồi người dùng. Các mô hình được cache trong bộ nhớ, có cơ chế warm-up bằng tensor giả để giảm độ trễ yêu cầu đầu tiên. Việc dùng khóa đồng bộ theo mô hình giúp suy luận an toàn hơn trong môi trường phục vụ đồng thời nhiều yêu cầu.

Một đóng góp thực tiễn quan trọng là vòng phản hồi dữ liệu: sau khi dự đoán, người dùng có thể xác nhận hoặc chỉnh nhãn; ảnh được sao chép vào thư mục reviewed theo nhãn xác nhận và metadata được lưu trong PostgreSQL. Cơ chế này tạo nền tảng cho chiến lược cải tiến liên tục dữ liệu huấn luyện theo chu trình human-in-the-loop.

## 8. Đánh giá mô hình: ý nghĩa chỉ số và phân tích lỗi

Đánh giá trong bài toán phân loại cần đa chỉ số để tránh kết luận phiến diện. Accuracy phản ánh tỷ lệ dự đoán đúng tổng thể:
\[
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}.
\]
Tuy nhiên, khi phân bố lớp không đều, accuracy cao chưa chắc đồng nghĩa mô hình tốt trên mọi lớp.

Do đó cần xem đồng thời precision, recall và F1-score. Precision đo mức chính xác của các dự đoán dương tính:
\[
\text{Precision}=\frac{TP}{TP+FP},
\]
trong khi recall đo khả năng thu hồi mẫu thực sự thuộc lớp:
\[
\text{Recall}=\frac{TP}{TP+FN}.
\]
F1-score là trung bình điều hòa:
\[
F1=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}.
\]
Trong hệ thống này, classification report từ thư viện đánh giá cung cấp các chỉ số theo từng lớp món ăn, giúp xác định cụ thể lớp nào đang yếu thay vì chỉ nhìn hiệu năng trung bình.

Confusion matrix là công cụ phân tích lỗi đặc biệt hữu ích cho bài toán món ăn vì nó chỉ ra cặp lớp hay bị nhầm lẫn. Từ ma trận nhầm lẫn, có thể suy luận nguyên nhân thị giác như màu sắc tương đồng, hình dáng gần nhau hoặc dữ liệu chưa đủ đa dạng cho một lớp nhất định. Đây là cơ sở khoa học để đề xuất cải tiến tiếp theo như tăng cường dữ liệu có mục tiêu, bổ sung mẫu khó, hoặc điều chỉnh chiến lược fine-tuning.

## 9. Tổng kết cơ sở lý thuyết gắn với dự án

Từ góc nhìn lý thuyết, dự án là sự kết hợp đồng bộ giữa xử lý ảnh số, học sâu dựa trên CNN và kỹ thuật triển khai dịch vụ AI. Về mô hình, dự án áp dụng transfer learning với ba backbone mạnh gồm EfficientNet-B3, ResNet50 và InceptionV3, đi kèm các kỹ thuật tối ưu hiện đại như label smoothing, AdamW và cosine annealing. Về dữ liệu, pipeline augmentation được thiết kế đúng bản chất biến thiên của ảnh món ăn, đồng thời phân tách Train/Validate/Test rõ ràng để đánh giá khách quan. Về hệ thống, kiến trúc API cùng cơ chế lưu phản hồi vào cơ sở dữ liệu tạo thành vòng lặp cải tiến liên tục, phù hợp định hướng ứng dụng thực tế của một luận văn kỹ thuật.

Như vậy, cơ sở lý thuyết của bài toán không chỉ nằm ở thuật toán phân loại ảnh, mà còn ở cách liên kết chặt chẽ giữa mô hình học sâu, quy trình dữ liệu và hạ tầng triển khai. Chính sự liên kết này quyết định chất lượng dự đoán trong môi trường vận hành thật và tạo nền tảng để mở rộng hệ thống trong các nghiên cứu tiếp theo.
