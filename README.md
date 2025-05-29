# Laporan Proyek Machine Learning - REZA NAGITA NURHAZIZAH

## Project Overview : Sistem Rekomendasi Produk Skincare Berbasis Content-Based Filtering

Dalam beberapa tahun terakhir, minat konsumen terhadap produk skincare meningkat secara signifikan seiring dengan kesadaran akan pentingnya perawatan kulit untuk menjaga kesehatan dan penampilan. Namun, banyaknya pilihan produk skincare di pasaran sering kali membuat konsumen bingung dalam memilih produk yang sesuai dengan kebutuhan kulit mereka. Setiap individu memiliki jenis kulit yang berbeda, seperti normal, kering, berminyak, kombinasi, atau sensitif, serta masalah kulit spesifik seperti jerawat, pori-pori besar, atau pigmentasi. Oleh karena itu, sistem rekomendasi produk skincare yang dipersonalisasi menjadi solusi penting untuk membantu konsumen menemukan produk yang sesuai dengan karakteristik kulit dan preferensi mereka.[1]

Sistem rekomendasi berbasis konten (content-based filtering) adalah pendekatan yang efektif untuk merekomendasikan produk berdasarkan kesamaan fitur, seperti komposisi bahan (ingredients), jenis kulit yang ditargetkan, atau manfaat produk.[2] Dengan memanfaatkan teknik cosine similarity, sistem ini dapat mengukur kesamaan antara profil preferensi pengguna (berdasarkan jenis kulit atau bahan yang diinginkan) dan atribut produk dalam dataset. Dataset yang digunakan dalam proyek ini berasal dari [Kaggle](https://www.kaggle.com/datasets/muhammadrefki/dataset-skincare), yaitu "Dataset Skincare" oleh Muhammad Refki, yang berisi informasi tentang produk skincare, termasuk nama produk, merek, jenis kulit yang cocok, bahan-bahan, dan manfaat produk.

**Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan**

Memilih produk skincare yang tidak sesuai dengan jenis kulit atau kebutuhan spesifik dapat menyebabkan efek samping, seperti iritasi, alergi, atau ketidakefektifan produk. Pendekatan tradisional, seperti mengandalkan produk terlaris atau saran dari toko, sering kali tidak mempertimbangkan variasi individu dalam kondisi kulit, sehingga kurang efektif.[3] Sistem rekomendasi berbasis konten dengan cosine similarity dapat mengatasi masalah ini dengan:

1. Personalisasi: Menganalisis preferensi pengguna (misalnya, jenis kulit atau bahan yang diinginkan) dan mencocokkannya dengan produk yang memiliki fitur serupa.

2. Efisiensi: Membantu konsumen menghemat waktu dalam memilih produk dari ribuan opsi yang tersedia di pasaran.

3. Akurasi: Menggunakan cosine similarity untuk menghitung kesamaan berbasis vektor, yang memungkinkan rekomendasi yang lebih tepat berdasarkan fitur produk, seperti bahan aktif atau manfaat spesifik.

Pendekatan ini relevan karena dapat meningkatkan pengalaman pengguna dalam memilih produk skincare, mengurangi risiko pembelian produk yang tidak cocok, dan mendukung pengambilan keputusan yang lebih terinformasi.

Referensi

[1] J. Lee, H. Yoon, S. Kim, C. Lee, J. Lee, and S. Yoo, "Deep learning-based skin care product recommendation: A focus on cosmetic ingredient analysis and facial skin conditions," J. Cosmet. Dermatol., vol. 23, pp. 2066–2077, 2024, doi: 10.1111/jocd.16218.

[2] G. Lee, J. V. Moniaga, R. A. Thejaswi, and R. B. Dayananda, "A content-based skincare product recommendation system," in Proc. Int. Conf. Inf. Commun. Technol. (ICOIACT), IEEE, 2023, pp. 1–6.

[3] Y. Nakajima, H. Honma, H. Aoshima, et al., "Recommender system based on user evaluations and cosmetic ingredients," in Proc. 4th Int. Conf. Inf. Technol. (InCIT), IEEE, 2019, pp. 22–27.

## Business Understanding

### Problem Statements
**1. Kesulitan Konsumen dalam Memilih Produk Skincare yang Sesuai:** Konsumen sering kali kesulitan memilih produk skincare yang sesuai dengan kebutuhan kulit mereka karena banyaknya variasi produk di pasaran, terutama terkait komposisi bahan (ingredients) yang relevan dengan jenis atau masalah kulit mereka.

**2. Kurangnya Personalisasi Berdasarkan Bahan Produk:** Rekomendasi produk skincare yang umum, seperti berdasarkan popularitas, sering kali tidak mempertimbangkan komposisi bahan yang sesuai dengan preferensi atau kebutuhan spesifik pengguna, seperti bahan aktif untuk hidrasi atau anti-jerawat.

**3. Risiko Pemilihan Produk yang Tidak Tepat:** Pemilihan produk tanpa mempertimbangkan kesamaan bahan dapat menyebabkan ketidakefektifan produk atau efek samping seperti iritasi, terutama bagi pengguna dengan kulit sensitif.

### Goals
**1. Menyediakan Rekomendasi Berbasis Komposisi Bahan:** Mengembangkan sistem rekomendasi berbasis konten yang merekomendasikan produk skincare berdasarkan kesamaan bahan (ingredients) menggunakan TF-IDF dan cosine similarity dari dataset skincare di Kaggle.

**2. Meningkatkan Relevansi Rekomendasi:** Memastikan rekomendasi produk memiliki kesamaan bahan yang tinggi dengan produk referensi yang dipilih pengguna, sehingga relevan dengan kebutuhan kulit mereka.

**3. Mengurangi Risiko Efek Samping:** Memberikan rekomendasi produk dengan bahan serupa untuk meminimalkan risiko iritasi atau ketidaksesuaian dengan kebutuhan kulit pengguna.

### Solution Approach
#### Solution statements
**Content-Based Filtering dengan TF-IDF dan Cosine Similarity:**
  * Deskripsi: Menggunakan TF-IDF untuk mengubah kolom Ingredients menjadi representasi vektor numerik, kemudian menghitung kesamaan antar produk menggunakan cosine similarity. Fungsi get_recommendations akan mengembalikan produk dengan skor kesamaan tertinggi berdasarkan input nama produk.
  * Keunggulan: Pendekatan ini efektif untuk menangkap kesamaan berbasis teks pada bahan produk, yang merupakan faktor utama dalam menentukan kecocokan produk skincare.
  * Implementasi: Kolom Ingredients akan diproses dengan TF-IDF Vectorizer, diikuti oleh perhitungan cosine similarity untuk membangun matriks kesamaan. Fungsi rekomendasi akan mengurutkan produk berdasarkan skor kesamaan dan mengembalikan top-N rekomendasi.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah [Dataset Skincare](https://www.kaggle.com/datasets/muhammadrefki/dataset-skincare) dari kaggle yang diunggah oleh Muhammad Refki pada tahun 2023. Dataset ini terdiri dari 185 entri (baris) dengan 9 kolom  termasuk nama produk, jenis produk, jenis kulit yang sesuai, masalah kulit yang ditargetkan, ukuran, bahan aktif utama, dan tahun rilis, yang relevan untuk pengembangan sistem rekomendasi berbasis konten (content-based filtering).

| Variabel | Keterangan |
| ------ | ------ |
| ID | ID produk skincare |
| Brand | Nama merek produk skincare |
| Nama Product | Nama produk skincare |
| Jenis produk | Kategori produk, misalnya pembersih, toner, serum, pelembap, atau tabir surya. |
| Jenis kulit | Jenis kulit yang menjadi target produk, seperti normal, kering, berminyak, kombinasi, atau sensitif |
| Masalah kulit | Masalah Kulit yang dialami(Jerawat, lek hitam, dehidrasi |
| Ukuran | Ukuran isi produk (misalnya: 50mL, 100mL, 250mL). |
| Bahan Aktif | Daftar bahan aktif atau komposisi utama yang terkandung dalam produk. |
| Tahun Rilis | Tahun pembuatan/rilis product  |

### Exploratory Data Analysis (EDA)
  **Statistik Deskriptif**
  ```
  skincare_product.describe(include='all')
  ```
  ![image](https://github.com/user-attachments/assets/898e300a-1d2f-42f2-bc0d-0b4cfa2ac5df)
  1. skincare_product.describe() menampilkan statistik numerik saja, insight yg didapatkan :
         - Terdapat 185 data / ID
         - Produk dengan tahun rilis paling lama di tahun 2018 dan terbaru 2023 serta rata-rata rilis produk di tahun 2020
  2. skincare_product.describe(include='all') menampilkan seluruh statistik dari variabel dataset, insight yang didapatkan, merk produk paling top/ dipakai adalah "Avoskin", nama produk      "ceramide barrier moisturizer
  3. Masalah kulit paling banyak yaitu karena kusam
  4. Statistik ini juga menampilkan bahwa semua data sebanyak 185 tidak ada missing value

  **Visualisasi Permasalahan Kulit yang dapat**
  Terlihat dengan jelas dari hasil visualisasi berikut bahwa banyak produk yang dapat menangani masalah kulit Kusam, Poti-pori besar, dan iritasi
  ![image](https://github.com/user-attachments/assets/19abfdcf-1656-4103-8a6f-7f9d817beefa)
  **Visualisasi produk skincare**
  Menampilkan produk dengan menggunakan word cloud untuk melihat frekuensi produk skincare, ini menampilkan produk dengan ingrediens yang dapat membantu mengatasi kulit kusam/ cocok direkomendasikan untuk permasalahan kulit kusam 
  ![image](https://github.com/user-attachments/assets/3650ccb3-1176-4aa1-a19f-cd7765891ae6)
  
## Data Preparation
### Langkah-langkah melakukan data preparation
1. **Mengubah nama variabel** yang ada di dataset skincare_produk karena masih memiliki variabel yang ambigu untuk dipahami, selain itu untuk memudahkan ke tahapan analisis selanjutnya
2. **Mengubah tipe data** variabel "Ukuran" dari object ke int dan menghapus kata "ml"
3. **Menghapus kolom ID** karena kurang relevan utuk digunakan pada proyek ini

## Modeling
Model sistem rekomendasi dibangun menggunakan pendekatan content-based filtering dengan TF-IDF dan cosine similarity. Berikut adalah langkah-langkah implementasi:

  **1. TF-IDF Vectorization:** mengubah kolom Ingredients menjadi vektor numerik yang merepresentasikan bobot relatif bahan dalam setiap produk.
  **2. Cosine Similarity:** mengukur kesamaan antar produk berdasarkan vektor tersebut(TF-IDF).
  **3. Fungsi Rekomendasi:** menerima nama produk sebagai input, mengambil indeks produk dari dataset, mengurutkan produk lain berdasarkan skor kesamaan, dan mengembalikan top-5 rekomendasi dengan informasi seperti Nama_Brand, Nama_Produk, Jenis_Produk, Ingredients, Jenis_Kulit, Masalah_Kulit, dan Similarity_Score.
  **4.Testing sistem rekomendasi content based filtering**
  ```
  get_recommendations('Glow Boost Serum')
  ```
| No | Nama_Brand     | Nama_Produk                          | Jenis_Produk | Ingredients     | Jenis_Kulit | Masalah_Kulit   | Similarity_Score |
|----|----------------|--------------------------------------|--------------|------------------|--------------|------------------|------------------|
| 1  | The Ordinary   | Bakuchiol Night Cream                | Serum        | Salicylic Acid   | Normal       | Flek hitam       | 1.0              |
| 2  | Avoskin        | Glow Boost Serum                     | Moisturizer  | Salicylic Acid   | Kering       | Jerawat          | 1.0              |
| 3  | Somethinc      | Miraculous Retinol Ampoule           | Sunscreen    | Salicylic Acid   | Normal       | Kusam            | 1.0              |
| 4  | The Ordinary   | Bakuchiol Night Cream                | Moisturizer  | Salicylic Acid   | Normal       | Bekas jerawat    | 1.0              |
| 5  | COSRX          | Luminous Whitening Night Cream       | Cleanser     | Salicylic Acid   | Sensitif     | Iritasi          | 1.0              |

**Kelebihan menggunakan pendekatan ini sebagai solusi**: 
- Dataset yang berukuran kecil ini dengan menggunakan TF-IDF sangat mudah diimplementasikan dan cepat
- Menggunakan teknik pengukuran jarak dengan _cosine similarity_ efektif dalam menangkap kesamaan fitur ingredients sebagai faktor utama dalam rekomendasi skincare
**Kekurangan menggunakan pendekatan ini sebagai solusi**:
  - keterbatasan semantik pada fitur ingredients yang hanya sedikit (1 bahan aktif saja) ketika dilakukan TF-IDF
  - tidak mempertimbangkan fitur lain atau hanya menggunakan Ingredients, sehingga Jenis_Kulit atau Masalah_Kulit tidak diintegrasikan langsung dalam perhitungan kesamaan.

## Evaluation
Untuk mengevaluasi performa sistem rekomendasi, pendekatan kualitatif digunakan karena content-based filtering tidak memiliki metrik evaluasi standar seperti akurasi pada model supervised learning. **Evaluasi dilakukan menggunakan metrik precision** dengan pendekatan otomatis berdasarkan **similarity threshold**. Metrik ini dipilih karena sesuai dengan konteks sistem rekomendasi berbasis konten, di mana relevansi rekomendasi diukur berdasarkan skor kesamaan (cosine similarity) terhadap produk input. Hasil evaluasi dianalisis untuk memastikan bahwa sistem mampu memberikan rekomendasi yang relevan sesuai dengan problem statement (yaitu, menyediakan rekomendasi produk yang sesuai dengan kebutuhan kulit berdasarkan bahan) dan goals (meningkatkan relevansi dan mengurangi risiko pemilihan produk yang tidak tepat).

```
precision_evaluation('Luminous Whitening Night Cream', k=10, threshold=0.7)
```
Dalam konteks ini, sebuah rekomendasi dianggap relevan jika skor cosine similarity-nya melebihi threshold tertentu (misalnya, 0.7). Precision dihitung dengan rumus berikut:
![image](https://github.com/user-attachments/assets/1cceaa67-4518-4a8c-a381-f6108a22e756)
 ** * Jumlah rekomendasi relevan:** Jumlah produk yang direkomendasikan dengan Similarity_Score di atas threshold.
 ** * Total rekomendasi (k):** Jumlah rekomendasi yang diminta (dalam kasus ini, k=10)
 ** * Threshold:** Batas minimum cosine similarity untuk menentukan relevansi (misalnya, 0.7, yang berarti hanya produk dengan skor kesamaan di atas 0.7 dianggap relevan).

**Cara Kerja Metrik**
  - Fungsi precision_evaluation memanggil fungsi get_recommendations untuk mendapatkan top-k rekomendasi berdasarkan cosine similarity.
  - Untuk setiap rekomendasi, Similarity_Score dibandingkan dengan threshold. Jika skor lebih besar dari threshold, rekomendasi tersebut dianggap relevan.
  - Precision dihitung sebagai persentase rekomendasi relevan dari total k rekomendasi.

### Hasil evaluasi 
![image](https://github.com/user-attachments/assets/53a4267c-db48-46e3-b130-dd275589428e)
Keterangan:
 **Precision 100%** menunjukkan bahwa semua rekomendasi memiliki kesamaan bahan yang tinggi, yang sesuai dengan tujuan proyek untuk memberikan rekomendasi berbasis komposisi bahan. Ini mendukung problem statement tentang kesulitan memilih produk yang sesuai dan goal untuk meningkatkan relevansi rekomendasi.
  
**---Ini adalah bagian akhir laporan---**
