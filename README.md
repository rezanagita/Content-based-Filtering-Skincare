# Laporan Proyek Machine Learning - REZA NAGITA NURHAZIZAH

## Project Overview : Sistem Rekomendasi Produk Skincare Berbasis Content-Based Filtering

Dalam beberapa tahun terakhir, minat konsumen terhadap produk skincare meningkat secara signifikan seiring dengan kesadaran akan pentingnya perawatan kulit untuk menjaga kesehatan dan penampilan. Namun, banyaknya pilihan produk skincare di pasaran sering kali membuat konsumen bingung dalam memilih produk yang sesuai dengan kebutuhan kulit mereka. Setiap individu memiliki jenis kulit yang berbeda, seperti normal, kering, berminyak, kombinasi, atau sensitif, serta masalah kulit spesifik seperti jerawat, pori-pori besar, atau pigmentasi. Oleh karena itu, sistem rekomendasi produk skincare yang dipersonalisasi menjadi solusi penting untuk membantu konsumen menemukan produk yang sesuai dengan karakteristik kulit dan preferensi mereka.

Sistem rekomendasi berbasis konten (content-based filtering) adalah pendekatan yang efektif untuk merekomendasikan produk berdasarkan kesamaan fitur, seperti komposisi bahan (ingredients), jenis kulit yang ditargetkan, atau manfaat produk. Dengan memanfaatkan teknik cosine similarity, sistem ini dapat mengukur kesamaan antara profil preferensi pengguna (berdasarkan jenis kulit atau bahan yang diinginkan) dan atribut produk dalam dataset. Dataset yang digunakan dalam proyek ini berasal dari [Kaggle](https://www.kaggle.com/datasets/muhammadrefki/dataset-skincare), yaitu "Dataset Skincare" oleh Muhammad Refki, yang berisi informasi tentang produk skincare, termasuk nama produk, merek, jenis kulit yang cocok, bahan-bahan, dan manfaat produk.

**Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan**

Memilih produk skincare yang tidak sesuai dengan jenis kulit atau kebutuhan spesifik dapat menyebabkan efek samping, seperti iritasi, alergi, atau ketidakefektifan produk. Pendekatan tradisional, seperti mengandalkan produk terlaris atau saran dari toko, sering kali tidak mempertimbangkan variasi individu dalam kondisi kulit, sehingga kurang efektif. Sistem rekomendasi berbasis konten dengan cosine similarity dapat mengatasi masalah ini dengan:

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
1. Kesulitan Konsumen dalam Memilih Produk Skincare yang Sesuai: Konsumen sering kali menghadapi kebingungan dalam memilih produk skincare yang sesuai dengan jenis kulit mereka (misalnya, kering, berminyak, kombinasi, atau sensitif) dan kebutuhan spesifik (seperti anti-jerawat, hidrasi, atau pencerah kulit) karena banyaknya variasi produk di pasaran.
2. Kurangnya Personalisasi dalam Rekomendasi Produk: Pendekatan tradisional seperti rekomendasi berdasarkan popularitas produk atau saran umum dari toko tidak mempertimbangkan preferensi individu, seperti jenis kulit atau bahan yang diinginkan, sehingga sering kali menghasilkan rekomendasi yang tidak relevan atau tidak efektif.
3. Risiko Efek Samping dari Pemilihan Produk yang Tidak Tepat: Penggunaan produk skincare yang tidak sesuai dengan kondisi kulit dapat menyebabkan iritasi, alergi, atau masalah kulit lainnya, yang menurunkan kepuasan konsumen dan kepercayaan terhadap merek atau platform e-commerce.

### Goals
1. Menyediakan Rekomendasi Produk Skincare yang Dipersonalisasi: Mengembangkan sistem rekomendasi berbasis konten yang dapat merekomendasikan produk skincare berdasarkan jenis kulit dan preferensi bahan pengguna, menggunakan dataset skincare dari Kaggle (https://www.kaggle.com/datasets/muhammadrefki/dataset-skincare).
2. Meningkatkan Relevansi Rekomendasi: Memanfaatkan teknik cosine similarity untuk memastikan rekomendasi produk memiliki kesamaan fitur yang tinggi dengan preferensi pengguna, sehingga meningkatkan kepuasan dan kepercayaan konsumen.
3. Mengurangi Risiko Pemilihan Produk yang Tidak Tepat: Memberikan rekomendasi yang akurat berdasarkan analisis bahan dan manfaat produk, sehingga meminimalkan kemungkinan efek samping seperti iritasi atau ketidakefektifan produk.

### Solution Approach
#### Solution statements
1. **Content-Based Filtering dengan Cosine Similarity dan TF-IDF**: Pendekatan ini menggunakan cosine similarity untuk mengukur kesamaan antara vektor preferensi pengguna (berdasarkan jenis kulit atau bahan yang diinginkan) dan vektor fitur produk (seperti bahan, manfaat, atau jenis kulit yang ditargetkan). Data teks seperti daftar bahan akan diproses menggunakan Term Frequency-Inverse Document Frequency (TF-IDF) untuk mengubahnya menjadi representasi vektor numerik.
  - Keunggulan: Metode ini efektif untuk dataset dengan informasi teks yang kaya, seperti daftar bahan dan deskripsi produk. Cosine similarity memungkinkan perhitungan kesamaan yang akurat dan cepat.
  - Implementasi: Dataset akan diproses untuk mengekstrak fitur seperti bahan dan jenis kulit, kemudian diubah menjadi vektor TF-IDF. Cosine similarity akan digunakan untuk mencari produk dengan skor kesamaan tertinggi.

2. **Content-Based Filtering dengan Word Embeddings (Word2Vec atau GloVe):** Menggunakan model word embeddings seperti Word2Vec atau GloVe untuk merepresentasikan bahan dan deskripsi produk dalam ruang vektor berdimensi tinggi. Kesamaan dihitung menggunakan cosine similarity atau metrik lainnya.
  - Keunggulan: Word embeddings dapat menangkap hubungan semantik antara bahan, sehingga rekomendasi lebih kontekstual dan mampu menangani sinonim atau istilah terkait.
  - Implementasi: Model word embeddings akan dilatih atau menggunakan model pra-latih pada data teks bahan dan deskripsi produk. Vektor preferensi pengguna dibandingkan dengan vektor produk untuk menghasilkan rekomendasi.

3. **Hybrid Approach dengan Clustering:** Menggabungkan content-based filtering dengan teknik clustering (misalnya, K-Means) untuk mengelompokkan produk berdasarkan fitur seperti jenis kulit atau manfaat. Cosine similarity digunakan untuk merekomendasikan produk dari klaster yang relevan.
  - Keunggulan: Mengurangi kompleksitas perhitungan dengan membatasi pencarian pada klaster tertentu, sehingga lebih efisien untuk dataset besar.
  - Implementasi: Dataset akan dianalisis untuk mengelompokkan produk menggunakan K-Means berdasarkan fitur seperti bahan atau jenis kulit. Cosine similarity diterapkan dalam klaster yang sesuai dengan input pengguna.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai jumlah data, kondisi data, dan informasi mengenai data yang digunakan. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya, uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
