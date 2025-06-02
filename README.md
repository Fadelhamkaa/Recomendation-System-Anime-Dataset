# Laporan Proyek Machine Learning - Muhammad Fadel Hamka

## Project Overview

### 1.1. Latar Belakang
Industri hiburan anime global menawarkan ribuan judul yang mencakup berbagai genre, tipe, dan demografi penonton. Dengan volume konten yang begitu besar—tersedia melalui platform streaming, situs basis data seperti MyAnimeList, dan forum diskusi—pengguna seringkali menghadapi tantangan dalam menemukan anime baru yang benar-benar sesuai dengan preferensi unik mereka. Fenomena ini dikenal sebagai *information overload* atau *choice overload*, di mana banyaknya pilihan justru dapat menyebabkan kebingungan dan mengurangi kepuasan pengguna dalam mengeksplorasi konten. Proyek ini bertujuan untuk mengatasi masalah tersebut dengan merancang dan mengimplementasikan sebuah sistem rekomendasi anime menggunakan dataset "Anime Recommendations Database" dari Kaggle. Sistem ini akan memanfaatkan teknik *machine learning* untuk menganalisis preferensi pengguna dan karakteristik anime, sehingga dapat memberikan saran tontonan yang dipersonalisasi.

### 1.2. Pentingnya Proyek untuk Diselesaikan
Penyelesaian masalah *information overload* dalam penemuan konten anime menjadi penting karena beberapa alasan:
* **Meningkatkan Pengalaman Pengguna:** Dengan menyajikan rekomendasi yang relevan, pengguna dapat lebih mudah menemukan anime yang mereka nikmati tanpa harus menghabiskan banyak waktu untuk mencari secara manual. Ini secara langsung meningkatkan kepuasan dan *engagement* pengguna.
* **Mendukung Eksplorasi Konten:** Sistem rekomendasi yang baik dapat membantu pengguna menemukan judul-judul anime yang mungkin tidak populer namun berkualitas tinggi atau sesuai dengan selera spesifik mereka (*niche content*), sehingga memperluas wawasan tontonan mereka.
* **Potensi Manfaat bagi Platform (Hipotetis):** Bagi platform penyedia layanan anime, sistem rekomendasi yang efektif dapat meningkatkan metrik penting seperti durasi menonton, retensi pengguna, dan diversitas konten yang dikonsumsi.
* **Pembelajaran dan Aplikasi Keterampilan:** Proyek ini memberikan kesempatan untuk menerapkan konsep dan teknik *machine learning* dalam konteks dunia nyata, khususnya dalam membangun sistem rekomendasi yang merupakan salahA satu aplikasi AI paling umum dan berdampak.

### 1.3. Hasil Riset atau Referensi Terkait
Pengembangan sistem rekomendasi didasarkan pada penelitian dan praktik yang sudah mapan di bidang *machine learning* dan analisis data. Beberapa poin dan referensi yang relevan:
* Sistem rekomendasi adalah alat yang fundamental dalam personalisasi layanan digital, seperti yang telah berhasil diimplementasikan oleh platform besar seperti Netflix untuk film dan serial TV, Spotify untuk musik, dan Amazon untuk produk. Kesuksesan mereka menunjukkan nilai signifikan dari rekomendasi yang akurat.
* Pendekatan utama dalam sistem rekomendasi, yaitu *Content-Based Filtering* dan *Collaborative Filtering* (serta pendekatan hibrida), telah banyak diteliti dan terbukti efektif dalam berbagai domain. Proyek ini akan mengeksplorasi kedua pendekatan tersebut.

**Referensi (Contoh Format IEEE):**
[1] C. A. Gomez-Uribe and N. Hunt, "The Netflix Recommender System: Algorithms, Business Value, and Innovation," *ACM Transactions on Management Information Systems (TMIS)*, vol. 6, no. 4, pp. 1-19, Jan. 2016.
[2] M. D. Ekstrand, J. T. Riedl, and J. A. Konstan, "Collaborative filtering recommender systems," *Foundations and Trends® in Human–Computer Interaction*, vol. 4, no. 2, pp. 81-173, 2011.
[3] F. Ricci, L. Rokach, and B. Shapira, *Recommender Systems Handbook*. Springer, 2011.


---

## Business Understanding

Pada bagian ini, dijelaskan proses klarifikasi masalah yang dihadapi dalam konteks penemuan konten anime dan bagaimana proyek ini bertujuan untuk menyelesaikannya. Industri anime memiliki katalog konten yang sangat luas, yang seringkali menyulitkan pengguna untuk menemukan judul baru yang sesuai dengan preferensi pribadi mereka. Proyek ini berfokus pada pemanfaatan data rating pengguna dan atribut anime untuk membangun sistem rekomendasi yang efektif.

### Problem Statements
Berikut adalah pernyataan masalah yang akan dijawab oleh proyek ini:
- **Pernyataan Masalah 1:** Bagaimana cara mengurangi kesulitan pengguna dalam menemukan anime baru yang relevan dengan selera mereka di tengah banyaknya pilihan yang tersedia dalam dataset "Anime Recommendations Database"?
- **Pernyataan Masalah 2:** Bagaimana sistem dapat memberikan rekomendasi anime yang dipersonalisasi kepada setiap pengguna berdasarkan riwayat tontonan dan rating yang mereka berikan pada anime lain?
- **Pernyataan Masalah 3:** Bagaimana cara membantu pengguna menemukan anime yang memiliki kemiripan karakteristik (seperti genre atau tipe) dengan anime tertentu yang sudah mereka sukai, terutama untuk pengguna yang mungkin belum memiliki banyak data interaksi?

### Goals
Tujuan proyek yang menjawab pernyataan masalah di atas adalah sebagai berikut:
- **Jawaban pernyataan masalah 1 & 2:** Mengembangkan sebuah sistem rekomendasi yang mampu menghasilkan daftar top-N anime yang dipersonalisasi untuk pengguna, berdasarkan analisis data rating dan preferensi dari "Anime Recommendations Database".
- **Jawaban pernyataan masalah 3 & Tujuan Umum Proyek:** Menerapkan dan membandingkan dua pendekatan utama dalam sistem rekomendasi, yaitu Content-Based Filtering (berdasarkan fitur anime seperti genre dan tipe) dan Collaborative Filtering (berdasarkan data rating pengguna).
- **Tujuan Tambahan Proyek:**
    * Mengevaluasi performa kedua model rekomendasi menggunakan metrik evaluasi yang sesuai untuk mengukur efektivitasnya dalam konteks dataset yang digunakan.
    * Menyediakan output rekomendasi yang dapat dengan mudah diinterpretasikan oleh pengguna.

### Solution statements (Solution Approach)
Untuk mencapai tujuan yang telah ditetapkan, proyek ini akan mengimplementasikan dan mengeksplorasi dua pendekatan solusi utama (*solution approach*) dalam sistem rekomendasi:

1.  **Content-Based Filtering:**
    Pendekatan ini akan merekomendasikan anime kepada pengguna berdasarkan kemiripan atribut atau konten dari anime itu sendiri. Atribut yang akan dipertimbangkan dari `anime.csv` meliputi `genre`, `type`, dan fitur lain yang relevan seperti `rating` (rata-rata rating anime) atau `members`. Metode ini berguna untuk merekomendasikan item kepada pengguna yang memiliki preferensi spesifik terhadap atribut tertentu atau untuk mengatasi masalah *cold start* pada item baru yang belum memiliki banyak interaksi pengguna namun sudah memiliki deskripsi fitur.

2.  **Collaborative Filtering:**
    Pendekatan ini bekerja dengan menganalisis pola perilaku pengguna dalam skala besar, khususnya data rating yang terdapat dalam `rating.csv`. Rekomendasi dihasilkan berdasarkan kesamaan preferensi antara pengguna (*user-based CF*) atau kesamaan pola rating antar anime (*item-based CF*), atau melalui teknik faktorisasi matriks (*model-based CF*) seperti Singular Value Decomposition (SVD). Metode ini efektif dalam menemukan rekomendasi yang bersifat *serendipitous* atau tidak terduga namun tetap relevan, serta mampu menangkap preferensi implisit pengguna yang mungkin tidak terwakili hanya oleh fitur konten.

Dengan menerapkan kedua pendekatan ini, diharapkan sistem rekomendasi yang dibangun dapat memberikan saran yang lebih komprehensif dan akurat kepada pengguna berdasarkan dataset yang tersedia.

---

## Data Understanding

Tahap ini bertujuan untuk memahami data yang akan digunakan dalam proyek. Dataset yang dipilih adalah "Anime Recommendations Database" yang bersumber dari Kaggle (diunggah oleh pengguna CooperUnion). Tautan untuk mengunduh dataset: [https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database). Dataset ini terdiri dari dua file utama: `anime.csv` dan `rating.csv`.

Berdasarkan inspeksi awal (output dari sel 15 & 17 di notebook):
* File `anime.csv` berisi informasi mengenai **12.294 judul anime** dengan 7 fitur.
* File `rating.csv` berisi **7.813.737 data rating** yang diberikan oleh **73.515 pengguna** terhadap **11.200 anime**.
* Kondisi data awal menunjukkan adanya beberapa nilai yang hilang (*missing values*) pada `anime.csv` di kolom `genre`, `type`, dan `rating` (rata-rata anime). Kolom `episodes` juga memiliki nilai 'Unknown' yang perlu penanganan. Sementara itu, `rating.csv` tidak memiliki missing values pada kolom-kolom utamanya.

### Variabel-variabel pada Dataset
Berikut adalah penjelasan untuk setiap variabel dalam dataset yang digunakan:

**Pada `anime.csv` (Notebook: Sel 18):**
* `anime_id`: ID unik untuk setiap anime (dari myanimelist.net).
* `name`: Judul lengkap anime.
* `genre`: Daftar genre yang dipisahkan koma untuk anime tersebut.
* `type`: Tipe anime (misalnya, TV, Movie, OVA, Special, ONA, Music).
* `episodes`: Jumlah episode dalam anime (nilai '1' jika tipe Movie). Bisa juga 'Unknown'.
* `rating`: Rating rata-rata anime (skala 1-10) berdasarkan semua pengguna di MyAnimeList.
* `members`: Jumlah anggota komunitas MyAnimeList yang memiliki anime ini dalam daftar mereka.

**Pada `rating.csv` (Notebook: Sel 18):**
* `user_id`: ID unik yang digenerate secara acak untuk setiap pengguna.
* `anime_id`: ID anime yang dinilai oleh pengguna (merujuk ke `anime_id` di `anime.csv`).
* `rating`: Rating yang diberikan pengguna untuk anime tersebut (skala 1-10). Nilai `-1` berarti pengguna telah menonton anime tersebut tetapi tidak memberikan rating.

### Exploratory Data Analysis (EDA)
Beberapa tahapan visualisasi data dilakukan untuk memahami distribusi dan karakteristik data lebih lanjut:

* **Distribusi Tipe Anime (`anime_df`):** Visualisasi menunjukkan bahwa tipe 'TV' adalah yang paling umum (lebih dari 3700 anime), diikuti oleh 'OVA' dan 'Movie'. Tipe 'Music' memiliki jumlah paling sedikit. (Merujuk pada output Sel 21 notebook) ![image](https://github.com/Fadelhamkaa/Recomendation-System-Anime-Dataset/blob/main/Screenshot%202025-06-01%20113502.png)
* **Distribusi Jumlah Episode (`anime_df`):** Sebagian besar anime memiliki jumlah episode yang relatif pendek (median 2 episode, 75% di bawah 12 episode). Terdapat 340 anime dengan jumlah episode 'Unknown'. (Merujuk pada output Sel 22 notebook) ![image](https://github.com/Fadelhamkaa/Recomendation-System-Anime-Dataset/blob/main/Screenshot%202025-06-01%20113521.png)
* **Distribusi Rating Rata-rata Anime (`anime_df`):** Rating rata-rata anime terdistribusi mendekati normal dengan puncak di sekitar 6.57 (median). (Merujuk pada output Sel 23 notebook) ![image](https://github.com/Fadelhamkaa/Recomendation-System-Anime-Dataset/blob/main/Screenshot%202025-06-01%20113537.png)
* **Distribusi Jumlah Anggota Komunitas (`anime_df`):** Distribusi jumlah anggota sangat condong ke kanan (*right-skewed*), mengindikasikan bahwa sebagian kecil anime sangat populer, sementara mayoritas memiliki jumlah anggota yang lebih sedikit. (Merujuk pada output Sel 24 notebook) ![image](https://github.com/Fadelhamkaa/Recomendation-System-Anime-Dataset/blob/main/Screenshot%202025-06-01%20113550.png)
* **Analisis Genre Anime (`anime_df`):** Genre 'Comedy' adalah yang paling umum (lebih dari 4600 anime), diikuti oleh 'Action', 'Adventure', 'Fantasy', dan 'Sci-Fi'. (Merujuk pada output Sel 25 notebook) ![image](https://github.com/Fadelhamkaa/Recomendation-System-Anime-Dataset/blob/main/Screenshot%202025-06-01%20113603.png)
* **Distribusi Rating Pengguna (`rating_df`):** Pengguna paling sering memberikan rating 7, 8, dan 9. Jumlah anime yang 'Watched (No Rating)' (-1) sangat signifikan (sekitar 1,47 juta). (Merujuk pada output Sel 26 notebook) ![image](https://github.com/Fadelhamkaa/Recomendation-System-Anime-Dataset/blob/main/Screenshot%202025-06-01%20113631.png)
* **Jumlah Rating per Pengguna (`rating_df`):** Distribusinya juga condong ke kanan. Banyak pengguna memberikan relatif sedikit rating (median 57 rating), sementara beberapa pengguna sangat aktif. (Merujuk pada output Sel 27 notebook) ![image](https://github.com/Fadelhamkaa/Recomendation-System-Anime-Dataset/blob/main/Screenshot%202025-06-03%20004824.png)
* **Jumlah Rating per Anime (`rating_df`):** Distribusinya juga sangat condong ke kanan (*long-tail*). Sebagian kecil anime menerima sangat banyak rating, sementara mayoritas hanya menerima sedikit rating (median sekitar 51 rating). (Merujuk pada output Sel 28 notebook) ![image](https://github.com/Fadelhamkaa/Recomendation-System-Anime-Dataset/blob/main/Screenshot%202025-06-03%20004830.png)

### **Ringkasan Temuan Data Understanding:**

* **Dataset Anime (`anime.csv`):**
    * Memiliki **12.294 entri anime unik**.
    * Informasi detail tentang atribut anime mencakup `genre`, `type`, `episodes`, `rating` (rata-rata anime), dan `members`.
    * Terdapat nilai yang hilang (missing values) pada kolom `genre`, `type`, dan `rating` (rata-rata anime).
    * Kolom `episodes` bertipe *object* karena adanya **340 entri bernilai 'Unknown'** yang perlu ditangani jika fitur ini akan digunakan secara numerik. Sebagian besar anime memiliki jumlah episode sedikit (median 2, 75% di bawah 12 episode).
    * Distribusi jumlah `members` sangat condong ke kanan (right-skewed), mengindikasikan bahwa hanya sebagian kecil anime yang sangat populer.
    * Rating rata-rata anime terdistribusi mendekati normal dengan **median sekitar 6.57**.
    * Tipe anime yang paling dominan adalah **'TV'**.

* **Dataset Rating (`rating.csv`):**
    * Berisi lebih dari **7,8 juta entri rating** dari **73.515 pengguna unik** untuk **11.200 anime unik**.
    * Tidak ditemukan missing values pada kolom-kolomnya.
    * Menyimpan interaksi pengguna dengan anime, termasuk rating eksplisit (skala 1-10) dan indikasi telah ditonton tanpa rating (nilai **-1**, yang jumlahnya signifikan, sekitar **1,47 juta entri**).

* **Pola Umum pada Data:**
    * Distribusi data untuk jumlah `members` (anime), jumlah rating per pengguna (median 57 rating/pengguna), dan jumlah rating per anime (median sekitar 51 rating/anime) cenderung **sangat condong ke kanan (right-skewed atau long-tail)**. Ini menunjukkan adanya item/pengguna yang sangat populer/aktif sementara mayoritas lainnya kurang, yang mengarah pada potensi masalah *sparsity*.
    * Genre paling populer antara lain **Comedy (sekitar 4600+ anime), Action (2800+), Adventure (2300+), Fantasy (2300+), dan Sci-Fi (2000+)**.
    * Data rating pengguna menunjukkan preferensi yang kuat pada **skor tinggi (paling banyak rating 7, 8, dan 9)**.

* **Implikasi untuk Tahap Selanjutnya:**
    * Temuan ini akan menjadi dasar krusial untuk tahap **Data Preparation**. Langkah-langkah seperti penanganan missing values, konversi tipe data (misalnya, `episodes`), dan transformasi fitur (misalnya, untuk `genre`) akan diperlukan.
    * Sifat data yang *sparse* dan *skewed* juga perlu dipertimbangkan saat memilih dan mengimplementasikan algoritma model rekomendasi.

---

## Data Preparation
Pada tahap Data Preparation (Notebook: Sel 30-44), beberapa langkah kunci telah dilakukan untuk membersihkan dan mentransformasi dataset agar siap untuk pemodelan. Proses ini dilakukan secara berurutan dan setiap teknik diterapkan dengan alasan tertentu untuk meningkatkan kualitas data.

### 4.1. Pembuatan Salinan DataFrame (Notebook: Sel 31)
* **Teknik & Proses:** Membuat salinan dari DataFrame asli (`anime_df` dan `rating_df`) menjadi `anime_prep_df` dan `rating_prep_df`.
* **Alasan:** Ini adalah praktik terbaik untuk menjaga integritas data asli, memungkinkan analisis ulang atau perbandingan jika diperlukan tanpa mengubah sumber data awal.

### 4.2. Penanganan Missing Values pada `anime_prep_df` (Notebook: Sel 33)
Berdasarkan EDA, beberapa kolom di `anime_prep_df` memiliki nilai yang hilang.
* **Kolom `genre`:**
    * **Teknik:** Imputasi dengan nilai placeholder.
    * **Proses:** 62 missing values diisi dengan string 'Unknown'.
    * **Alasan:** Genre adalah fitur penting untuk Content-Based Filtering. Mengisi dengan 'Unknown' memungkinkan anime tersebut tetap diproses.
* **Kolom `type`:**
    * **Teknik:** Imputasi dengan modus.
    * **Proses:** 25 missing values diisi dengan modus ('TV').
    * **Alasan:** Jumlah missing values relatif kecil, dan modus ('TV') adalah tipe paling dominan, sehingga imputasi ini tidak banyak mengubah distribusi.
* **Kolom `rating` (rata-rata anime):**
    * **Teknik:** Imputasi dengan median.
    * **Proses:** 230 missing values diisi dengan median (6.57).
    * **Alasan:** Median lebih robust terhadap outlier dibandingkan mean untuk distribusi rating.
Output dari sel 17 (notebook) mengkonfirmasi bahwa kolom `genre`, `type`, dan `rating` (rata-rata anime) tidak lagi memiliki missing values.

### 4.3. Pemrosesan Kolom `episodes` pada `anime_prep_df` (Notebook: Sel 35)
* **Teknik:** Penggantian nilai spesifik, konversi tipe data, dan imputasi median.
* **Proses:** 340 entri 'Unknown' diubah menjadi `NaN`, kolom dikonversi ke numerik, lalu `NaN` diisi dengan median jumlah episode (2.0).
* **Alasan:** Konversi ke numerik diperlukan untuk analisis kuantitatif. Imputasi median dipilih karena distribusi episode yang *skewed*.
Output dari sel 18 (notebook) menunjukkan kolom `episodes` kini bertipe `float64` dan bebas missing values.

### 4.4. Penanganan Rating `-1` pada `rating_prep_df` (Notebook: Sel 37)
* **Teknik:** Filtering data.
* **Proses:** Entri rating bernilai -1 (sekitar 1,47 juta entri, menandakan ditonton tanpa skor) difilter. Hasilnya adalah `explicit_rating_df` dengan ~6,3 juta rating eksplisit (1-10).
* **Alasan:** Untuk model Collaborative Filtering berbasis rating eksplisit (seperti SVD), hanya skor rating aktual yang relevan dan menyederhanakan skala rating.

### 4.5. Feature Engineering untuk Content-Based Filtering pada `anime_prep_df`
* **Kolom `genre` (Notebook: Sel 39):**
    * **Teknik:** TF-IDF (Term Frequency-Inverse Document Frequency).
    * **Proses:** Kolom `genre` diproses menjadi token-token genre individual, kemudian TF-IDF Vectorizer diterapkan, menghasilkan matriks fitur (12294 anime x 48 fitur genre).
    * **Alasan:** TF-IDF menangkap pentingnya genre dalam anime relatif terhadap dataset, memberikan bobot lebih pada genre spesifik.
* **Kolom `type` (Notebook: Sel 41):**
    * **Teknik:** One-Hot Encoding.
    * **Proses:** Kolom `type` diubah menjadi 6 kolom biner baru (misalnya, `type_Movie`, `type_TV`).
    * **Alasan:** Merepresentasikan data kategorikal nominal tanpa asumsi urutan.

### 4.6. Finalisasi dan Verifikasi (Notebook: Sel 43)
* **Proses:** Pengecekan duplikasi `anime_id` pada `anime_prep_df` (ditemukan 0 duplikasi).
* **Alasan:** Memastikan integritas data anime.

### **Ringkasan Tahap Data Preparation:**

Pada tahap Data Preparation, beberapa langkah kunci telah dilakukan untuk membersihkan dan mentransformasi dataset `anime.csv` dan `rating.csv` agar siap untuk pemodelan:

* **Penanganan Missing Values pada `anime_prep_df`:**
    * Missing values pada kolom `genre` berhasil diisi dengan placeholder 'Unknown'.
    * Kolom `type` yang memiliki missing values diisi menggunakan modus ('TV').
    * Kolom `rating` (rata-rata rating anime) yang hilang diimputasi dengan nilai median (6.57).
    * Hasilnya, `anime_prep_df` kini tidak memiliki missing values pada kolom-kolom krusial tersebut.

* **Pemrosesan Kolom `episodes` pada `anime_prep_df`:**
    * Nilai 'Unknown' (sebanyak 340 entri) pada kolom `episodes` berhasil diidentifikasi dan kemudian diimputasi dengan nilai median (2.0 episode).
    * Kolom `episodes` telah dikonversi menjadi tipe data numerik (`float64`) dan bebas dari missing values.

* **Penanganan Rating `-1` pada `rating_prep_df`:**
    * Entri rating dengan nilai -1 (menandakan anime ditonton tanpa skor, sebanyak ~1,47 juta entri) telah difilter dari `rating_prep_df`.
    * DataFrame baru, `explicit_rating_df`, dibuat dan kini berisi ~6,3 juta entri rating yang hanya mencakup skor eksplisit (1-10), siap untuk model Collaborative Filtering.

* **Feature Engineering untuk Content-Based pada `anime_prep_df`:**
    * Kolom `genre` telah diproses menggunakan TF-IDF, menghasilkan matriks fitur dengan dimensi (12294 anime, 48 fitur genre).
    * Kolom `type` telah diubah menggunakan One-Hot Encoding, menghasilkan 6 kolom fitur biner baru.

* **Finalisasi dan Verifikasi:**
    * Tidak ditemukan adanya duplikasi `anime_id` pada `anime_prep_df`.
    * `anime_prep_df` kini lebih bersih, dengan tipe data yang sesuai dan fitur-fitur baru hasil engineering (`genre_processed` untuk TF-IDF dan kolom-kolom `type_` dari OHE) yang siap mendukung pemodelan Content-Based.
    * `explicit_rating_df` telah disiapkan khusus untuk pemodelan Collaborative Filtering berbasis rating eksplisit.

Secara keseluruhan, data telah melalui pembersihan dan transformasi penting, menjadikannya lebih robust dan sesuai untuk tahap pengembangan model rekomendasi selanjutnya.

---

## Modeling and Result
Pada tahap ini (Notebook: Sel 45-55), dua model sistem rekomendasi dikembangkan dan diuji untuk memberikan top-N rekomendasi.

### 5.1. Content-Based Filtering
Pendekatan ini merekomendasikan anime berdasarkan kemiripan fitur konten, khususnya `genre`, menggunakan TF-IDF dan Cosine Similarity.

* **Proses Pemodelan (Notebook: Sel 47):**
    1.  Matriks TF-IDF untuk genre (dimensi 12294x48) yang telah dibuat sebelumnya digunakan.
    2.  Matriks Cosine Similarity (12294x12294) dihitung dari matriks TF-IDF untuk mendapatkan skor kemiripan antar semua pasangan anime.
    3.  Sebuah fungsi `get_anime_recommendations_content_based` dibuat untuk mengambil judul anime sebagai input dan mengembalikan top-N anime paling mirip berdasarkan skor kemiripan genre.

* **Hasil Rekomendasi (Top-N) (Notebook: Sel 49):**
    * **Untuk "Shingeki no Kyojin":** Top-5 rekomendasi adalah seri turunan dari franchise yang sama (OVA, Movie, Special, Season 2) dengan skor kemiripan 1.0. Ini menunjukkan model berhasil mengidentifikasi variasi dari judul yang sama.
    * **Untuk "Kimi no Na wa.":** Top-5 rekomendasi ("Wind: A Breath of Heart OVA & TV", "Aura: Maryuuin Kouga Saigo no Tatakai", dll.) memiliki genre yang tumpang tindih (Drama, Romance, School, Supernatural) dengan skor kemiripan antara 0.860 hingga 1.0. Ini menunjukkan model mampu menemukan anime lain dengan profil genre serupa.
    * **Untuk "Mushishi":** Rekomendasi juga didominasi oleh sekuel dan spesial dari franchise "Mushishi" dengan skor kemiripan 1.0.

### 5.2. Collaborative Filtering (SVD)
Pendekatan ini menggunakan algoritma Singular Value Decomposition (SVD) dari library `surprise` untuk memprediksi rating berdasarkan pola rating laten dari data historis pada `explicit_rating_df`.

* **Proses Pemodelan (Notebook: Sel 52-53):**
    1.  Data rating eksplisit (`explicit_rating_df`) diubah ke format `Dataset` yang dibutuhkan oleh library `surprise`, menggunakan `Reader` dengan skala rating 1-10.
    2.  Seluruh data ini digunakan untuk membangun `trainset` (full trainset).
    3.  Model SVD diinisialisasi (dengan parameter `n_factors=50`, `n_epochs=20`, dll.) dan dilatih menggunakan `trainset` tersebut. Proses pelatihan model SVD berhasil.
    4.  Sebuah fungsi `get_anime_recommendations_collaborative_svd` dibuat untuk mengambil `user_id` sebagai input, memprediksi rating untuk anime yang belum ditonton pengguna, dan mengembalikan top-N anime dengan prediksi rating tertinggi.

* **Hasil Rekomendasi (Top-N) (Notebook: Sel 55):**
    * **Untuk User ID 57243 (berdasarkan output notebook):** Top-5 rekomendasi dari model SVD adalah "Gintama", "Gintama&#039;", "Gintama°", "Ginga Eiyuu Densetsu", dan "Clannad: After Story". Anime-anime ini mendapatkan prediksi rating SVD yang sangat tinggi, yaitu 10.000 untuk ketiga seri "Gintama", 9.946 untuk "Ginga Eiyuu Densetsu", dan 9.925 untuk "Clannad: After Story". Rekomendasi ini mencakup anime dari genre Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen (untuk Gintama), Drama, Military, Sci-Fi, Space (untuk Ginga Eiyuu Densetsu), serta Drama, Fantasy, Romance, Slice of Life, Supernatural (untuk Clannad).

    

    * **Untuk User ID 27805 (berdasarkan output notebook):** Top-5 rekomendasi dari model SVD adalah "Kimi no Na wa.", "Ginga Eiyuu Densetsu", "Code Geass: Hangyaku no Lelouch R2", "Rurouni Kenshin: Meiji Kenkaku Romantan - Tsuiokuhen", dan "Fullmetal Alchemist: Brotherhood". Prediksi rating SVD untuk anime-anime ini juga sangat tinggi, berkisar antara 9.265 hingga 9.484. Rekomendasi ini menampilkan keberagaman genre, termasuk Drama, Romance, School, Supernatural; Drama, Military, Sci-Fi, Space; Action, Drama, Mecha; Action, Drama, Historical, Martial Arts; dan Action, Adventure, Fantasy.

### 5.3. Kelebihan dan Kekurangan Pendekatan yang Dipilih (Notebook: Sel 56)
* **Content-Based Filtering:**
    * **Kelebihan:** Relevansi tinggi untuk item serupa/franchise, transparan, tidak memerlukan data pengguna lain untuk rekomendasi item-ke-item.
    * **Kekurangan:** Terbatas pada kualitas fitur, serendipity rendah (cenderung monoton), potensi overspesialisasi.
* **Collaborative Filtering (SVD):**
    * **Kelebihan:** Menemukan rekomendasi lintas genre (serendipity), personalisasi berdasarkan perilaku pengguna, tidak memerlukan fitur item.
    * **Kekurangan:** Masalah *user/item cold start*, ketergantungan pada kualitas/kuantitas data rating (*sparsity*), kurang transparan ("black box"), potensi bias popularitas.

---

## Evaluation
Pada tahap evaluasi (Notebook: Sel 57-63), kinerja kedua model sistem rekomendasi diukur menggunakan metrik yang sesuai.

### 6.1. Metrik dan Evaluasi Model Content-Based Filtering

* **Metrik yang Digunakan:**
    * **Evaluasi Kualitatif (Notebook: Sel 59):** Menganalisis relevansi dan koherensi top-N rekomendasi secara manual.
        * **Cara Kerja & Kesesuaian:** Dilakukan dengan memeriksa apakah anime yang direkomendasikan memiliki genre yang mirip atau merupakan alternatif yang masuk akal. Ini sesuai untuk menilai apakah model menangkap kemiripan konten.
    * **Rata-rata Kemiripan Genre (Average Genre Jaccard Similarity @K) (Notebook: Sel 60):**
        * **Konsep & Cara Kerja:** Menghitung rata-rata koefisien Jaccard Similarity antara set genre anime input dengan set genre masing-masing anime yang direkomendasikan (top-K). Skor Jaccard (0-1) yang lebih tinggi menandakan kemiripan genre lebih besar.
        * **Formula Jaccard Similarity ($A$, $B$ adalah set genre):**
            $$J(A,B) = \frac{|A \cap B|}{|A \cup B|}$$
            * Dimana `$|A \cap B|$` adalah jumlah genre yang sama, dan `$|A \cup B|$` adalah jumlah total genre unik.
        * **Kesesuaian:** Mengukur aspek kemiripan genre yang merupakan inti model CBF ini.

* **Hasil Proyek Berdasarkan Metrik Evaluasi (Content-Based):**
    * **Evaluasi Kualitatif:** Rekomendasi untuk "Shingeki no Kyojin" dan "Mushishi" sangat relevan (seri turunan). Rekomendasi untuk "Kimi no Na wa." menunjukkan anime lain dengan genre inti serupa. Model bekerja sesuai harapan dalam menemukan tema serupa.
    * **Rata-rata Jaccard Similarity @5 (Output Notebook Sel 60):**
        * 'Shingeki no Kyojin': **1.0000** (genre identik).
        * 'Kimi no Na wa.': **0.8600** (tumpang tindih genre sangat tinggi).
        * 'Mushishi': **1.0000** (genre identik).
        Hasil Jaccard mendukung temuan kualitatif bahwa model berhasil menemukan anime dengan profil genre sangat mirip.

### 6.2. Metrik dan Evaluasi Model Collaborative Filtering (SVD)

* **Metrik yang Digunakan (Notebook: Sel 61):**
    * **Root Mean Squared Error (RMSE):** Mengukur rata-rata besarnya error antara rating prediksi dan aktual, memberi bobot lebih pada error besar. Semakin kecil, semakin baik.
        * **Formula:** $$RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$
            * $N$: Jumlah total rating prediksi. $y_i$: Rating aktual. $\hat{y}_i$: Rating prediksi.
    * **Mean Absolute Error (MAE):** Mengukur rata-rata error absolut antara rating prediksi dan aktual. Semakin kecil, semakin baik.
        * **Formula:** $$MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$
            * $N$: Jumlah total rating prediksi. $y_i$: Rating aktual. $\hat{y}_i$: Rating prediksi.
    * **Kesesuaian Metrik:** RMSE dan MAE adalah standar industri untuk mengevaluasi akurasi prediksi rating pada model berbasis rating eksplisit seperti SVD. Data dibagi 80% train dan 20% test untuk evaluasi ini.

* **Hasil Proyek Berdasarkan Metrik Evaluasi (SVD) (Output Notebook Sel 62):**
    * Jumlah data rating trainset: 5.069.792, testset: 1.267.449.
    * **RMSE: 1.1309**
    * **MAE: 0.8434**
    * **Interpretasi:** Untuk skala rating 1-10, RMSE ~1.13 dan MAE ~0.84 menunjukkan performa awal yang cukup baik untuk model SVD tanpa tuning ekstensif. Nilai ini mengindikasikan rata-rata deviasi prediksi dari rating aktual. Perbedaan RMSE > MAE wajar, menandakan adanya beberapa prediksi dengan error lebih besar.

---
**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Gambar, tabel, dan code snippet dari notebook dapat disisipkan di sini sesuai kebutuhan untuk memperjelas penjelasan._
- _Pastikan semua sumber dan kutipan telah sesuai dengan panduan yang diberikan._
