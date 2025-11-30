# -*- coding: utf-8 -*-

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, AffinityPropagation
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import estimate_bandwidth
from scipy.cluster.hierarchy import linkage, dendrogram
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import LabelEncoder


import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Mall_Customers.csv")

print("Veri Setinin Ilk 5 Örnegi: \n", df.head(5))
print("Veri Setinin Sekli: \n", df.shape)

df.rename(columns={"Genre":"Gender"}, inplace=True)

print(df.info())
print(df.describe())
print(df.isnull().sum())

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
sns.boxplot(data=df, y="Annual Income (k$)")
plt.subplot(1,2,2)
sns.boxplot(data=df, y="Spending Score (1-100)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.set_style('darkgrid')
sns.distplot(df.Age)
plt.title("Distribution of AGE", fontsize=20, color="blue")
plt.xlabel("Age Range", fontsize=15)
plt.ylabel("Density", fontsize=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.set_style('darkgrid')
sns.distplot(df["Annual Income (k$)"])
plt.title("Distribution of Annual Income (k$)", fontsize=20, color="#1D546C")
plt.xlabel("Annual Income (k$)", fontsize=15)
plt.ylabel("Density", fontsize=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.set_style('darkgrid')
sns.distplot(df["Spending Score (1-100)"])
plt.title("Distribution of Spending Score (1-100)", fontsize=20, color="#5E936C")
plt.xlabel("Spending Score (1-100)", fontsize=15)
plt.ylabel("Density", fontsize=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
sns.set_style('darkgrid')
plt.title("Distribution Gender", fontsize=20, color="pink")
plt.xlabel("Gender", fontsize=15)
plt.ylabel("Count", fontsize=15)
sns.countplot(df.Gender, palette="nipy_spectral_r")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.set_style('darkgrid')
sns.scatterplot(data=df, x="Age", y= "Annual Income (k$)", hue="Gender", palette={"Male":"blue", "Female":"pink"}, s=60)
plt.title("Age VS Annual Income (k$)", fontsize=20, color="#811844")
plt.xlabel("Age", fontsize=15)
plt.ylabel("Annual Income (k$)", fontsize=15)
plt.tight_layout()
plt.show()

Age_18_25 = df.Age[(df.Age>=18) & (df.Age<=25)]
Age_26_35 = df.Age[(df.Age>=26) & (df.Age<=35)]
Age_36_45 = df.Age[(df.Age>=36) & (df.Age<=45)]
Age_46_55 = df.Age[(df.Age>=46) & (df.Age<=55)]
Age_55_Above = df.Age[(df.Age>=56)]

x = ["18-25","26-35","36-45","46-55","55 Above"]
y = [len(Age_18_25.values),len(Age_26_35.values),len(Age_36_45.values),len(Age_46_55.values),len(Age_55_Above.values)]

plt.figure(figsize=(10,6))
sns.barplot(x=x, y=y, palette="pastel")
plt.title("Customer's Age Barplot", fontsize=20, color="pink")
plt.xlabel("Age", fontsize=15)
plt.ylabel("Number of Customers", fontsize=15)
plt.tight_layout()
plt.show()


ss1_20 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 1) & (df["Spending Score (1-100)"] <= 20)]
ss21_40 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 21) & (df["Spending Score (1-100)"] <= 40)]
ss41_60 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 41) & (df["Spending Score (1-100)"] <= 60)]
ss61_80 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 61) & (df["Spending Score (1-100)"] <= 80)]
ss81_100 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 81) & (df["Spending Score (1-100)"] <= 100)]

score_x = ["1-20", "21-40", "41-60", "61-80", "81-100"]
score_y = [len(ss1_20.values), len(ss21_40.values), len(ss41_60.values), len(ss61_80.values), len(ss81_100.values)]

plt.figure(figsize=(10,6))
sns.barplot(x=score_x, y=score_y,palette="pastel")
plt.title("Spending Scores", fontsize=20, color="#C2A68C")
plt.xlabel("Score", fontsize=15)
plt.ylabel("Number of Customers", fontsize=15)
plt.tight_layout()
plt.show()

ai0_30 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 0) & (df["Annual Income (k$)"] <= 30)]
ai31_60 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 31) & (df["Annual Income (k$)"] <= 60)]
ai61_90 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 61) & (df["Annual Income (k$)"] <= 90)]
ai91_120 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 91) & (df["Annual Income (k$)"] <= 120)]
ai121_150 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 121) & (df["Annual Income (k$)"] <= 150)]

income_x = ["$0 - 30,000", "$30,001 - 60,000", "$60,001 - 90,000", "$90,001 - 120,000", "$120,001 - 150,000"]
income_y = [len(ai0_30.values), len(ai31_60.values), len(ai61_90.values), len(ai91_120.values), len(ai121_150.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=income_x, y=income_y, palette="pastel")
plt.title("Annual Incomes", fontsize=20, color="red")
plt.xlabel("Income", fontsize=15)
plt.ylabel("Number of Customer", fontsize=15)
plt.tight_layout()
plt.show()

df_scaled = df[["Age","Annual Income (k$)","Spending Score (1-100)"]]

# Class instance
scaler = StandardScaler()

# Fit_transform
df_scaled_fit = scaler.fit_transform(df_scaled)
df_scaled_fit = pd.DataFrame(df_scaled_fit)
df_scaled_fit.columns = ["Age","Annual Income (k$)","Spending Score (1-100)"]
df_scaled_fit.head()
var_list = df_scaled_fit[["Annual Income (k$)","Spending Score (1-100)"]]

# Model Eðitimi
ssd = []
for num_clusters in range(1, 11):
    
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50, random_state=42, n_init=10)
    kmeans.fit(var_list)
    ssd.append(kmeans.inertia_)


plt.figure(figsize=(12,6))
plt.plot(range(1,11), ssd, linewidth=2, color="red", marker="o")
plt.title("Elbow Curve", fontsize=20, color="green")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("SSD")
plt.tight_layout()
plt.show()

# Final KMeans (Kritik Düzeltme: n_clusters=5 olarak sabitlendi)
kmeans = KMeans(n_clusters=5, max_iter=50, random_state=42, n_init=10) 
kmeans.fit(var_list)
df["Label"] = kmeans.labels_
df.head()

plt.figure(figsize=(10,6))
plt.title("Ploting the data into 5 clusters", fontsize=20, color="#3C467B")
sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Label", s=60, palette=["#AFCBFF", "#FFC1CC", "#B8E0D2", "#FFD8A9", "#D0BFFF"])
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='Label', y='Annual Income (k$)', data=df, palette="pastel")
plt.title("Label Wise Customer's Income", fontsize=20, color="#8D5F8C")
plt.xlabel(xlabel="Label", fontsize=15)
plt.ylabel(ylabel="Annual Income (k$)",fontsize=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='Label', y='Spending Score (1-100)', data=df, palette="pastel")
plt.title("Label Wise Spending Score", fontsize=20, color="#FDAAAA")
plt.xlabel(xlabel="Label", fontsize=15)
plt.ylabel(ylabel="Spending Score",fontsize=15)
plt.tight_layout()
plt.show()

# Getting the CustomerId for each group
cust1 = df[df.Label==0]
print("The number of customers in 1st group = ", len(cust1))
print("The Customer Id are - ", cust1.CustomerID.values)
print("============================================================================================\n")

cust2 = df[df.Label==1]
print("The number of customers in 2nd group = ", len(cust2))
print("The Customer Id are - ", cust2.CustomerID.values)
print("============================================================================================\n")

cust3 = df[df.Label==2]
print("The number of customers in 3rd group = ", len(cust3))
print("The Customer Id are - ", cust3.CustomerID.values)
print("============================================================================================\n")

cust4 = df[df.Label==3]
print("The number of customers in 4th group = ", len(cust4))
print("The Customer Id are - ", cust4.CustomerID.values)
print("============================================================================================\n")

cust5 = df[df.Label==4]
print("The number of customers in 5th group = ", len(cust5))
print("The Customer Id are - ", cust5.CustomerID.values)
print("============================================================================================\n")


#3D Plot as we did the clustering on the basis of 3 input features
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Age[df.Label == 0], df["Annual Income (k$)"][df.Label == 0], df["Spending Score (1-100)"][df.Label == 0], c='purple', s=60)
ax.scatter(df.Age[df.Label == 1], df["Annual Income (k$)"][df.Label == 1], df["Spending Score (1-100)"][df.Label == 1], c='red', s=60)
ax.scatter(df.Age[df.Label == 2], df["Annual Income (k$)"][df.Label == 2], df["Spending Score (1-100)"][df.Label == 2], c='blue', s=60)
ax.scatter(df.Age[df.Label == 3], df["Annual Income (k$)"][df.Label == 3], df["Spending Score (1-100)"][df.Label == 3], c='green', s=60)
ax.scatter(df.Age[df.Label == 4], df["Annual Income (k$)"][df.Label == 4], df["Spending Score (1-100)"][df.Label == 4], c='yellow', s=60)
ax.view_init(35, 135)
plt.title("3D view of the data distribution", fontsize=20, color="green")
plt.xlabel("Age", fontsize=15)
plt.ylabel("Annual Income (k$)", fontsize=15)
ax.set_zlabel('Spending Score (1-100)', fontsize=15)
plt.tight_layout()
plt.show()

cust1 = df[df.Label==0]
print("The number of customers in 1st group = ", len(cust1))
print("The Customer Id are - ", cust1.CustomerID.values)
print("============================================================================================\n")

cust2 = df[df.Label==1]
print("The number of customers in 2nd group = ", len(cust2))
print("The Customer Id are - ", cust2.CustomerID.values)
print("============================================================================================\n")

cust3 = df[df.Label==2]
print("The number of customers in 3rd group = ", len(cust3))
print("The Customer Id are - ", cust3.CustomerID.values)
print("============================================================================================\n")

cust4 = df[df.Label==3]
print("The number of customers in 4th group = ", len(cust4))
print("The Customer Id are - ", cust4.CustomerID.values)
print("============================================================================================\n")

cust5 = df[df.Label==4]
print("The number of customers in 5th group = ", len(cust5))
print("The Customer Id are - ", cust5.CustomerID.values)
print("============================================================================================\n")


# ==================================================================
# 2. VERI YUKLEME VE ON ISLEME (EDA ONCESI TEMIZLIK)


# Sutun isimlerini duzeltme
print("--- Veri Yüklendi, Gender Düzeltildi, Aykýrý Deðerler Temizlendi ---")
numeric_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

numeric_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
sns.pairplot(
    data=df,
    vars=numeric_features,
    hue="Gender", 
    diag_kind="kde",
    palette={'Male':'#1f77b4', 'Female':'#ff7f0e'} 
) 


plt.suptitle("Yas - Gelir - Harcama Pairplot (Cinsiyete Gore Renkli)", y=1.02, fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # ASCII uyumlu metin kullanildi
plt.show()


df_raw = pd.read_csv(r"C:\Users\Lenovo\Desktop\Musteri_Segmentasyon\Mall_Customers.csv")
df_raw.rename(columns={"Genre":"Gender"}, inplace=True)
df_clean = df_raw.copy()

# Uç değer (Outlier Capping) işlemi
Q1 = df_clean['Annual Income (k$)'].quantile(0.25)
Q3 = df_clean['Annual Income (k$)'].quantile(0.75)
IQR = Q3 - Q1
ust_sinir = Q3 + 1.5 * IQR
df_clean.loc[df_clean['Annual Income (k$)'] > ust_sinir, 'Annual Income (k$)'] = ust_sinir

print("--- Veri Yüklendi ve Uç Değer Düzeltmesi (df_clean) Tamamlandı. ---")

# ==================================================================
# 2. VERİ ÖLÇEKLENDİRME VE KÜMELEME İÇİN HAZIRLIK
# ==================================================================
features = ["Annual Income (k$)","Spending Score (1-100)"]
scaler = StandardScaler()

# Ham ve Temiz Veri Ölçeklendirme
X_raw = df_raw[features]
X_raw_scaled_df = pd.DataFrame(scaler.fit_transform(X_raw), columns=features)

X_clean = df_clean[features]
X_clean_scaled = scaler.fit_transform(X_clean)
X_cluster = pd.DataFrame(X_clean_scaled, columns=features) # Temiz veri kümeleme matrisi

print("--- Veri Ölçeklendirme Tamamlandı. ---")

# 3. KÜMELEME MODELLERİNİ EĞİTME (Temiz Veri Üzerinden)

K = 5
algorithms = {
    'K-Means': KMeans(n_clusters=K, random_state=42, n_init=10),
    'Hiyerarşik': AgglomerativeClustering(n_clusters=K, metric='euclidean', linkage='ward'),
    'Affinity Prop.': AffinityPropagation(damping=0.9, random_state=42, max_iter=200, convergence_iter=15)
}

# Modelleri eğitme ve etiketleri df_clean'e ekleme
df_clean["KMeans_Label"] = algorithms['K-Means'].fit_predict(X_cluster)
df_clean["HC_Label"] = algorithms['Hiyerarşik'].fit_predict(X_cluster)
df_clean["AP_Label"] = algorithms['Affinity Prop.'].fit_predict(X_cluster)

print("--- Tüm Kümeleme Modelleri Eğitildi. ---")

# ==================================================================
# 4. GÖRSELLEŞTİRMELER 

### 4.1. Korelasyon Matrisi 
df_encoded = pd.get_dummies(df_clean, columns=['Gender'], drop_first=True)
df_corr_final = df_encoded.drop(columns=['CustomerID', 'KMeans_Label', 'HC_Label', 'AP_Label'], errors='ignore')

plt.figure(figsize=(9, 8))
sns.heatmap(df_corr_final.corr(numeric_only=True), annot=True, cmap='Blues', fmt=".2f", linewidths=0.5, linecolor='white')
plt.title('Temiz Veri Korelasyon Matrisi ', fontsize=16)
plt.tight_layout()
plt.show() 

### 4.2. Hiyerarşik Kümeleme Dendrogramı 
linked = linkage(X_cluster, method='ward', metric='euclidean')
plt.figure(figsize=(15, 7))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True,
           color_threshold=linked[-5, 2] 
          )
plt.axhline(y=10.0, color='r', linestyle='--') 
plt.title('Hiyerarşik Kümeleme Dendrogramı', fontsize=18)
plt.ylabel('Mesafe (Distance)')
plt.xlabel('Müşteriler (gruplandırılmış)', fontsize=12)
plt.tight_layout()
plt.show() 

# ------------------------------------------------------------------
# 4.2.1 HIYERARŞİK KÜMELEME: HAM (RAW) VERİ İÇİN DENDROGRAM
# ------------------------------------------------------------------


features_for_dendrogram = ["Annual Income (k$)","Spending Score (1-100)"]
scaler_raw = StandardScaler()
# df_raw, kodun ust kisminda tanimli olmasi gerekir.
X_raw_cluster = pd.DataFrame(scaler_raw.fit_transform(df_raw[features_for_dendrogram]), columns=features_for_dendrogram)

# 3. Baglanti Hesabi (Agglomerative/Ward)
linked_raw = linkage(X_raw_cluster, method="ward")

# 4. Dendrogram Cizimi
plt.figure(figsize=(12, 7))
dendrogram(linked_raw,
           orientation='top',
           show_leaf_counts=False,
           # 5 kume icin esik degeri
           color_threshold=linked_raw[-5, 2] 
           )
plt.title('Dendrogram: HAM VERI (Uc Degerlerin Etkisi)', fontsize=18, color='red')
plt.ylabel('Mesafe (Distance)')
plt.xlabel('Musteriler', fontsize=12)
plt.show()
print("\n--- Ham Veri Dendrogrami Tamamlandi. ---")

### 4.3. Küme Dağılım Grafikleri

# K-Means Dağılım
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x="Annual Income (k$)", y="Spending Score (1-100)", 
                hue="KMeans_Label", s=60, palette="deep")
plt.title("K-Means (K=5) Dağılımı", fontsize=18)
plt.tight_layout()
plt.show()

# Hiyerarşik Kümeleme Dağılım
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x="Annual Income (k$)", y="Spending Score (1-100)", 
                hue="HC_Label", s=60, palette="deep")
plt.title("Hiyerarşik Kümeleme (K=5) Dağılımı", fontsize=18)
plt.tight_layout()
plt.show()

# Affinity Propagation Dağılım
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x="Annual Income (k$)", y="Spending Score (1-100)", 
                hue="AP_Label", s=60, palette="tab20")
plt.title(f"Affinity Propagation Dağılımı (K={len(df_clean['AP_Label'].unique())})", fontsize=18)
plt.tight_layout()
plt.show()

features_2d = ["Annual Income (k$)","Spending Score (1-100)"]
features_3d = ["Age","Annual Income (k$)","Spending Score (1-100)"]
features_4d = ["Gender","Age","Annual Income (k$)","Spending Score (1-100)"]
K = 5 # Sabit kume sayisi

print("\n--- AGGLOMERATIVE CLUSTERING: Boyut Karsilastirmasi ---")

for feats in [features_2d, features_3d, features_4d]:
    # 1. Veri Hazirligi ve Olceklendirme
    label_encoder = LabelEncoder()


    df_clean['Gender'] = label_encoder.fit_transform(df_clean['Gender'])
    scaler = StandardScaler() 
    X = df_clean[feats]
    X_scaled = scaler.fit_transform(X)

    # 2. Modeli Egitme
    hc = AgglomerativeClustering(n_clusters=K, metric='euclidean', linkage='ward')
    y_hc = hc.fit_predict(X_scaled)

    # 3. Metrik Hesaplama (Calinski Harabasz kaldirildi)
    sil = silhouette_score(X_scaled, y_hc)
    db = davies_bouldin_score(X_scaled, y_hc)
    # ch KALDIRILDI

    # 4. Sonuclari Yazdirma
    print(f"\n Ozellik Seti : {feats}")
    print(f"Silhouette Score: {sil:.4f}")
    print(f"Davies Bouldin Score: {db:.4f}")
    # Calinski Harabasz Score yazdirilmayacak
    print("-------------")

print("\n--- HC Boyut Karsilastirmasi Tamamlandi. ---")

# 5. SKOR HESAPLAMA VE KIYASLAMA 

# K-Means Ham Veri Skoru
kmeans_raw_labels = algorithms['K-Means'].fit_predict(X_raw_scaled_df)
raw_kmeans_scores = {
    'Silhouette': silhouette_score(X_raw_scaled_df, kmeans_raw_labels),
    'Davies-Bouldin': davies_bouldin_score(X_raw_scaled_df, kmeans_raw_labels)
}

# Temiz Veri Skorları
clean_scores = {}
algo_names = list(algorithms.keys())
label_cols = ['KMeans_Label', 'HC_Label', 'AP_Label'] 

for algo_name, label_col in zip(algo_names, label_cols):
    labels = df_clean[label_col]
    
    if len(np.unique(labels)) > 1:
        sil_score = silhouette_score(X_cluster, labels)
        db_score = davies_bouldin_score(X_cluster, labels)
    else:
        sil_score = 0.0
        db_score = 100.0

    clean_scores[algo_name] = {'Silhouette': sil_score, 'Davies-Bouldin': db_score}

print("\n--- Tüm Kümeleme Algoritmalarının Skorları Hesaplandı. ---")


# 6. SONUÇ TABLOLARI VE KIYASLAMA GRAFİKLERİ
# ==================================================================

# Tablo 1: Temiz Veri Üzerinde Algoritma Kıyaslaması
df_clean_summary = pd.DataFrame(clean_scores).T
print("\n*** 1. TABLO: TEMİZ VERİ ÜZERİNDE TÜM ALGORİTMALARIN KIYASLANMASI ***")
print(df_clean_summary.to_markdown(floatfmt=".4f"))

# Tablo 2: K-Means (Ham vs. Temiz) Kıyaslaması
df_comparison_km = pd.DataFrame({
    'K-Means (Ham)': raw_kmeans_scores,
    'K-Means (Temiz)': clean_scores['K-Means']
}).T
print("\n*** 2. TABLO: K-MEANS PERFORMANS KIYASLAMASI (HAM vs. TEMİZ VERİ) ***")
print(df_comparison_km.to_markdown(floatfmt=".4f"))


# Görsel Kıyaslama (Temiz Veri Üzerinde Tüm Algoritmalar)
modeller_kume = algo_names
skorlar_sil_values = df_clean_summary['Silhouette'].tolist()
skorlar_db_values = df_clean_summary['Davies-Bouldin'].tolist()

plt.figure(figsize=(14, 6))
plt.suptitle('Temiz Veri Üzerinde Tüm Algoritmaların Kıyaslanması', fontsize=16)

# Silhouette Skoru
plt.subplot(1, 2, 1)
sns.barplot(x=modeller_kume, y=skorlar_sil_values, palette='viridis')
plt.title('Silhouette Skor Karşılaştırması (Yüksek Daha İyi)')
plt.ylabel('Silhouette Skoru')

# Davies-Bouldin Skoru
plt.subplot(1, 2, 2)
sns.barplot(x=modeller_kume, y=skorlar_db_values, palette='plasma')
plt.title('Davies-Bouldin Skor Karşılaştırması (Düşük Daha İyi)')
plt.ylabel('Davies-Bouldin Skoru')

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()
