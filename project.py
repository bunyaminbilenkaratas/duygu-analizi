from zemberek import TurkishMorphology, TurkishSentenceNormalizer

def turkce_metni_duzelt(metin):
    morphology = TurkishMorphology.create_with_defaults()
    normalizer = TurkishSentenceNormalizer(morphology)
    duzeltilmis_metin = normalizer.normalize(metin)
    return duzeltilmis_metin

def kelime_kok_neg(metin):
    morphology = TurkishMorphology.create_with_defaults()
    kelimeler = metin.split()
    yenimetin = ""
    for kelime in kelimeler:
        results = morphology.analyze(kelime),
        kok=""
        for result in results:
            if result.is_correct()== True:
                kok = result.analysis_results[0].get_stem()
                morphemes = result.analysis_results[0].get_morphemes()
                for morpheme in morphemes:
                    if morpheme.name == "Negative":
                        kok=kok+"NEG"
            else:
                kok=kelime
            yenimetin += kok + " "
    yenimetin = yenimetin.strip()
    return yenimetin

def detect_emoticons(metin):
    pozitif_emojiler = [':‑)', ':)', ':-]', ':]', ':->', ':>', '8-)', '8)', ':-}', ':}', ':o)', ':c)', ':^)', '=]', '=)', ':‑D', ':D', '8‑D', '8D', '=D', '=3', 'B^D', 'c:', 'C:', 'x‑D', 'xD', 'X‑D', 'XD', ':-))', ':))', ":'‑)", ":'(", ':=(', ':]', ';‑)', ';)', '*‑)', '*)', ';‑]', ';]', ';^)', ';>', ':‑,', ';D', ';3', ':‑P', ':P', 'X‑P', 'XP', 'x‑p', 'xp', ':‑p', ':p', ':‑Þ', ':Þ', ':‑þ', ':þ', ':‑b', ':b', 'd:', '=p', '>:P']
    negatif_emojiler = [':-(', ':(', ':-[', ':[', ':-<', ':<', '8-(', '8(', ':-{', ':{', ':o(', ':c(', ':^(', '=/', '=<', ':-/', ':/', ':-\\', ':\\', ':-|', ':|', ':‑c', ':c', ':‑<', ':<', ':‑[', ':[', ':-||', ':{', ':@', ':(', ';(', ":'‑(", ":'(", ':=(', ':(', ':=(', '>:(', '>:[', 'D‑\':', 'D:<', 'D:', 'D8', 'D;', 'D=', 'DX']

    for emoji in pozitif_emojiler:
        metin = metin.replace(emoji, 'POSEMOTİON')
    
    for emoji in negatif_emojiler:
        metin = metin.replace(emoji, 'NEGEMOTİON')

    return metin

def noktalama_isaretleri_kaldir(metin):
    noktalama_isaretleri = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

    for noktalama in noktalama_isaretleri:
        metin = metin.replace(noktalama, ' ')
        
    return metin

def fazla_bosluklari_kaldir(metin):
    kelimeler = metin.split()
    metin = ' '.join(kelimeler)
    
    return metin

def onisleme(metin):
    metin = turkce_metni_duzelt(metin)
    metin = kelime_kok_neg(metin)
    metin = detect_emoticons(metin)
    metin = noktalama_isaretleri_kaldir(metin)
    metin = fazla_bosluklari_kaldir(metin)
    return metin

def dosyalari_onisle(pozitif_yorumlar, negatif_yorumlar):
    pozitif_girdi_dosya = open('pozitif_yorumlar.txt', 'r', encoding='utf-8')
    pozitif_cikti_dosya = open('pozitif_yorumlar_onislenmis.txt', 'a')
    
    negatif_girdi_dosya = open('negatif_yorumlar.txt', 'r', encoding='utf-8')
    negatif_cikti_dosya = open('negatif_yorumlar_onislenmis.txt', 'a')
    
    for metin in pozitif_girdi_dosya:
        metin = onisleme(metin)
        pozitif_cikti_dosya.write(metin)
        
    for metin in negatif_girdi_dosya:
        metin = onisleme(metin)
        negatif_cikti_dosya.write(metin)


onislenmis_pozitif_veriseti = open('pozitif_yorumlar_onislenmis.txt', 'r', encoding='utf-8')
onislenmis_negatif_veriseti = open('negatif_yorumlar_onislenmis.txt', 'r', encoding='utf-8')

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
onislenmis_pozitif_veriseti.seek(0)
onislenmis_negatif_veriseti.seek(0)
pozitif_veriler = onislenmis_pozitif_veriseti.readlines()
negatif_veriler = onislenmis_negatif_veriseti.readlines()


for i, (train_index, test_index) in enumerate(kf.split(pozitif_veriler)):
    fold_klasor_adi = f'fold{i+1}'
    os.makedirs(fold_klasor_adi, exist_ok=True)
    
    test_pozitif = [pozitif_veriler[i] for i in test_index]
    egitim_pozitif = [pozitif_veriler[i] for i in train_index]
    
    test_negatif = [negatif_veriler[i] for i in test_index]
    egitim_negatif = [negatif_veriler[i] for i in train_index]

    with open(os.path.join(fold_klasor_adi, 'test_pozitif.txt'), 'w') as f:
        for satir in test_pozitif:
            f.write(satir)
    
    with open(os.path.join(fold_klasor_adi, 'egitim_pozitif.txt'), 'w') as f:
        for satir in egitim_pozitif:
            f.write(satir)
    
    with open(os.path.join(fold_klasor_adi, 'test_negatif.txt'), 'w') as f:
        for satir in test_negatif:
            f.write(satir)
    
    with open(os.path.join(fold_klasor_adi, 'egitim_negatif.txt'), 'w') as f:
        for satir in egitim_negatif:
            f.write(satir)


def oznitelik_hesapla(fold_klasoru):
    oznitelikler = []
    
    egitim_pozitif_yolu = os.path.join(fold_klasoru, 'egitim_pozitif.txt')
    egitim_negatif_yolu = os.path.join(fold_klasoru, 'egitim_negatif.txt')
    
    tum_veri = []
    # Eğitim pozitif verilerini oku ve değişkene ekle
    with open(egitim_pozitif_yolu, 'r') as f:
        f.seek(0)
        egitim_pozitif = f.readlines()
        tum_veri.extend(egitim_pozitif)

    # Eğitim negatif verilerini oku ve değişkene ekle
    with open(egitim_negatif_yolu, 'r') as f:
        f.seek(0)
        egitim_negatif = f.readlines()
        tum_veri.extend(egitim_negatif)

    for satir in tum_veri:
        kelimeler = satir.split()
        for kelime in kelimeler:
            if kelime not in oznitelikler:
                oznitelikler.append(kelime)
                    
    return oznitelikler

def tfidf_hesapla(fold_klasoru, oznitelikler, csv_oneki=''):
    egitim_pozitif_yolu = os.path.join(fold_klasoru, 'egitim_pozitif.txt')
    egitim_negatif_yolu = os.path.join(fold_klasoru, 'egitim_negatif.txt')
    
    # Boş bir değişken oluştur ve içine birleştirilen verileri aktar.
    tum_veri = []
    # Eğitim pozitif verilerini oku ve değişkene ekle
    with open(egitim_pozitif_yolu, 'r') as f:
        f.seek(0)
        egitim_pozitif = f.readlines()
        tum_veri.extend(egitim_pozitif)

    # Eğitim negatif verilerini oku ve değişkene ekle
    with open(egitim_negatif_yolu, 'r') as f:
        f.seek(0)
        egitim_negatif = f.readlines()
        tum_veri.extend(egitim_negatif)
    
    
    training_df_vector = np.zeros(len(oznitelikler))
    training_idf_vector = np.zeros(len(oznitelikler))
    
    for i, oznitelik in enumerate(oznitelikler):
        for satir in tum_veri:
            satirdaki_kelimeler = satir.split()
            if oznitelik in satirdaki_kelimeler:
                training_df_vector[i] += 1
                continue
    
    training_documents = tum_veri
    
    training_idf_vector = np.log10(len(training_documents) / (training_df_vector))
    training_tf_matrix = np.zeros((len(training_documents), len(oznitelikler)))
    
    # Her bir döküman için TF vektörünü hesapla
    for i, document in enumerate(training_documents):
        # Dökümanı kelimelere ayır
        kelimeler = document.split()
        # Dökümandaki her bir kelime için TF değerini hesapla ve TF vektörüne ekle
        for kelime in kelimeler:
            if kelime in oznitelikler:
                # Kelimenin öznitelikler listesindeki indeksini bul
                indeks = oznitelikler.index(kelime)
                # TF matrisinde ilgili döküman ve öznitelik için sayısını bir artır
                training_tf_matrix[i, indeks] += 1
                
                
    
    training_idf_dataframe = pd.DataFrame(training_idf_vector, index=oznitelikler)
    
    E_li_indexler = ['E' + str(i+1) for i in range(len(training_documents))]
    training_tf_dataframe = pd.DataFrame(training_tf_matrix, index=E_li_indexler, columns=oznitelikler)
    
    training_tfidf_matrix = training_tf_matrix * training_idf_vector
    training_tfidf_dataframe = pd.DataFrame(training_tfidf_matrix, index=E_li_indexler, columns=oznitelikler)

    training_positive_documents = egitim_pozitif
    training_negative_documents = egitim_negatif
    
    training_classes = [1] * len(training_positive_documents) + [0] * len(training_negative_documents)
    training_classes_array = np.array(training_classes)
    data = np.hstack((training_tfidf_matrix, training_classes_array.reshape(-1, 1)))
    df = pd.DataFrame(data, columns=list(oznitelikler) + ['class'])
    df.to_csv(f"{fold_klasoru}/train{csv_oneki}.csv", index=False)
    
    
    test_pozitif_yolu = os.path.join(fold_klasoru, 'test_pozitif.txt')
    test_negatif_yolu = os.path.join(fold_klasoru, 'test_negatif.txt')
    
    # Boş bir değişken oluştur
    tum_veri = []

    # Test pozitif verilerini oku ve değişkene ekle
    with open(test_pozitif_yolu, 'r') as f:
        f.seek(0)
        test_pozitif = f.readlines()
        tum_veri.extend(test_pozitif)

    # Test negatif verilerini oku ve değişkene ekle
    with open(test_negatif_yolu, 'r') as f:
        f.seek(0)
        test_negatif = f.readlines()
        tum_veri.extend(test_negatif)
    
    
    test_idf_vector = training_idf_vector # Ödevde eğitim kümesindeki aynı idf ve özniteliklerin kullanılması istendiği için.
    test_idf_dataframe = training_idf_dataframe
    
    test_documents = tum_veri
    
    test_tf_matrix = np.zeros((len(test_documents), len(oznitelikler))) # Ödevde eğitim kümesindeki aynı özniteliklerin kullanılması istendiği için.
    
    # Her bir döküman için TF vektörünü hesapla
    for i, document in enumerate(test_documents):
        # Dökümanı kelimelere ayır
        kelimeler = document.split()
        # Dökümandaki her bir kelime için TF değerini hesapla ve TF vektörüne ekle
        for kelime in kelimeler:
            if kelime in oznitelikler:
                # Kelimenin öznitelikler listesindeki indeksini bul
                indeks = oznitelikler.index(kelime)
                # TF matrisinde ilgili döküman ve öznitelik için sayısını bir artır
                test_tf_matrix[i, indeks] += 1
                
                
    T_li_indexler = ['T' + str(i+1) for i in range(len(test_documents))]
    test_tf_dataframe = pd.DataFrame(test_tf_matrix, index=T_li_indexler, columns=oznitelikler)
    
    test_tfidf_matrix = test_tf_matrix * test_idf_vector
    test_tfidf_dataframe = pd.DataFrame(test_tfidf_matrix, index=T_li_indexler, columns=oznitelikler)

    test_positive_documents = test_pozitif
    test_negative_documents = test_negatif
    
    test_classes = [1] * len(test_positive_documents) + [0] * len(test_negative_documents)
    test_classes_array = np.array(test_classes)
    data = np.hstack((test_tfidf_matrix, test_classes_array.reshape(-1, 1)))
    df = pd.DataFrame(data, columns=list(oznitelikler) + ['class'])
    df.to_csv(f"{fold_klasoru}/test{csv_oneki}.csv", index=False)
   
    
            
fold_klasorleri = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
for fold_klasoru in fold_klasorleri:
    oznitelikler = oznitelik_hesapla(fold_klasoru)
    tfidf_hesapla(fold_klasoru, oznitelikler)






import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Klasörlerin isimlerini tanımlayın
folders = ["fold1", "fold2", "fold3", "fold4", "fold5"]

# Her bir sınıflandırıcı için ayrı ayrı çapraz doğrulama yapın
svm_acc_scores = []
svm_f1_scores = []
rf_acc_scores = []
rf_f1_scores = []
lr_acc_scores = []
lr_f1_scores = []
gb_acc_scores = []
gb_f1_scores = []

for folder in folders:
    # Veri setini yükleyin
    train_data = pd.read_csv(f"{folder}/train.csv")
    test_data = pd.read_csv(f"{folder}/test.csv")
    
    print("aşama1")
    print(folder)
    
    X_train = train_data.drop(columns=train_data.columns[-1])  # Son sütunu hariç al
    y_train = train_data[train_data.columns[-1]]   # Sadece son sütunu al
    X_test = train_data.drop(columns=test_data.columns[-1])  # Son sütunu hariç al
    y_test = train_data[test_data.columns[-1]]   # Sadece son sütunu al
    
    print("aşama2")
    print(folder)

    # Sınıflandırıcıları tanımlayın
    svm_clf = SVC()
    rf_clf = RandomForestClassifier()
    lr_clf = LogisticRegression()
    gb_clf = GradientBoostingClassifier()

    # Sınıflandırıcıları değerlendirin
    def evaluate_classifier(clf, X_train, y_train):
        acc_scores = cross_val_score(clf, X_train, y_train, scoring='accuracy')
        f1_scores = cross_val_score(clf, X_train, y_train, scoring='f1')
        return acc_scores, f1_scores
    
    print("aşama3")
    print(folder)

    svm_acc, svm_f1 = evaluate_classifier(svm_clf, X_train, y_train)
    
    print("aşama4")
    print(folder)
    
    
    rf_acc, rf_f1 = evaluate_classifier(rf_clf, X_train, y_train)
    
    print("aşama5")
    print(folder)
    
    
    lr_acc, lr_f1 = evaluate_classifier(lr_clf, X_train, y_train)
    
    print("aşama6")
    print(folder)
    
    
    gb_acc, gb_f1 = evaluate_classifier(gb_clf, X_train, y_train)
    
    print("aşama7")
    print(folder)
    

    # Sonuçları toplayın
    svm_acc_scores.extend(svm_acc)
    svm_f1_scores.extend(svm_f1)
    rf_acc_scores.extend(rf_acc)
    rf_f1_scores.extend(rf_f1)
    lr_acc_scores.extend(lr_acc)
    lr_f1_scores.extend(lr_f1)
    gb_acc_scores.extend(gb_acc)
    gb_f1_scores.extend(gb_f1)
    
    print("Fold:", folder)

    print("SVM Results:")
    print("Mean Accuracy:", svm_acc)
    print("Mean F1 Score:", svm_f1)
    print()
    print("Random Forest Results:")
    print("Mean Accuracy:", rf_acc)
    print("Mean F1 Score:", rf_f1)
    print()
    print("Logistic Regression Results:")
    print("Mean Accuracy:", lr_acc)
    print("Mean F1 Score:", lr_f1)
    print()
    print("Gradient Boosting Results:")
    print("Mean Accuracy:", gb_acc)
    print("Mean F1 Score:", gb_f1)
    
    print("aşama8")
    print(folder)
    

# Ortalama değerleri hesaplayın
svm_mean_acc = sum(svm_acc_scores) / len(svm_acc_scores)
svm_mean_f1 = sum(svm_f1_scores) / len(svm_f1_scores)
rf_mean_acc = sum(rf_acc_scores) / len(rf_acc_scores)
rf_mean_f1 = sum(rf_f1_scores) / len(rf_f1_scores)
lr_mean_acc = sum(lr_acc_scores) / len(lr_acc_scores)
lr_mean_f1 = sum(lr_f1_scores) / len(lr_f1_scores)
gb_mean_acc = sum(gb_acc_scores) / len(gb_acc_scores)
gb_mean_f1 = sum(gb_f1_scores) / len(gb_f1_scores)

print("aşama5")
print(folder)

# Sonuçları yazdırın

print("Öznitelik çıkartılmadan sonuçlar:")

print("SVM Results:")
print("Mean Accuracy:", svm_mean_acc)
print("Mean F1 Score:", svm_mean_f1)
print()
print("Random Forest Results:")
print("Mean Accuracy:", rf_mean_acc)
print("Mean F1 Score:", rf_mean_f1)
print()
print("Logistic Regression Results:")
print("Mean Accuracy:", lr_mean_acc)
print("Mean F1 Score:", lr_mean_f1)
print()
print("Gradient Boosting Results:")
print("Mean Accuracy:", gb_mean_acc)
print("Mean F1 Score:", gb_mean_f1)


print("aşama6")
print(folder)





# Klasörlerin isimlerini tanımlayın
fold_klasorleri = ["fold1", "fold2", "fold3", "fold4", "fold5"]

import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import SelectKBest, chi2

for fold_klasoru in fold_klasorleri:
    # train.csv dosyasını yükleyin
    train_data = pd.read_csv(f"{fold_klasoru}/train.csv")
    test_data = pd.read_csv(f"{fold_klasoru}/test.csv")
    
    # Özellikler (features) ve etiketler (labels) ayırın
    #X_train = train_data.drop("target_column", axis=1)  # target_column doğru hedef sütunun adı ile değiştirilmelidir
    #y_train = train_data["target_column"]  # target_column doğru hedef sütunun adı ile değiştirilmelidir
    
    X_train = train_data.drop(columns=train_data.columns[-1])  # Son sütunu hariç al
    y_train = train_data[train_data.columns[-1]]   # Sadece son sütunu al
    X_test = train_data.drop(columns=test_data.columns[-1])  # Son sütunu hariç al
    y_test = train_data[test_data.columns[-1]]   # Sadece son sütunu al
    
    # Seçilecek öznitelik sayıları
    oznitelik_sayilari = [250, 500, 1000, 2500, 5000]
    
    # Chi-kare algoritması ile öznitelik seçimi
    for oznitelik_sayisi in oznitelik_sayilari:
        selector = SelectKBest(score_func=chi2, k=oznitelik_sayisi)
        X_new = selector.fit_transform(X_train, y_train)
        secilen_oznitelik_indisleri = selector.get_support(indices=True)
        secilen_oznitelik_isimleri = X_train.columns[secilen_oznitelik_indisleri].tolist()
        tfidf_hesapla(fold_klasoru, secilen_oznitelik_isimleri, oznitelik_sayisi)
    
    

import pandas as pd
import numpy as np
import os

# Her bir fold klasörünü gez ve pozitif ve veriyleri birleştir
fold_klasorleri = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
feature_counts = [250, 500, 1000, 2500, 5000]


for fold_klasoru in fold_klasorleri:
    egitim_pozitif_yolu = os.path.join(fold_klasoru, 'egitim_pozitif.txt')
    egitim_negatif_yolu = os.path.join(fold_klasoru, 'egitim_negatif.txt')
    
    # Boş bir değişken oluştur ve içine birleştirilen verileri aktar.
    tum_veri = []
    # Eğitim pozitif verilerini oku ve değişkene ekle
    with open(egitim_pozitif_yolu, 'r') as f:
        f.seek(0)
        egitim_pozitif = f.readlines()
        tum_veri.extend(egitim_pozitif)

    # Eğitim negatif verilerini oku ve değişkene ekle
    with open(egitim_negatif_yolu, 'r') as f:
        f.seek(0)
        egitim_negatif = f.readlines()
        tum_veri.extend(egitim_negatif)
        
        
    for feature_count in feature_counts:
        secilmis_csv = pd.read_csv(f"{folder}/train{feature_count}.csv")
        oznitelikler = secilmis_csv.columns.tolist()
        
        training_df_vector = np.zeros(len(oznitelikler))
        training_idf_vector = np.zeros(len(oznitelikler))
        
        for i, oznitelik in enumerate(oznitelikler):
            for satir in tum_veri:
                satirdaki_kelimeler = satir.split()
                if oznitelik in satirdaki_kelimeler:
                    training_df_vector[i] += 1
                    continue
        
    
    training_documents = tum_veri
    
    training_idf_vector = np.log10(len(training_documents) / (training_df_vector))
    training_tf_matrix = np.zeros((len(training_documents), len(oznitelikler)))
    
    # Her bir döküman için TF vektörünü hesapla
    for i, document in enumerate(training_documents):
        # Dökümanı kelimelere ayır
        kelimeler = document.split()
        # Dökümandaki her bir kelime için TF değerini hesapla ve TF vektörüne ekle
        for kelime in kelimeler:
            if kelime in oznitelikler:
                # Kelimenin öznitelikler listesindeki indeksini bul
                indeks = oznitelikler.index(kelime)
                # TF matrisinde ilgili döküman ve öznitelik için sayısını bir artır
                training_tf_matrix[i, indeks] += 1
    
    
    training_idf_dataframe = pd.DataFrame(training_idf_vector, index=oznitelikler)
    
    E_li_indexler = ['E' + str(i+1) for i in range(len(training_documents))]
    training_tf_dataframe = pd.DataFrame(training_tf_matrix, index=E_li_indexler, columns=oznitelikler)
    
    training_tfidf_matrix = training_tf_matrix * training_idf_vector
    training_tfidf_dataframe = pd.DataFrame(training_tfidf_matrix, index=E_li_indexler, columns=oznitelikler)

    training_positive_documents = egitim_pozitif
    training_negative_documents = egitim_negatif
    
    training_classes = [1] * len(training_positive_documents) + [0] * len(training_negative_documents)
    training_classes_array = np.array(training_classes)
    data = np.hstack((training_tfidf_matrix, training_classes_array.reshape(-1, 1)))
    df = pd.DataFrame(data, columns=list(oznitelikler) + ['class'])
    df.to_csv(fold_klasoru + '/' + 'train{feature_count}.csv', index=False)











import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Klasörlerin isimlerini tanımlayın
folders = ["fold1", "fold2", "fold3", "fold4", "fold5"]
feature_counts = [250, 500, 1000, 2500, 5000]

for feature_count in feature_counts:
    # Her bir sınıflandırıcı için ayrı ayrı çapraz doğrulama yapın
    svm_acc_scores = []
    svm_f1_scores = []
    rf_acc_scores = []
    rf_f1_scores = []
    lr_acc_scores = []
    lr_f1_scores = []
    gb_acc_scores = []
    gb_f1_scores = []
    
    for folder in folders:
        # Veri setini yükleyin
        train_data = pd.read_csv(f"{folder}/train{feature_count}.csv")
       # df.to_csv(f"{fold_klasoru}/train{feature_count}.csv", index=False)
        test_data = pd.read_csv(f"{folder}/test{feature_count}.csv")
        
        print("aşama1")
        print(folder)
    
        
        X_train = train_data.drop(columns=train_data.columns[-1])  # Son sütunu hariç al
        y_train = train_data[train_data.columns[-1]]   # Sadece son sütunu al
        X_test = train_data.drop(columns=test_data.columns[-1])  # Son sütunu hariç al
        y_test = train_data[test_data.columns[-1]]   # Sadece son sütunu al
        
        print("aşama2")
        print(folder)
    
        # Sınıflandırıcıları tanımlayın
        svm_clf = SVC()
        rf_clf = RandomForestClassifier()
        lr_clf = LogisticRegression()
        gb_clf = GradientBoostingClassifier()
    
        # Sınıflandırıcıları değerlendirin
        def evaluate_classifier(clf, X_train, y_train):
            acc_scores = cross_val_score(clf, X_train, y_train, scoring='accuracy')
            f1_scores = cross_val_score(clf, X_train, y_train, scoring='f1')
            return acc_scores, f1_scores
        
        print("aşama3")
        print(folder)
    
        svm_acc, svm_f1 = evaluate_classifier(svm_clf, X_train, y_train)
        
        print("aşama4")
        print(folder)
        
        
        rf_acc, rf_f1 = evaluate_classifier(rf_clf, X_train, y_train)
        
        print("aşama5")
        print(folder)
        
        
        lr_acc, lr_f1 = evaluate_classifier(lr_clf, X_train, y_train)
        
        print("aşama6")
        print(folder)
        
        
        gb_acc, gb_f1 = evaluate_classifier(gb_clf, X_train, y_train)
        
        print("aşama7")
        print(folder)
        
    
        # Sonuçları toplayın
        svm_acc_scores.extend(svm_acc)
        svm_f1_scores.extend(svm_f1)
        rf_acc_scores.extend(rf_acc)
        rf_f1_scores.extend(rf_f1)
        lr_acc_scores.extend(lr_acc)
        lr_f1_scores.extend(lr_f1)
        gb_acc_scores.extend(gb_acc)
        gb_f1_scores.extend(gb_f1)
        
        print("feature:", folder)
    
        print("SVM Results:")
        print("Menonan Accuracy:", svm_acc)
        print("Mean F1 Score:", svm_f1)
        print()
        print("Random Forest Results:")
        print("Mean Accuracy:", rf_acc)
        print("Mean F1 Score:", rf_f1)
        print()
        print("Logistic Regression Results:")
        print("Mean Accuracy:", lr_acc)
        print("Mean F1 Score:", lr_f1)
        print()
        print("Gradient Boosting Results:")
        print("Mean Accuracy:", gb_acc)
        print("Mean F1 Score:", gb_f1)
        
        print("aşama8")
        print(folder)
    

    # Her bir feature countun ortalama değerleri
    svm_mean_acc = sum(svm_acc_scores) / len(svm_acc_scores)
    svm_mean_f1 = sum(svm_f1_scores) / len(svm_f1_scores)
    rf_mean_acc = sum(rf_acc_scores) / len(rf_acc_scores)
    rf_mean_f1 = sum(rf_f1_scores) / len(rf_f1_scores)
    lr_mean_acc = sum(lr_acc_scores) / len(lr_acc_scores)
    lr_mean_f1 = sum(lr_f1_scores) / len(lr_f1_scores)
    gb_mean_acc = sum(gb_acc_scores) / len(gb_acc_scores)
    gb_mean_f1 = sum(gb_f1_scores) / len(gb_f1_scores)
    
    print("aşama5")
    print(folder)
    
    # Sonuçları yazdırın
    
    print("feature {feature_count} için ortalama sonuçlar:")
    
    print("SVM Results:")
    print("Mean Accuracy:", svm_mean_acc)
    print("Mean F1 Score:", svm_mean_f1)
    print()
    print("Random Forest Results:")
    print("Mean Accuracy:", rf_mean_acc)
    print("Mean F1 Score:", rf_mean_f1)
    print()
    print("Logistic Regression Results:")
    print("Mean Accuracy:", lr_mean_acc)
    print("Mean F1 Score:", lr_mean_f1)
    print()
    print("Gradient Boosting Results:")
    print("Mean Accuracy:", gb_mean_acc)
    print("Mean F1 Score:", gb_mean_f1)


 
    
    
