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
    negatif_cikti_dosya = open('negatif_yorumlar_onislenmis.txt.txt', 'a')
    
    for metin in pozitif_girdi_dosya:
        metin = onisleme(metin)
        pozitif_cikti_dosya.write(metin)
        
    for metin in negatif_girdi_dosya:
        metin = onisleme(metin)
        negatif_cikti_dosya.write(metin)