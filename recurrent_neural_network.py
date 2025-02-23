"""
Solve Classification (Sentiment Analysis in NLP) with RNN (Deep Lerning based Language Model)

duygu analizi -> bir cumlenin etiketlenmesi (positive ve negative)
restaurant yorumlari degerlendirme
"""

# import libraries
import pandas as pd
import numpy as np

from gensim.models import Word2Vec # metin temsili

from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# create dataset

# Not: verisetinde positif ve negatif verileri dengesiz olması sebebiyle model negatif sınıfı daha iyi öğrendi. Sınıflandırma düzenlendi.
data = {
    "text": [
        "Yemekler çok güzeldi",  # Positive
        # "Servis yavaş ve garsonlar ilgisizdi",  # Negative
        "Harika bir deneyim yaşadım, tekrar geleceğim",  # Positive
        # "Yemekler çok soğuktu, hiç beğenmedim",  # Negative
        "Mekan çok güzel ve atmosfer harikaydı",  # Positive
        # "Çalışanlar güler yüzlüydü ama yemekler beklediğim gibi değildi",  # Negative
        "Fiyatlar çok uygundu ve porsiyonlar büyüktü",  # Positive
        # "Bu kadar kötü yemek olamaz, tam bir hayal kırıklığı",  # Negative
        "Lezzetli yemekler ve hızlı servis, çok memnun kaldım",  # Positive
        # "Burası çok pahalı, verdiğim paraya değmedi",  # Negative
        "Restoranın manzarası çok güzeldi, yemekleri de harikaydı",  # Positive
        # "Yemekler çok geç geldi, biraz daha hızlı olabilirlerdi",  # Negative
        "Çalışanlar çok yardımcıydı, çok memnun kaldım",  # Positive
        "Tatlıları gerçekten harikaydı, kesinlikle tavsiye ederim",  # Positive
        # "Yemekler kötüydü, bir daha gelmem",  # Negative
        "Yemekler mükemmeldi, porsiyonlar da çok büyüktü",  # Positive
        "Burası gerçekten çok pahalı, ancak yemekler mükemmeldi",  # Positive
        "Fiyatlar çok yüksekti, ancak yemekler lezzetliydi",  # Positive
        "Atmosfer harikaydı, yemekler de gayet lezzetliydi",  # Positive
        # "Çok kalabalıktı, servis biraz daha iyi olabilirdi",  # Negative
        "Mekanın iç tasarımı çok hoştu, yemekler de harikaydı",  # Positive
        "Servis çok hızlıydı, kesinlikle tekrar gelirim",  # Positive
        "Yemekler gerçekten çok lezzetliydi, tekrar geleceğim",  # Positive
        # "Hizmet çok kötüydü, garsonlar ilgisizdi",  # Negative
        # "Yemekler soğuktu ve servisi beklemek zorundaydık",  # Negative
        # "Yemekler çok lezzetliydi ama servis biraz yavaştı",  # Negative
        # "Garsonlar çok nazikti, fakat yemekler beklediğim gibi değildi",  # Negative
        "Restoranın dekorasyonu çok hoştu, yemekler de lezzetliydi",  # Positive
        "Birçok seçenek vardı ve yemekler çok lezzetliydi",  # Positive
        "Çalışanlar çok güler yüzlüydü, yemekler harikaydı",  # Positive
        "Yemekler çok lezzetliydi, mekan da çok şık",  # Positive
        "Atmosfer çok güzel, yemekler de mükemmeldi",  # Positive
        # "Yemekler mükemmeldi, hizmet ise vasattı",  # Negative
        # "Garsonlar çok yardımcıydı ama yemekler çok tuzluydu",  # Negative
        "Fiyat/performans oranı çok yüksek, çok memnun kaldım",  # Positive
        # "Yemekler harikaydı, ancak servis biraz yavaştı",  # Negative
        # "Çalışanlar çok samimiydi, ancak yemekler beklediğimin altındaydı",  # Negative
        # "Restoranın atmosferi çok güzeldi ama yemekler pek iyi değildi",  # Negative
        # "Çok pahalı ama lezzetliydi, bir daha gelir miyim bilmiyorum",  # Negative
        # "Yemekler çok lezzetliydi, ama servis biraz daha hızlı olabilirdi",  # Negative
        "Fiyatlar çok yüksekti, ama yemekler harikaydı",  # Positive
        # "Servis harikaydı, yemekler ise oldukça kötüydü",  # Negative
        # "Yemekler çok soğuktu, bir daha asla gelmem",  # Negative
        # "Lezzetli yemekler, ancak servis biraz zayıftı",  # Negative
        # "Mekan çok şık ama yemekler sıradandı",  # Negative
        # "Restoranın iç tasarımı çok hoştu ama yemekler beklediğim gibi değildi",  # Negative
        "Burası çok pahalı, ancak yemeklerin lezzeti harika",  # Positive
        # "Yemekler beklediğimin çok altındaydı",  # Negative
        # "Yemekler lezzetliydi ama servis biraz yavaştı",  # Negative
        # "Burası kesinlikle bir daha gelmeye değmez, yemekler kötüydü",  # Negative
        "Yemekler harika, mekan da çok hoştu",  # Positive
        # "Yemekler çok tuzluydu, asla gelmem",  # Negative
        # "Hizmet çok iyiydi, ancak yemekler vasattı",  # Negative
        "Burası çok pahalı ama yemekler harikaydı",  # Positive
        "Servis çok hızlıydı, yemekler de güzeldi",  # Positive
        "Yemekler mükemmeldi, bir dahaki sefere daha fazla kişiyle geleceğim",  # Positive
        # "Mekanın atmosferi çok güzeldi, ama yemekler ortalama",  # Negative
        "Çalışanlar çok yardımcıydı, yemekler çok lezzetliydi",  # Positive
        "Restoran çok kalabalıktı ama yine de yemekler güzel",  # Positive
        "Yemekler gerçekten çok güzeldi, her şey mükemmeldi",  # Positive
        # "Çalışanlar çok nazikti, yemekler ise berbattı",  # Negative
        # "Yemekler çok tuzluydu, ancak atmosfer güzel",  # Negative
        # "Fiyat/performans oranı çok kötüydü",  # Negative
        # "Yemekler çok soğuktu, servis berbattı",  # Negative
        # "Harika yemekler, ancak servis biraz daha iyi olabilir",  # Negative
        "Burası kesinlikle pahalı ama yemekler güzeldi",  # Positive
        # "Atmosfer çok güzeldi ama yemekler beklentimi karşılamadı",  # Negative
        # "Yemekler harikaydı, fakat servis oldukça yavaştı",  # Negative
        # "Bu kadar kötü yemek yenir mi, hayal kırıklığına uğradım",  # Negative
        # "Restoran çok şık ama yemekler çok kötüydü",  # Negative
        "Harika bir deneyimdi, yemekler ve servis mükemmeldi",  # Positive
        "Yemekler oldukça ortalamaydı, ancak servis çok iyiydi",  # Negative
        "Restoran çok güzeldi, ama yemekler beklentimi karşılamadı",  # Negative
        "Yemekler mükemmeldi, ancak fiyatlar biraz yüksek",  # Positive
        "Yemekler harikaydı, ama garsonlar biraz daha hızlı olabilirdi",  # Negative
        "Atmosfer çok güzeldi, yemekler çok lezzetliydi",  # Positive
        "Burası pahalı ama yemekler güzeldi",  # Positive
        "Yemekler mükemmel, servis ise zayıftı",  # Negative
        "Mekanın iç tasarımı çok hoştu ama yemekler pek lezzetli değildi",  # Negative
        "Yemekler harika, ancak servis çok kötüydü",  # Negative
        "Restoranın dekorasyonu çok güzeldi ama yemekler kötüydü",  # Negative
        "Harika yemekler, ancak servis yavaştı",  # Negative
        "Servis mükemmeldi ama yemekler kötüydü",  # Negative
        "Burası çok pahalı, ama yemekler çok lezzetliydi",  # Positive
        "Yemekler harika, ancak fiyatlar çok yüksekti",  # Positive
        "Restoranın atmosferi çok hoştu, ancak yemekler beklediğimin altındaydı",  # Negative
        "Yemekler çok lezzetliydi, ancak servis biraz daha iyi olabilir",  # Negative
        "Burası pahalı ama yemekler gerçekten güzeldi",  # Positive
        "Yemekler mükemmeldi, ancak biraz daha hızlı servis edilseydi daha iyi olurdu",  # Negative
        "Fiyat/performans oranı çok düşüktü, yemekler kötüydü",  # Negative
        "Harika yemekler, ancak restoran çok kalabalıktı",  # Negative
        "Mekan çok güzel ama yemekler pek iyi değildi",  # Negative
        "Yemekler harikaydı ama servis çok yavaştı",  # Negative
        "Atmosfer çok güzeldi ama yemekler vasattı",  # Negative
        "Çalışanlar çok yardımcıydı ama yemekler kötüydü",  # Negative
        "Burası gerçekten çok pahalı, yemekler beklediğimin çok altındaydı",  # Negative
        "Yemekler mükemmeldi, ancak servis biraz daha hızlı olabilirdi",  # Negative
        "Restoran harikaydı, ancak yemekler pek iyi değildi",  # Negative
        "Harika yemekler, fakat servis biraz yavaştı",  # Negative
        "Burası pahalı ama yemekler güzeldi"  # Positive
    ],
    "label": [
        	"positive",
        	# "negative",
        	"positive",
        	# "negative",
        	"positive",
        	# "negative",
        	"positive",
        	# "negative",
        	"positive",
        	# "negative",
        	"positive",
        	# "negative",
        	"positive",
        	"positive",
        	# "negative",
        	"positive",
        	"positive",
        	"positive",
        	"positive",
        	# "negative",
        	"positive",
        	"positive",
        	"positive",
        	# "negative",
        	# "negative",
        	# "negative",
        	# "negative",
        	"positive",
        	"positive",
        	"positive",
        	"positive",
        	"positive",
        	# "negative",
        	# "negative",
        	"positive",
        	# "negative",
        	# "negative",
        	# "negative",
        	# "negative",
        	# "negative",
        	"positive",
        	# "negative",
        	# "negative",
        	# "negative",
        	# "negative",
        	# "negative",
        	"positive",
        	# "negative",
        	# "negative",
        	# "negative",
        	"positive",
        	# "negative",
        	# "negative",
        	"positive",
        	"positive",
        	"positive",
        	# "negative",
        	"positive",
        	"positive",
        	"positive",
        	# "negative",
        	# "negative",
        	# "negative",
        	# "negative",
        	# "negative",
        	"positive",
        	# "negative",
        	# "negative",
        	# "negative",
        	# "negative",
        	"positive",
        	"negative",
        	"negative",
        	"positive",
        	"negative",
        	"positive",
        	"positive",
        	"negative",
        	"negative",
        	"negative",
        	"negative",
        	"negative",
        	"negative",
        	"positive",
        	"positive",
        	"negative",
        	"negative",
        	"positive",
        	"negative",
        	"negative",
        	"negative",
        	"negative",
        	"negative",
        	"negative",
        	"negative",
        	"negative",
        	"negative",
        	"negative",
        	"negative",
        	"positive"
        ]
}

df = pd.DataFrame(data)
# %% metin temizleme ve preprocessing: tokenization, padding, label encoding, train test split

# tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
word_index = tokenizer.word_index

# padding process
maxlen = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen = maxlen)
print(X.shape)

# label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% metin temsili: word embedding: word2vec
sentences = [text.split() for text in df["text"]]
word2vec_model = Word2Vec(sentences, vector_size=50, window=5, min_count=1) # UYARI

embedding_dim = 50
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

# %% modelling: build, train ve test RNN modeli

# build model
model = Sequential()

# embedding
model.add(Embedding(input_dim = len(word_index) + 1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False))

# RNN layer
model.add(SimpleRNN(50, return_sequences=False))

# output layer
model.add(Dense(1, activation="sigmoid"))

# compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# train model
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

# evaluate rnn model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")

# cumle siniflandirma calismasi
# Not: model bizden sayisal deger bekliyor. ancak cumle olarak gelecek.
def classify_sentence(sentence):

    seq = tokenizer.texts_to_sequences([sentence])
    padded_seq = pad_sequences(seq, maxlen = maxlen)
    
    prediction = model.predict(padded_seq)
    
    predicted_class = (prediction > 0.5).astype(int)
    label = "positive" if predicted_class[0][0] == 1 else "negative"
    
    return label

sentence = "Restaurant cok temizdi ve yemekler cok guzeldi"

result = classify_sentence(sentence)
print(f"Result: {result}")










