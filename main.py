from preprocessing_data import preprocessing_mytelkom
from classifier_model import klasifikasi_mytelkom
import streamlit as st


st.title('Klasifikasi Sentimen dari Ulasan Aplikasi MyTelkomsel')
st.write('Tentukan sentimen dari ulasan yang diperoleh ')


contoh_teks = st.selectbox(
    'Contoh teks review / ulasan aplikasi MyTelkomsel: ',
    ('Buka aplikasi lain jaringan nya bagus... Pas buka aplikasi TELKOMSEL knp di suruh cek jaringan mulu... Makin aneh nh aplikasi ini... Gmn mau ngecek pulsa dan kuota nya bosss.', 'Semoga tujuan kita sama2 lancar ya jaringan,', 'Sangat kecewa dgn versi terbaru. Mencoba masuk gagal terus. Tolong untuk bisa dibenahi'))

if contoh_teks != None:
    processed_example = preprocessing_mytelkom(contoh_teks)
    hasil_sentimen, persentase = klasifikasi_mytelkom(processed_example)
    st.write(f"Sentimen dari '{contoh_teks}' adalah: :blue[{hasil_sentimen}] dengan tingkat persentase = :blue[{persentase}]",
             )
else:
    print('Maaf, ada masalah')

st.subheader('Masukkan :blue[ulasan] yang anda peroleh ke bawah:')

ulasan = st.text_area('Teks Ulasan:', )

if st.button('Prediksi Sentimen'):
    processed_sentence = preprocessing_mytelkom(ulasan)
    # Jika hasil preprocessing berupa kalimat kosong
    if len(processed_sentence) == 0:
        st.write(f"Tidak bisa melakukan klasifikasi. Kalimat yang anda masukkan terlalu singkat atau hanya berisi angka, emoticon atau kata-kata stopword.")
        st.write(f"Ulasan anda: {ulasan}")
    else:
        hasil_sentimen_mytelkom, persentase_mytelkom = klasifikasi_mytelkom(
            processed_sentence)
        st.write(f"Teks yang dimasukkan:",
                 )
        st.write(f"'{ulasan}'")
        st.write(
            f"Sentimen: :blue[{hasil_sentimen_mytelkom}] dengan tingkat persentase = :blue[{persentase_mytelkom}]")

else:
    st.write('Belum ada ulasan yang masuk')
