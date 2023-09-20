import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
passage = "vnrvjiet campus departments ai ds cse ds cys automobile engineering civil engineering computer science and engineering csbs cse ai ml iot electrical and electronics engineering electronics and communication engineering electronics and instrumentation engineering humanities and sciences information technology mechanical engineering notifications training placements hostel menu home administration academics research training placement campus campus life key links connect with us management audited statements dean academics dean examination evaluation dean innovation incubation entrepreneurship globalization dean research consultancy dean student progression director of advancement dean governing council organogram principal sections examination cell administrative office alumni ed cell grievance redressal cell iiic iqac minority cell paac cdc antiragging and antidrug cell staff insurance login policies rules code of ethics and conduct work at vnr overview academic council commitee curriculum methodologies features academic awards feedback actions academic verification departments ai ds cse ds cys automobile engineering civil engineering computer science and engineering csbs cse ai ml iot electrical and electronics engineering electronics and communication engineering electronics and instrumentation engineering humanities and sciences information technology mechanical engineering repository academic regulations circulars notices condonation forms list of holidays notifications syllabus books training placements skill development activities iqac about history objectives walkthrough facilites activity centre food courts hostel labs library sports transport media happenings testimonials professional chapter societies acm asme csi diurnalis gsdc ici ieee iei igbc isoi iste iete iucee sae turing hut vsi clubs art of living candleves nrithya tarang classical dance creative arts crescendo dramatrix livewire kritomedh n army nss scintillate social media stentorian vignana jyothi sahithi vanam vj theatro vj spectral pyramid vnrsf vnrsf electoral literacy club elc campus celebrations annual day convergence cultural day ecficio national engineers day icmacc international womens day international yoga day national mathematics day open house sintillashunz sports fest national science day national teachers day national technology day republic day independance day traditional day tedx vnrvjiet world ip day world environment day world water day alumni achievements close close home administration management audited statements dean academics dean examination evaluation dean innovation incubation entrepreneurship globalization dean research consultancy dean student progression director of advancement dean governing council organogram principal sections examination cell administrative office alumni ed cell grievance redressal cell iiic iqac minority cell paac cdc antiragging and antidrug cell staff insurance login policies rules code of ethics and conduct work at vnr academics departments ai ds cse ds cys automobile engineering civil engineering computer science and engineering csbs cse ai ml iot electrical and electronics engineering electronics and communication engineering electronics and instrumentation engineering humanities and sciences information technology mechanical engineering repository academic regulations circulars notices condonation forms list of holidays notifications syllabus books overview academic council commitee curriculum methodologies features academic awards feedback actions academic verification research training placement campus about history objectives walkthrough facilites activity centre food courts hostel labs library sports transport media happenings testimonials campus life professional chapter societies acm asme csi diurnalis gsdc ici ieee iei igbc isoi iste iete iucee sae turing hut vsi clubs art of living candleves nrithya tarang classical dance creative arts crescendo dramatrix livewire kritomedh n army nss scintillate social media stentorian vignana jyothi sahithi vanam vj theatro vj spectral pyramid vnrsf vnrsf electoral literacy club elc campus celebrations annual day convergence cultural day ecficio national engineers day icmacc international womens day international yoga day national mathematics day open house sintillashunz sports fest national science day national teachers day national technology day republic day independance day traditional day tedx vnrvjiet world ip day world environment day world water day key links connect with us home admission admissions vallurupalli nageswara rao vignana jyothi institute of engineering and technology vnrvjiet sponsored by vignana jyothi was established in 1995 with the permission of all india council for technical education aicte new delhi and the government of erstwhile andhra pradesh the institute is autonomous under ugc and is affiliated to jntuhyderabad and is recognised under section 2f and 12 b of ugc act 1956 the institute is accredited by naac with a grade rated diamond by qs igauge and is iso 90012015 certified the institute offers 14 btech programmes13 mtech programmes and 5 phd programmes all eligible undergraduate programmes and four postgraduate programmes are accredited by the national board of accreditation nba the tuition fees for all programmes are stipulated as per the government order by telangana state which under convener quota is 135000 per year for btech and 130000 per year for mtech programmes vnrvjiet follows the relevant guidelines from the government of telangana state for admissions parents students are strongly discouraged to not approach any agencies consultants that make fake commitments about admissions under management quota or any other pathway the institute and its administration management are not responsible for any false commitments given by thirdparty agents for further details contact admissions coordinator admissions drdkanthi sudha senior office assistant mrschsailaja skilled assistant grajkumar phone numbers 04023044364 email admissions2023vnrvjietin for international admissions click here admission news catb admissions september 8 2023 mtech research assistantship september 8 2023 admission notification for 1st year btech supernumerary quota foreign nationalsociciwgc category july 20 2023 application for foreign nationalsociciwgc category july 20 2023 programmes offered btech programmes offered mtech ts eamcet cutoff ranks admission enquiry register for the ay202324 admission procedure apply now programmes offered btech s no year estd name of the course intake 1 2021 artificial intelligence and data science 60 2 2010 automobile engineering 60 3 2001 civil engineering 120 4 2019 computer science and business systems 60 5 1995 computer science and engineering 240 6 2020 cse artificial intelligence and machine learning 180 7 2020 cse data science 180 8 2020 cse cyber security 60 9 2020 cse internet of things 60 10 1995 electrical and electronics engineering 120 11 1995 electronics and communication engineering 240 12 1999 electronics instrumentation engineering 120 13 1997 information technology 180 14 1995 mechanical engineering 120 ts eamcet cutoff ranks ts eamcet2022 ranks ts eamcet2021 ranks ts eamcet2020 ranks tab 13 content programmes offered mtech s no name of the course intake 1 advanced manufacturing systemsme 12 2 computer networks information securityit 12 3 cadcamme 06 4 computer science and engineeringcse 12 5 embedded systemsece 12 6 electronics instrumentationeie 06 7 geotechnical engineeringce 12 8 highway engineeringce 12 9 power electronics eee 12 10 power systemseee 12 11 structural engineering ce 18 12 software engineeringcse 12 13 vlsi systems designece 12 address vignana jyothi nagar pragathi nagar nizampet so hyderabad telangana india 500 090 get directions ph no 91040230427 585960 fax 9104023042761 email postboxvnrvjietacin visit enquire apply approvals and accreditations aicte approvals nba minutes of statutory bodies aicte mandatory disclosure naac dapdatajntuh accreditation status nirf jntuhprincipal and faculty details jntuh affiliation order 202223 ariia report teqipii activities ugc undertaking qs i gauge rating 2019 spotlight fee payment academic regulations examination cell alumni circulars notices media hostel training placement events ict copyrights 199522 vnrvjiet all rights reserved made with love socialight careers disclaimer privacy policy sitemap screen reader"

# passage = "The Eiffel Tower is a famous landmark in Paris, France..."
questions = ["What are the departments of vnrvjiet campus?", "What are the professional student chapters in vnrvjiet?"]
answers = ["ai ds cse ds cys automobile engineering civil engineering computer science and engineering csbs cse ai ml iot electrical and electronics engineering electronics and communication engineering electronics and instrumentation engineering humanities and sciences information technology mechanical engineering", "acm asme csi diurnalis gsdc ici ieee iei igbc isoi iste iete iucee sae turing hut vsi"]

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts([passage] + questions + answers)
vocab_size = len(tokenizer.word_index) + 1

# Encode text data
passage_sequences = tokenizer.texts_to_sequences([passage])[0]
# print(passage_sequences.shape)  # (batch_size, input_features)

# Reshape the input data to add a time_steps dimension
passage_sequences = np.expand_dims(passage_sequences, axis=1)

# Now, the shape should be (batch_size, time_steps, input_features)
# print(passage_sequences.shape)

question_sequences = tokenizer.texts_to_sequences(questions)
answer_sequences = tokenizer.texts_to_sequences(answers)

# Pad sequences
MAX_SEQUENCE_LENGTH = 2
passage_sequences = pad_sequences([passage_sequences], padding='post', maxlen=MAX_SEQUENCE_LENGTH)[0]
question_sequences = pad_sequences(question_sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
answer_sequences = pad_sequences(answer_sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)

# Build the model
embedding_dim = 256
hidden_units = 512

encoder_inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

decoder_inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

# Attention mechanism
attention = Attention()([decoder_outputs, encoder_outputs])
decoder_concat = tf.concat([decoder_outputs, attention], axis=-1)
# decoder_dense = Dense(vocab_size, activation='softmax')
# output = decoder_dense(decoder_concat)
# print(output.shape)

# TimeDistributed layer to predict two values at each time step
decoder_dense = Dense(2, activation='softmax')
output = decoder_dense(decoder_concat)                                                                                                                                              
print(output.shape)

model = Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
EPOCHS = 8
BATCH_SIZE = 16
model.fit([passage_sequences, question_sequences], answer_sequences, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Inference: Generate questions and answers using the trained model

# ...

# Post-processing: Convert tokenized sequences back to text
# ...

# Save the model for future use
model.save('qa_model.h5')
