from django.contrib.auth import authenticate, login, logout
from django.db import IntegrityError
from django.http import HttpResponse, HttpResponseRedirect,JsonResponse
from django.shortcuts import render
from django.urls import reverse
import json
from django.views.decorators.csrf import csrf_exempt
from .models import User
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.core import serializers
from django.core.files.storage import FileSystemStorage
import os
from .apps import AutomatedgraderConfig

import collections
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import multiprocessing
from gensim.models import Word2Vec
import time


from sklearn.model_selection import KFold
import transformers as ppb
import pandas as pd
import numpy as np
import torch
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten, Input, Bidirectional,Conv2D, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Sequential, load_model, model_from_config, Model
import keras.backend as K
import warnings
from sklearn.metrics import cohen_kappa_score
warnings.filterwarnings('ignore')
import tensorflow as tf

# Create your views here.
def index(request):
   if request.user.is_authenticated:
        return render(request, "automatedgrader/index.html", { 
    })
   else:
    return HttpResponseRedirect(reverse("login"))


def login_view(request):
    if request.method == "POST":
        # Attempt to sign user in
        username = request.POST["username"] # username is the name of the input field in the form
        password = request.POST["password"] # password is the name of the input field in the form
        user = authenticate(request, username=username, password=password) # authenticate() is a built-in function provided by Django
        # If authentication successful, log user in
        if user is not None:
            login(request, user)
            return HttpResponseRedirect(reverse("index"))
        # Else, return login page again with new context
        else:
            return render(request, "automatedgrader/login.html", {
                "message": "Invalid email or password !!"
            })
    # If method is GET, return login page
    else:
        return render(request, "automatedgrader/login.html")

    
def logout_view(request):
    logout(request)
    return render(request, "automatedgrader/logout.html")

def register(request):
    if request.method == "POST":
        # Get username, email, password and confirmation password from the form
        username = request.POST["username"] # username is the name of the input field in the form
        email = request.POST["email"] # email is the name of the input field in the form
        password = request.POST["password"] # password is the name of the input field in the form
        confirmation = request.POST["confirmation"] # confirmation is the name of the input field in the form
        # Ensure password matches confirmation
        if password != confirmation:
            return render(request, "automatedgrader/register.html", {
                "message": "Passwords must match."
            })
        # Attempt to create new user
        try:
            user = User.objects.create_user(username, email, password) # create_user() is a built-in function provided by Django
            user.save()
        except IntegrityError:
            return render(request, "automatedgrader/register.html", {
                "message": "Username already taken."
            })
        login(request, user)
        return HttpResponseRedirect(reverse("index"))
    # If method is GET, return register page
    else:
        return render(request, "automatedgrader/register.html")

@login_required
def account(request):
    if request.method=="GET":
        if request.content_type == 'application/json':
            return JsonResponse(request.user.serialize())
        else:
            return render(request, "automatedgrader/account.html",{
                "User": request.user
            })
        
def about(request):
    return render(request, "automatedgrader/about.html")
def teachers(request):
    return render(request, "automatedgrader/teachers.html")
def contact(request):
    return render(request, "automatedgrader/contact.html")


@login_required
def grade(request):
        if request.method == 'POST':
            

            top10 = collections.defaultdict(int)

            

            def essay_to_wordlist(essay_v, remove_stopwords):
                """Remove the tagged labels and word tokenize the sentence."""
                essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
                words = essay_v.lower().split()
                #top10 = collections.defaultdict(int)
                if remove_stopwords:
                    stops = set(stopwords.words("english"))
                    for word in words:
                        if word not in stops:
                            # words.append(w)
                            top10[word]+=1
                    words = [w for w in words if not w in stops]
                return (words)

            def essay_to_sentences(essay_v, remove_stopwords):
                """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
                tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                raw_sentences = tokenizer.tokenize(essay_v.strip())
                sentences = []
                for raw_sentence in raw_sentences:
                    if len(raw_sentence) > 0:
                        sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
                return sentences

            def makeFeatureVec(words, model, num_features):
                """Make Feature Vector from the words list of an Essay."""
                featureVec = np.zeros((num_features,),dtype="float32")
                num_words = 0.
                index2word_set = set(model.wv.index_to_key)
                for word in words:
                    if word in index2word_set:
                        num_words += 1
                        featureVec = np.add(featureVec,model.wv.get_vector(word))
                featureVec = np.divide(featureVec,num_words)
                return featureVec

            def getAvgFeatureVecs(essays, model, num_features):
                """Main function to generate the word vectors for word2vec model."""
                counter = 0
                essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")
                for essay in essays:
                    essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
                    counter = counter + 1
                return essayFeatureVecs

            def build_word2vec(train_sentences, num_workers, num_features, min_word_count, context,
                                downsampling):
                model = Word2Vec(workers=num_workers, vector_size=num_features, min_count=min_word_count, window=context,
                                sample=downsampling)
                # saving the word2vec model
                # model.wv.save_word2vec_format('word2vec_'+ str(fold_count) +'.bin', binary=True)
                cores = multiprocessing.cpu_count()
                print("\n {} cores using".format(cores))
                start_time = time.time()
                model.build_vocab(train_sentences, progress_per=10000)
                print('Time to build vocab using word2vec: {} sec'.format(time.time() - start_time))
                start_time = time.time()
                model.train(train_sentences, total_examples=model.corpus_count, epochs=30, report_delay=1)
                print('Time to train the word2vec model: {} mins'.format(time.time() - start_time))
                model.init_sims(replace=True)
                sorted_dic = sorted(top10.items(), key=lambda k: k[1], reverse=True)
                return model,sorted_dic
            # Handle direct text input
            essay_text = request.POST.get('essay', '').strip()
            
            # Handle file upload
            essay_file = request.FILES.get('file', None)
            if essay_file:
                fs = FileSystemStorage()
                filename = fs.save(essay_file.name, essay_file)
                uploaded_file_url = fs.url(filename)

                # Assuming it's a text file for simplicity. Add logic for .docx files if needed.
                file_path = os.path.join(fs.location, filename)
                with open(file_path, 'r') as file:
                    essay_text = file.read()

            # At this point, `essay_text` contains the essay text, either from the textarea or the uploaded file
            # Proceed with processing `essay_text` as needed for grading
            Hidden_dim1=300
            Hidden_dim2=64
            return_sequences = True
            dropout=0.5
            recurrent_dropout=0.4
            input_size=768
            activation='relu'
            optimizer = 'adam'
            loss_function = 'mean_squared_error'
            batch_size= 64
            epoch =70
            model_name = "BiLSTM"
            
            model_path = os.path.join(os.path.dirname(__file__), 'ml_model', 'model_set1.h5')
            lstm_model = load_model(model_path)
            model_class, tokenizer_class, pretrained_weights = (
                    ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
            tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            model = model_class.from_pretrained(pretrained_weights)

            i=[essay_text]
            x=pd.DataFrame(i)
            input_essays=x[0]
            tokenized_input = input_essays.apply(
                            (lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=55)))
            max_len = 0
            for i in tokenized_input.values:
                if len(i) > max_len:
                    max_len = len(i)
            padded_input = np.array([i + [0] * (max_len - len(i)) for i in tokenized_input.values])
            attention_mask_train = np.where(padded_input != 0, 1, 0)

            input_ids = torch.tensor(padded_input)
            attention_mask = torch.tensor(attention_mask_train)
            with torch.no_grad():
                last_hidden_states_input = model(input_ids, attention_mask=attention_mask)

            input_features = last_hidden_states_input[0][:, 0, :].numpy()
            input_x, input_y = input_features.shape
            inputDataVectors = np.reshape(input_features, (input_x, 1, input_y))
            #AutomatedgraderConfig.ready()
            y_pred =lstm_model.predict(inputDataVectors)
            y_pred= y_pred.tolist()

            # Remember to delete the file after processing if it's no longer needed
            if essay_file:
                os.remove(file_path)

            return JsonResponse({"grade": y_pred})
        else:
            return render(request, "automatedgrader/index.html")





