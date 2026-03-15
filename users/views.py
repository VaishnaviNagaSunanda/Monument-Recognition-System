from django.http import JsonResponse
from django.shortcuts import render, HttpResponse
from django.contrib import messages
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from .forms import UserRegistrationForm
from .models import UserRegistrationModel


# ---------- LOAD MODEL ONCE ----------
MODEL_PATH = os.path.join(settings.BASE_DIR, "models", "trained_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)


# ---------- CLASS NAMES ----------
class_names = [
    'Ajanta Caves',
    'Chhota_Imambara',
    'alai_darwaza',
    'alai_minar',
    'basilica_of_bom_jesus',
    'charminar',
    'golden_temple',
    'tajmahal',
    'tanjavur temple',
    'victoria memorial'
]


# ---------- MONUMENT INFORMATION ----------
monument_info = {

    "Ajanta Caves": {
        "history": "The Ajanta Caves are 30 rock-cut Buddhist cave monuments in Maharashtra.",
        "3d_model_url": "/static/ar_models/ajanta.glb",
        "map_location": "https://maps.google.com/?q=Ajanta+Caves"
    },

    "alai_darwaza": {
        "history": "Built in 1311, the Alai Darwaza is the southern gateway of the Quwwat-ul-Islam Mosque in Delhi.",
        "3d_model_url": "/static/ar_models/alai_darwaza.glb",
        "map_location": "https://maps.google.com/?q=Alai+Darwaza"
    },

    "alai_minar": {
        "history": "An unfinished tower in the Qutb complex started by Alauddin Khalji.",
        "3d_model_url": "/static/ar_models/alai_minar.glb",
        "map_location": "https://maps.google.com/?q=Alai+Minar"
    },

    "basilica_of_bom_jesus": {
        "history": "UNESCO World Heritage Site in Goa, holds the remains of St. Francis Xavier.",
        "3d_model_url": "/static/ar_models/bom_jesus.glb",
        "map_location": "https://maps.google.com/?q=Basilica+of+Bom+Jesus"
    },

    "charminar": {
        "history": "Iconic 16th-century mosque in Hyderabad built by Muhammad Quli Qutb Shah.",
        "3d_model_url": "/static/ar_models/charminar.glb",
        "map_location": "https://maps.google.com/?q=Charminar"
    },

    "Chhota_Imambara": {
        "history": "Historical monument in Lucknow built by Muhammad Ali Shah in 1838.",
        "3d_model_url": "/static/ar_models/chhota_imambara.glb",
        "map_location": "https://maps.google.com/?q=Chhota+Imambara"
    },

    "golden_temple": {
        "history": "The holiest Gurdwara of Sikhism located in Amritsar.",
        "3d_model_url": "/static/ar_models/golden_temple.glb",
        "map_location": "https://maps.google.com/?q=Golden+Temple"
    },

    "tajmahal": {
        "history": "Famous white marble mausoleum built by Shah Jahan in Agra.",
        "3d_model_url": "/static/ar_models/tajmahal.glb",
        "map_location": "https://maps.google.com/?q=Taj+Mahal"
    },

    "tanjavur temple": {
        "history": "Known for its grand architecture, Brihadeeswarar Temple is a UNESCO World Heritage site.",
        "3d_model_url": "/static/ar_models/tanjavur_temple.glb",
        "map_location": "https://maps.google.com/?q=Tanjavur+Temple"
    },

    "victoria memorial": {
        "history": "Large marble building in Kolkata, built in honor of Queen Victoria.",
        "3d_model_url": "/static/ar_models/victoria_memorial.glb",
        "map_location": "https://maps.google.com/?q=Victoria+Memorial"
    },
}


# ---------- USER REGISTER ----------
def UserRegisterActions(request):

    if request.method == 'POST':

        form = UserRegistrationForm(request.POST)

        if form.is_valid():

            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()

            return render(request, 'UserRegistrations.html', {'form': form})

        else:
            messages.success(request, 'Email or Mobile Already Existed')

    else:
        form = UserRegistrationForm()

    return render(request, 'UserRegistrations.html', {'form': form})


# ---------- USER LOGIN ----------
def UserLoginCheck(request):

    if request.method == "POST":

        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')

        try:

            check = UserRegistrationModel.objects.get(
                loginid=loginid,
                password=pswd
            )

            status = check.status

            if status == "activated":

                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email

                return render(request, 'users/UserHomePage.html')

            else:

                messages.success(request, 'Your Account Not Activated')
                return render(request, 'UserLogin.html')

        except:
            messages.success(request, 'Invalid Login id and password')

    return render(request, 'UserLogin.html')


# ---------- USER HOME ----------
def UserHome(request):
    return render(request, 'users/UserHomePage.html')


# ---------- BROWSER PREDICTION ----------
@csrf_exempt
def prediction(request):

    if request.method == "GET":
        return render(request, "users/monument_prediction.html")

    if request.method == "POST" and 'monument_image' in request.FILES:

        uploaded_file = request.FILES['monument_image']

        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_path = fs.path(file_path)

        img = image.load_img(full_path, target_size=(300, 300))
        img_array = image.img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_batch)

        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) * 100

        predicted_class = class_names[predicted_index]

        info = monument_info.get(predicted_class, {})

        return render(request, "users/monument_prediction.html", {

            "uploaded_file_url": fs.url(file_path),
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "history": info.get("history", ""),
            "map_link": info.get("map_location", ""),
            "model_3d": info.get("3d_model_url", "")

        })

    return JsonResponse({"error": "Invalid request"})


# ---------- MOBILE API ----------
@csrf_exempt
def api_predict(request):

    if request.method == "POST" and 'monument_image' in request.FILES:

        uploaded_file = request.FILES['monument_image']

        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_path = fs.path(file_path)

        img = image.load_img(full_path, target_size=(300, 300))
        img_array = image.img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_batch)

        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) * 100

        predicted_class = class_names[predicted_index]

        info = monument_info.get(predicted_class, {})

        return JsonResponse({

            "uploaded_image": fs.url(file_path),
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "history": info.get("history", ""),
            "map_link": info.get("map_location", ""),
            "model_3d": info.get("3d_model_url", "")

        })

    return JsonResponse({"error": "Invalid request"})