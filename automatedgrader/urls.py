from django.urls import path
from . import views

urlpatterns=[
    path("",views.index,name='index'),
    path("login",views.login_view,name="login"),
    path("logout",views.logout_view,name="logout"),
    path("register",views.register,name="register"),
    path("account",views.account,name="Account"),
    path("about",views.about,name="about"),
    path("teachers",views.teachers,name="teachers"),
    path("contact",views.contact,name="contact"),
    path("grade",views.grade,name="grade")
]