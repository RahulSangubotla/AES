from django.db import models
from django.contrib.auth.models import AbstractUser
# Create your models here.

class User(AbstractUser):
   
    image = models.ImageField(upload_to='users', default='allfurnitures/users/default.jpg')
    mobile=models.CharField(max_length=10,blank=True)
    def serialize(self):
        return {
            "id": self.id,
            "username": self.username,
            "image": self.image.url,
            "email": self.email,
        }
    
