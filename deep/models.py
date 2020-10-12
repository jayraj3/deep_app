from django.db import models

# Create your models here.
# models.py 
class OImage(models.Model): 
	#name = models.CharField(max_length=50) 
	Img = models.ImageField(upload_to='images/') 
