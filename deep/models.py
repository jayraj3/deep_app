from django.db import models
from string import Template
from django.utils.safestring import mark_safe
from django_resized import ResizedImageField
# Create your models here.
# models.py 

# class PictureWidget(models.forms.widgets.Widget):
#     def render(self, name, value, attrs=None, **kwargs):
#         html =  Template("""<img src="$link"/>""")
#         return mark_safe(html.substitute(link=value))

class OImage(models.Model): 
	#name = models.CharField(max_length=50) 
	Image =  ResizedImageField(size=[1024, 768], upload_to='images/', blank=True, null=True, quality=50)#models.ImageField(upload_to='images/')
	CHOICES = (
		('1', 'Remove'),
		('2', 'Blur'),
		('3', 'Black and White'),
	)
	action = models.CharField(max_length=100, default=3, choices=CHOICES)