from django.db import models
from string import Template
from django.utils.safestring import mark_safe
# Create your models here.
# models.py 

# class PictureWidget(models.forms.widgets.Widget):
#     def render(self, name, value, attrs=None, **kwargs):
#         html =  Template("""<img src="$link"/>""")
#         return mark_safe(html.substitute(link=value))

class OImage(models.Model): 
	#name = models.CharField(max_length=50) 
	Img = models.ImageField(upload_to='images/') 
