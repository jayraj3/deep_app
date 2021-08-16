from django.urls import path, re_path
from django.conf import settings
from django.conf import settings
from django.views.static import serve


from . import views
app_name = 'deep'
urlpatterns = [
    re_path(r'^media/(?P<path>.*)$', serve,{'document_root': settings.MEDIA_ROOT}),# Serve static file
    re_path(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}),
    path('index', views.index, name='index'),
    path('', views.OImageCreate.as_view(), name='image_upload'),
    path('image/', views.OImageList.as_view(), name='image_view')
]

