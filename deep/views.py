from django.views.generic.edit import CreateView
from django.shortcuts import render
from django.http import HttpResponse
from django.urls import reverse_lazy
from django.views import View



from deep.models import OImage

def index(request):
    return render(request,'deep/index.html')


class OImageList(View):
    def get(self,request):
        im = OImage.objects.all().last()
        ctx = {'images': im}
        return render(request, 'deep/OImage_list.html', ctx)


class OImageCreate(CreateView):
    model = OImage
    fields = '__all__'
    success_url = reverse_lazy('deep:image_view')


