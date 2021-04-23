from deep.models import OImage
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit, Layout
from django.forms import ModelForm


class OImageForm(ModelForm):
    def __init__(self, *args, **kwargs):
        super(OImageForm, self).__init__(*args, **kwargs)

        self.helper = FormHelper()
        self.helper.form_method= "post"
        self.helper.add_input(Submit("submit", "submit"))

    class Meta:
        model = OImage
        fields = ['Image', 'action']