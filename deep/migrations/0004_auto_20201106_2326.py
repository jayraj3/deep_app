# Generated by Django 3.1.2 on 2020-11-06 22:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('deep', '0003_auto_20201106_2156'),
    ]

    operations = [
        migrations.AlterField(
            model_name='oimage',
            name='action',
            field=models.CharField(choices=[('1', 'Remove'), ('2', 'Blur'), ('3', 'Black and White')], default='Black and White', max_length=100),
        ),
    ]
