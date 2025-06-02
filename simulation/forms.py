from django import forms

class KMCSimulationForm(forms.Form):
    lattice_size = forms.IntegerField(label='Lattice Size (N)', min_value=1, initial=10)
    steps = forms.IntegerField(label='Number of Steps', min_value=1, initial=100)
    temperature = forms.FloatField(label='Temperature', min_value=0.1, initial=300.0)
