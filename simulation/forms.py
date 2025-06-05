from django import forms

class KMCSimulationForm(forms.Form):
    lattice_size = forms.IntegerField(
        label="Lattice Size",
        min_value=1,
        max_value=100,
        initial=10,
        widget=forms.NumberInput(attrs={
            'class': 'input input-bordered w-full',
            'placeholder': 'Enter lattice size'
        })
    )
    
    steps = forms.IntegerField(
        label="Simulation Steps",
        min_value=1,
        max_value=100000,
        initial=100,
        widget=forms.NumberInput(attrs={
            'class': 'input input-bordered w-full',
            'placeholder': 'Enter number of steps'
        })
    )
    
    temperature = forms.FloatField(
        label="Temperature (K)",
        min_value=0.1,
        max_value=10000,
        initial=1000.0,
        widget=forms.NumberInput(attrs={
            'class': 'input input-bordered w-full',
            'placeholder': 'Enter temperature in Kelvin'
        })
    )

    impurity_concentration = forms.FloatField(
        label="Impurity Concentration (0 to 1)",
        min_value=0.0,
        max_value=1.0,
        initial=0.05,
        widget=forms.NumberInput(attrs={
            'class': 'input input-bordered w-full',
            'step': '0.01',
            'placeholder': 'e.g. 0.05'
        })
    )
