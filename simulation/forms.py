from django import forms

class KMCSimulationForm(forms.Form):
    lattice_size = forms.IntegerField(
        label="Lattice Size",
        min_value=5,
        max_value=25,
        initial=10,
        widget=forms.NumberInput(attrs={
            'class': 'input input-bordered w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-lg focus:ring-blue-500/20 transition duration-300 ease-in-out',
            'placeholder': 'Size (5-25)',
        })
    )
    
    steps = forms.IntegerField(
        label="Simulation Steps",
        min_value=100,
        max_value=10000,
        initial=1000,
        widget=forms.NumberInput(attrs={
            'class': 'input input-bordered w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-lg focus:ring-blue-500/20 transition duration-300 ease-in-out',
            'placeholder': 'Steps (100-10000)',
        }),
    )
    
    temperature = forms.FloatField(
        label="Temperature (°K)",
        min_value=300,
        max_value=2000,
        initial=1000,
        widget=forms.NumberInput(attrs={
            'class': 'input input-bordered w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-lg focus:ring-blue-500/20 transition duration-300 ease-in-out',
            'placeholder': 'Temperature (300-2000 K)',
        })
    )

    impurity_concentration = forms.FloatField(
        label="Impurity Concentration",
        min_value=0.0,
        max_value=0.3,
        initial=0.05,
        widget=forms.NumberInput(attrs={
            'class': 'input input-bordered w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-lg focus:ring-blue-500/20 transition duration-300 ease-in-out',
            'step': '0.01',
            'placeholder': 'Impurity (0-0.3)',
        })
    )

    run_sweep = forms.BooleanField(
        label="Run Temperature Sweep",
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'checkbox rounded border-gray-300 focus:ring focus:ring-blue-500/20',
        })
    )